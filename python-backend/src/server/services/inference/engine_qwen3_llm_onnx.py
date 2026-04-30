"""
Qwen3-0.6B ONNX LLM 推理引擎。

纯 ONNX 运行时 + 手动 KV Cache 管理，支持思考/非思考双模式切换。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import onnxruntime as ort

from server.services.inference import register_engine
from server.services.inference.base import LlmEngine, ChatResult

logger = logging.getLogger(__name__)

# Qwen3 特殊 token ID
BOS_TOKEN = 151643
EOS_TOKEN = 151645
THINK_START_TOKEN = 151667
THINK_END_TOKEN = 151668

# 模型架构参数（Qwen3-0.6B 固定）
NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128


@register_engine("llm", "onnx")
class Qwen3LlmOnnxEngine(LlmEngine):
    """Qwen3-0.6B ONNX 引擎。

    load() 加载 ONNX 模型 + HuggingFace tokenizer；
    generate() 执行自回归生成（含手动 KV Cache + 思考模式解析）。
    """

    def __init__(self, variant: str, model_dir: str) -> None:
        super().__init__(variant, model_dir)
        self._sess: ort.InferenceSession | None = None
        self._tokenizer = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        self._enable_thinking: bool = True

    # ------------------------------------------------------------------
    async def load(self) -> None:
        # CUDA DLL 预加载
        try:
            ort.preload_dlls()
        except (AttributeError, Exception):
            pass

        from transformers import AutoTokenizer

        model_path = str(self.model_dir / "model.onnx")
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"ONNX 模型不存在: {model_path}")

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        self._sess = ort.InferenceSession(model_path, sess_opts, providers=providers)
        self._input_names = [inp.name for inp in self._sess.get_inputs()]
        self._output_names = [out.name for out in self._sess.get_outputs()]
        logger.info("Qwen3 ONNX LLM: provider=%s, inputs=%s",
                    self._sess.get_providers()[0], self._input_names)

        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir), trust_remote_code=True, use_fast=True,
        )
        # 加载自定义 chat_template（若存在）
        tmpl_path = Path(self.model_dir) / "chat_template.jinja"
        if tmpl_path.is_file():
            self._tokenizer.chat_template = tmpl_path.read_text(encoding="utf-8")

        logger.info("Qwen3 ONNX LLM tokenizer vocab=%d", len(self._tokenizer))
        self._loaded = True

    async def unload(self) -> None:
        self._sess = None
        self._tokenizer = None
        self._loaded = False

    # ------------------------------------------------------------------
    def generate(self, messages: list[dict], stream: bool = False,
                 max_tokens: int = 512, temperature: float = 0.7,
                 top_p: float = 0.9, top_k: int = 20,
                 enable_thinking: bool | None = None):
        """对话生成。非流式返回 ChatResult，流式返回同步生成器 yielding dict。

        enable_thinking: 覆盖实例默认的思考模式开关（None=保持当前状态）。
        """
        self._ensure_loaded()

        # 临时切换思考模式
        prev_thinking = self._enable_thinking
        if enable_thinking is not None:
            self._enable_thinking = enable_thinking

        try:
            prompt_text = self._tokenizer.apply_chat_template(  # type: ignore[union-attr]
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=self._enable_thinking,
            )
            input_ids = self._tokenizer.encode(prompt_text, return_tensors="np")  # type: ignore[union-attr]

            if input_ids.shape[1] == 0:
                raise ValueError("tokenize 后输入为空，请检查 messages 格式")

            if stream:
                return self._stream_generate(input_ids, max_tokens, temperature, top_p, top_k)

            generated_ids, stats = self._autoregressive_generate(
                input_ids, max_tokens, temperature, top_p, top_k,
            )

            thinking, content = self._parse_thinking(generated_ids)
            full_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)  # type: ignore[union-attr]

            return ChatResult(
                id=f"chatcmpl-{int(time.time()*1000)}",
                created=int(time.time()),
                model=self.variant,
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "thinking": thinking,
                    },
                    "finish_reason": "stop" if len(generated_ids) < max_tokens else "length",
                }],
            )
        finally:
            self._enable_thinking = prev_thinking

    # ------------------------------------------------------------------
    # KV Cache
    # ------------------------------------------------------------------

    def _empty_kv_cache(self, batch_size: int = 1) -> Dict[str, np.ndarray]:
        kv: Dict[str, np.ndarray] = {}
        for i in range(NUM_LAYERS):
            kv[f"past_key_values.{i}.key"] = np.zeros(
                (batch_size, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float16,
            )
            kv[f"past_key_values.{i}.value"] = np.zeros(
                (batch_size, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float16,
            )
        return kv

    def _prepare_inputs(self, input_ids: np.ndarray,
                        kv_cache: Optional[Dict[str, np.ndarray]] = None
                        ) -> Dict[str, np.ndarray]:
        batch_size, seq_len = input_ids.shape
        if kv_cache is None:
            kv_cache = self._empty_kv_cache(batch_size)
        past_len = kv_cache[f"past_key_values.0.key"].shape[2]

        attn_mask = np.ones((batch_size, past_len + seq_len), dtype=np.int64)
        pos_ids = np.arange(past_len, past_len + seq_len, dtype=np.int64).reshape(1, -1)

        inputs: Dict[str, np.ndarray] = {
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": attn_mask,
            "position_ids": pos_ids,
        }
        inputs.update(kv_cache)
        return inputs

    @staticmethod
    def _extract_kv_cache(outputs: List[np.ndarray]) -> Dict[str, np.ndarray]:
        kv: Dict[str, np.ndarray] = {}
        for i in range(NUM_LAYERS):
            kv[f"past_key_values.{i}.key"] = outputs[1 + i * 2].astype(np.float16)
            kv[f"past_key_values.{i}.value"] = outputs[2 + i * 2].astype(np.float16)
        return kv

    # ------------------------------------------------------------------
    # 生成循环
    # ------------------------------------------------------------------

    def _autoregressive_generate(self, input_ids: np.ndarray, max_tokens: int,
                                  temperature: float, top_p: float, top_k: int
                                  ) -> Tuple[List[int], dict]:
        generated: list[int] = []
        kv_cache: Optional[Dict[str, np.ndarray]] = None
        current = input_ids

        t_start = time.time()
        first_token_time: Optional[float] = None

        for step in range(max_tokens):
            inputs = self._prepare_inputs(current, kv_cache)
            outputs = self._sess.run(self._output_names, inputs)  # type: ignore[union-attr]
            logits = outputs[0]
            kv_cache = self._extract_kv_cache(outputs)

            if temperature > 0:
                next_token = self._sample(logits, temperature, top_k, top_p)
            else:
                next_token = int(np.argmax(logits[0, -1, :]))

            if first_token_time is None:
                first_token_time = time.time()

            if next_token == EOS_TOKEN:
                break

            generated.append(next_token)
            current = np.array([[next_token]], dtype=np.int64)

        total_time = time.time() - t_start
        ftl = (first_token_time - t_start) if first_token_time and generated else 0

        return generated, {
            "total_tokens": len(generated),
            "first_token_latency": ftl,
            "total_time": total_time,
        }

    def _stream_generate(self, input_ids: np.ndarray, max_tokens: int,
                         temperature: float, top_p: float, top_k: int):
        """流式自回归生成器，逐 token yield OpenAI 兼容格式。"""
        kv_cache: Optional[Dict[str, np.ndarray]] = None
        current = input_ids
        created = int(time.time())
        past_think_end = False  # 是否已越过 <｜end▁of▁thinking｜> 分隔
        pending_thinking: list[int] = []  # 尚未输出的思考 token

        for step in range(max_tokens):
            inputs = self._prepare_inputs(current, kv_cache)
            outputs = self._sess.run(self._output_names, inputs)  # type: ignore[union-attr]
            logits = outputs[0]
            kv_cache = self._extract_kv_cache(outputs)

            if temperature > 0:
                next_token = self._sample(logits, temperature, top_k, top_p)
            else:
                next_token = int(np.argmax(logits[0, -1, :]))

            if next_token == EOS_TOKEN:
                yield {
                    "id": f"chatcmpl-{created}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": self.variant,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                return

            # 处理思考/回答分离
            delta_content = ""
            delta_thinking = ""

            if not past_think_end:
                pending_thinking.append(next_token)
                if next_token == THINK_END_TOKEN:
                    #  decode 思考部分
                    thinking_text = self._tokenizer.decode(  # type: ignore[union-attr]
                        pending_thinking, skip_special_tokens=True,
                    ).strip()
                    delta_thinking = thinking_text
                    pending_thinking.clear()
                    past_think_end = True
            else:
                # 回答部分的 token 逐字输出
                token_text = self._tokenizer.decode(  # type: ignore[union-attr]
                    [next_token], skip_special_tokens=True,
                )
                # 跳过纯空白的 token（BOS/EOS/特殊标记）
                if token_text.strip():
                    delta_content = token_text

            if delta_thinking or delta_content:
                delta: dict = {}
                if delta_thinking:
                    delta["thinking"] = delta_thinking
                if delta_content:
                    delta["content"] = delta_content
                if delta:
                    yield {
                        "id": f"chatcmpl-{created}",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": self.variant,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                    }

            current = np.array([[next_token]], dtype=np.int64)

        # 达到 max_tokens 上限
        yield {
            "id": f"chatcmpl-{created}",
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.variant,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
        }

    # ------------------------------------------------------------------
    # 采样
    # ------------------------------------------------------------------

    @staticmethod
    def _sample(logits: np.ndarray, temperature: float,
                top_k: int, top_p: float) -> int:
        logits_1d = logits[0, -1, :].astype(np.float32)
        if temperature <= 0:
            return int(np.argmax(logits_1d))
        logits_1d = logits_1d / temperature
        logits_1d = logits_1d - np.max(logits_1d)
        probs = np.exp(logits_1d)
        probs = probs / np.sum(probs)

        if top_k > 0:
            indices = np.argsort(probs)[-top_k:]
            mask = np.zeros_like(probs)
            mask[indices] = 1
            probs = probs * mask
            probs = probs / np.sum(probs)

        if top_p < 1.0:
            sorted_idx = np.argsort(probs)[::-1]
            sorted_p = probs[sorted_idx]
            cumsum = np.cumsum(sorted_p)
            cutoff = int(np.searchsorted(cumsum, top_p)) + 1
            keep = sorted_idx[:cutoff]
            mask = np.zeros_like(probs)
            mask[keep] = 1
            probs = probs * mask
            probs = probs / np.sum(probs)

        return int(np.random.choice(len(probs), p=probs))

    # ------------------------------------------------------------------
    # 思考模式解析
    # ------------------------------------------------------------------

    def _parse_thinking(self, token_ids: List[int]) -> Tuple[str, str]:
        try:
            idx = token_ids.index(THINK_END_TOKEN)
            thinking_ids = token_ids[:idx + 1]
            content_ids = token_ids[idx + 1:]
        except ValueError:
            thinking_ids = []
            content_ids = token_ids

        thinking = self._tokenizer.decode(thinking_ids, skip_special_tokens=True).strip()  # type: ignore[union-attr]
        content = self._tokenizer.decode(content_ids, skip_special_tokens=True).strip()  # type: ignore[union-attr]
        return thinking, content

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("LLM ONNX 引擎尚未加载")
