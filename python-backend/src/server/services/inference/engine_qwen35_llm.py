"""
Qwen3.5 LLM GGUF 推理引擎。

通过 qwen-llm-gguf 包的 ChatEngine（llama.cpp ctypes 绑定）进行推理。
支持 Qwen3.5-0.8B 系列量化模型（Q4_K_M / Q8_0），Vulkan/CUDA/CPU 后端自动检测。
"""

from __future__ import annotations

import logging
import time
from importlib import import_module
from pathlib import Path
from typing import Generator

from server.services.inference import register_engine
from server.services.inference.base import LlmEngine, ChatResult

logger = logging.getLogger(__name__)


@register_engine("llm", "gguf")
class Qwen35LlmEngine(LlmEngine):
    """Qwen3.5 LLM GGUF 引擎。

    通过 qwen-llm-gguf/inference/chat.py 的 ChatEngine 加载 llama.cpp 模型，
    支持流式/非流式生成与 thinking 模式开关。
    """

    def __init__(self, variant: str, model_dir: str) -> None:
        super().__init__(variant, model_dir)
        self._engine = None  # ChatEngine 实例

    # ------------------------------------------------------------------
    async def load(self) -> None:
        _qwen_gguf = import_module(
            "server.services.inference.qwen-llm-gguf.inference"
        )
        ChatConfig = _qwen_gguf.ChatConfig
        ChatEngine = _qwen_gguf.ChatEngine

        llm_fn = f"{self.variant}.gguf"
        model_path = self.model_dir / llm_fn
        if not model_path.exists():
            raise FileNotFoundError(f"GGUF 模型文件不存在: {model_path}")

        config = ChatConfig(
            model_dir=str(self.model_dir),
            llm_fn=llm_fn,
            llm_backend="auto",
            n_ctx=2048,
            verbose=True,
            enable_thinking=True,
        )
        logger.info("加载 Qwen3.5 GGUF 模型: %s (backend=auto)", model_path)
        self._engine = ChatEngine(config)
        self._loaded = True
        logger.info(
            "Qwen3.5 LLM GGUF 引擎加载完成 (backend=%s, gpu_layers=%d)",
            self._engine.active_backend,
            config.n_gpu_layers,
        )

    async def unload(self) -> None:
        if self._engine is not None:
            del self._engine
            self._engine = None
        self._loaded = False

    # ------------------------------------------------------------------
    def generate(
        self,
        messages: list[dict],
        stream: bool = False,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs: object,
    ):
        """对话生成。非流式返回 ChatResult，流式返回生成器 yielding dict。"""
        self._ensure_loaded()

        enable_thinking = kwargs.get("enable_thinking", None)
        sys_prompt, prompt = self._extract_messages(messages)

        if stream:
            return self._stream_generate(prompt, sys_prompt, max_tokens,
                                         temperature, top_p, enable_thinking)
        else:
            return self._sync_generate(prompt, sys_prompt, max_tokens,
                                       temperature, top_p, enable_thinking)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if not self._loaded or self._engine is None:
            raise RuntimeError("Qwen35LlmEngine 尚未加载，请先调用 load()")

    @staticmethod
    def _extract_messages(messages: list[dict]) -> tuple[str, str]:
        """从消息列表中提取 system prompt 和用户对话内容。"""
        sys_prompt = ""
        user_parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                sys_prompt = content
            else:
                user_parts.append(content)
        prompt = "\n".join(user_parts) if user_parts else ""
        return sys_prompt, prompt

    def _sync_generate(
        self, prompt: str, sys_prompt: str,
        max_tokens: int, temperature: float, top_p: float,
        enable_thinking: bool | None,
    ) -> ChatResult:
        result = self._engine.chat(
            prompt=prompt,
            system_prompt=sys_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            enable_thinking=enable_thinking,
        )
        thinking, content = self._split_thinking(result)
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
                "finish_reason": "stop",
            }],
        )

    def _stream_generate(
        self, prompt: str, sys_prompt: str,
        max_tokens: int, temperature: float, top_p: float,
        enable_thinking: bool | None,
    ) -> Generator[dict, None, None]:
        created = int(time.time())
        chunk_id = f"chatcmpl-{created}"
        gen = self._engine.stream_chat(
            prompt=prompt,
            system_prompt=sys_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            enable_thinking=enable_thinking,
        )

        buf = ""
        state = "before_think"  # before_think | in_think | after_think
        think_buf = ""

        for piece in gen:
            buf += piece

            if state == "before_think":
                idx = buf.find("<think>")
                if idx != -1:
                    think_start = idx + len("<think>")
                    state = "in_think"
                    buf = buf[think_start:]
                elif len(buf) > 8:
                    # 没有 <think> 标签，直接输出
                    yield self._make_chunk(chunk_id, created, content=buf)
                    buf = ""
                    state = "after_think"

            if state == "in_think":
                end_idx = buf.find("</think>")
                if end_idx != -1:
                    think_buf += buf[:end_idx]
                    if think_buf.strip():
                        yield self._make_chunk(chunk_id, created, thinking=think_buf.strip())
                    think_buf = ""
                    buf = buf[end_idx + len("</think>"):]
                    state = "after_think"
                    if buf:
                        yield self._make_chunk(chunk_id, created, content=buf)
                        buf = ""
                else:
                    # 积累思考内容，每 32 字符输出一次
                    think_buf += buf
                    buf = ""
                    if len(think_buf) >= 32:
                        yield self._make_chunk(chunk_id, created, thinking=think_buf)
                        think_buf = ""

            if state == "after_think" and buf:
                yield self._make_chunk(chunk_id, created, content=buf)
                buf = ""

        # 刷新剩余的 buffer
        if state == "before_think" and buf:
            yield self._make_chunk(chunk_id, created, content=buf)
        elif state == "in_think" and think_buf.strip():
            yield self._make_chunk(chunk_id, created, thinking=think_buf.strip())

        # 发送结束标记
        yield {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.variant,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }

    @staticmethod
    def _make_chunk(
        chunk_id: str, created: int,
        content: str = "", thinking: str = "",
    ) -> dict:
        delta: dict = {}
        if thinking:
            delta["thinking"] = thinking
        if content:
            delta["content"] = content
        return {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": "",
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }

    @staticmethod
    def _split_thinking(text: str) -> tuple[str, str]:
        """将模型输出分离为 (thinking, content)。"""
        think_start = text.find("<think>")
        think_end = text.find("</think>")
        if think_start != -1 and think_end != -1:
            thinking = text[think_start + len("<think>"):think_end].strip()
            content = text[think_end + len("</think>"):].strip()
            return thinking, content
        return "", text
