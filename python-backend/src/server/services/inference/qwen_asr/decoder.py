# coding=utf-8
"""
Qwen3-ASR 解码管线 — Prompt 构造、LLM 解码器、分块流式管线。

将 qwen_asr_gguf QwenASREngine 的内部逻辑拆分为三个可组合类：
  - AsrPromptBuilder：构造 LLM 输入嵌入
  - AsrDecoder：自回归 LLM 解码（含重复熔断与回退重试）
  - ChunkedAsrPipeline：分块编码 → 解码 → 记忆更新的完整流水线
"""

from __future__ import annotations

import time
import codecs
import dataclasses
import logging
from collections import deque
from typing import List, Optional

import numpy as np

from server.services.inference.qwen_asr import llama_binding as llama

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据类型
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class DecodeResult:
    """LLM 单次解码输出。"""
    text: str = ""
    stable_tokens: list[int] = dataclasses.field(default_factory=list)
    t_prefill: float = 0.0
    t_generate: float = 0.0
    n_prefill: int = 0
    n_generate: int = 0
    is_aborted: bool = False


@dataclasses.dataclass
class ChunkSegment:
    """分片记忆及其物理时间坐标。"""
    idx: int
    audio_start: float
    audio_end: float
    text: str = ""


# ---------------------------------------------------------------------------
# AsrPromptBuilder
# ---------------------------------------------------------------------------

class AsrPromptBuilder:
    """构造 LLM 输入嵌入序列 — 将音频嵌入夹在文本 token 嵌入之间。"""

    def __init__(self, model: llama.LlamaModel,
                 embd_table: llama.LlamaEmbeddingTable) -> None:
        self.model = model
        self.embd_table = embd_table
        self.table = embd_table  # 兼容旧测试别名
        self.ID_IM_START = model.token_to_id("<|im_start|>")
        self.ID_IM_END = model.token_to_id("<|im_end|>")
        self.ID_AUDIO_START = model.token_to_id("<|audio_start|>")
        self.ID_AUDIO_END = model.token_to_id("<|audio_end|>")
        self.ID_ASR_TEXT = model.token_to_id("<asr_text>")

    def build(self, audio_embd: np.ndarray, prefix_text: str = "",
              context: Optional[str] = None,
              language: Optional[str] = None) -> np.ndarray:
        """构造完整的 Prompt Embedding。

        拼装顺序:
          prefix(系统 + 用户前缀) → audio_embd → suffix(指令 + 历史)

        Args:
            audio_embd: 编码器输出的音频嵌入 (T_audio, n_embd)
            prefix_text: 之前片段的已解码文本，用于上下文传播
            context: 系统提示词
            language: 目标语言标签，None 为自动检测

        Returns:
            full_embd: (N_total, n_embd) float32
        """
        def tk(t: str) -> list[int]:
            return self.model.tokenize(t)

        # 区块 A: 系统提示 + 用户前缀
        prefix_str = f"system\n{context or 'You are a helpful assistant.'}"
        prefix_tokens = (
            [self.ID_IM_START] + tk(prefix_str) + [self.ID_IM_END] +
            [self.ID_IM_START] + tk("user\n") + [self.ID_AUDIO_START]
        )

        # 区块 B: 音频之后的指令 + 助手头 + 历史文本
        suffix_head = "assistant\n"
        if language:
            suffix_head += f"language {language}"
        suffix_tokens = (
            [self.ID_AUDIO_END] + [self.ID_IM_END] +
            [self.ID_IM_START] + tk(suffix_head) + [self.ID_ASR_TEXT] + tk(prefix_text)
        )

        n_pre = len(prefix_tokens)
        n_aud = audio_embd.shape[0]
        n_suf = len(suffix_tokens)
        total_embd = np.zeros((n_pre + n_aud + n_suf, self.model.n_embd),
                              dtype=np.float32)

        total_embd[:n_pre] = self.embd_table[prefix_tokens]
        total_embd[n_pre:n_pre + n_aud] = audio_embd
        total_embd[n_pre + n_aud:] = self.embd_table[suffix_tokens]

        return total_embd


# ---------------------------------------------------------------------------
# AsrDecoder
# ---------------------------------------------------------------------------

class AsrDecoder:
    """LLM 自回归解码器 — Prefill + 自回归生成 + 重复熔断 + 加温重试。"""

    def __init__(self, model: llama.LlamaModel, ctx: llama.LlamaContext,
                 rollback_num: int = 5) -> None:
        self.model = model
        self.ctx = ctx
        self.rollback_num = rollback_num

    def decode(self, full_embd: np.ndarray, prefix_text: str = "",
               is_last_chunk: bool = False, temperature: float = 0.4,
               streaming: bool = True) -> DecodeResult:
        """执行单次 LLM 生成循环（物理推理）。

        Args:
            full_embd: Prompt 嵌入 (N_total, n_embd)
            prefix_text: 历史文本（用于跳过已输出内容）
            is_last_chunk: 是否为最后一个片段（决定是否 flush 延迟队列）
            temperature: 采样温度
            streaming: 是否流式输出

        Returns:
            DecodeResult 包含解码文本和性能统计
        """
        result = DecodeResult()

        total_len = full_embd.shape[0]
        pos_base = np.arange(0, total_len, dtype=np.int32)
        pos_arr = np.concatenate([
            pos_base, pos_base, pos_base, np.zeros(total_len, dtype=np.int32)
        ])
        batch = llama.LlamaBatch(max(total_len * 4, 8192), self.model.n_embd, 1)
        batch.set_embd(full_embd, pos=pos_arr)

        # Prefill
        self.ctx.clear_kv_cache()
        t_pre_start = time.time()
        self.ctx.decode(batch)
        prefill_time = time.time() - t_pre_start

        # Generation Loop
        t_gen_start = time.time()
        n_gen_tokens = 0
        display_queue: deque[int] = deque()
        stable_tokens: list[int] = []
        stable_text_acc = ""
        text_decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')

        seed = int(np.random.randint(0, 2 ** 31 - 1))
        sampler = llama.LlamaSampler(temperature=temperature, seed=seed)
        last_sampled_token = sampler.sample(self.ctx)

        for _ in range(512):  # 每片段最大生成 token 数
            if last_sampled_token in [self.model.eos_token, self.ID_IM_END]:
                break

            if self.ctx.decode_token(last_sampled_token) != 0:
                break

            display_queue.append(last_sampled_token)
            if len(display_queue) > self.rollback_num:
                ready_token = display_queue.popleft()
                stable_tokens.append(ready_token)
                piece = text_decoder.decode(
                    self.model.token_to_bytes(ready_token)
                )
                if piece:
                    stable_text_acc += piece

            # 熔断检查：连续 token 种类过少则判定为循环重复
            if len(stable_tokens) > 15:
                if len(set(stable_tokens[-15:])) <= 3:
                    result.is_aborted = True
                    break

            last_sampled_token = sampler.sample(self.ctx)
            n_gen_tokens += 1

        gen_time = time.time() - t_gen_start
        del sampler
        del batch

        # 最后一片需 flush 延迟队列
        if is_last_chunk and not result.is_aborted:
            while display_queue:
                t = display_queue.popleft()
                stable_tokens.append(t)
                piece = text_decoder.decode(self.model.token_to_bytes(t))
                if piece:
                    stable_text_acc += piece
            final_p = text_decoder.decode(b"", final=True)
            if final_p:
                stable_text_acc += final_p

        result.text = stable_text_acc
        result.stable_tokens = stable_tokens
        result.t_prefill = prefill_time
        result.t_generate = gen_time
        result.n_prefill = total_len
        result.n_generate = n_gen_tokens
        return result

    def safe_decode(self, full_embd: np.ndarray, prefix_text: str = "",
                    is_last_chunk: bool = False, temperature: float = 0.4,
                    streaming: bool = True, max_retries: int = 1) -> DecodeResult:
        """带熔断加温重试的高层解码封装。

        Args:
            max_retries: 最大重试次数（默认 1 次）
        """
        for _ in range(max_retries):
            res = self.decode(full_embd, prefix_text, is_last_chunk,
                              temperature, streaming=streaming)
            if not res.is_aborted:
                break
            temperature += 0.3
            res.text += "====解码有误，强制熔断===="
            logger.warning("ASR 解码熔断触发，加温 %.1f 重试", temperature)
        return res

    @property
    def ID_IM_END(self) -> int:
        return self.model.token_to_id("<|im_end|>")


# ---------------------------------------------------------------------------
# ChunkedAsrPipeline
# ---------------------------------------------------------------------------

class ChunkedAsrPipeline:
    """分块流式 ASR 管线 — 逐片编码 → 解码 → 记忆更新。"""

    SR = 16000

    def __init__(self, encoder, prompt_builder: AsrPromptBuilder,
                 decoder: AsrDecoder, chunk_size_sec: float = 40.0,
                 memory_chunks: int = 1) -> None:
        self.encoder = encoder
        self.prompt_builder = prompt_builder
        self.decoder = decoder
        self.chunk_size_sec = chunk_size_sec
        self.memory_chunks = memory_chunks

    def transcribe(self, audio: np.ndarray, context: str = "",
                   language: Optional[str] = None,
                   temperature: float = 0.4) -> tuple[str, dict]:
        """运行完整转录流水线。

        Args:
            audio: float32 单声道 (N_samples,)，采样率 16000Hz
            context: 系统提示词
            language: 语言标签（None = 自动）
            temperature: 采样温度

        Returns:
            (完整转录文本, 性能统计数据)
        """
        samples_per_chunk = int(self.chunk_size_sec * self.SR)
        total_len = len(audio)
        num_chunks = int(np.ceil(total_len / samples_per_chunk))
        total_duration = total_len / self.SR

        # 预定义所有分片的物理边界
        all_segments: list[ChunkSegment] = [
            ChunkSegment(
                idx=i,
                audio_start=i * self.chunk_size_sec,
                audio_end=min((i + 1) * self.chunk_size_sec, total_duration),
            ) for i in range(num_chunks)
        ]
        asr_memory: deque[tuple[np.ndarray, str]] = deque(
            maxlen=self.memory_chunks
        )
        total_full_text = ""

        stats = {
            "prefill_time": 0.0, "decode_time": 0.0,
            "prefill_tokens": 0, "decode_tokens": 0,
            "encode_time": 0.0,
        }
        t_main_start = time.time()

        for i in range(num_chunks):
            s = i * samples_per_chunk
            e = min((i + 1) * samples_per_chunk, total_len)
            chunk_data = audio[s:e].copy()

            # 不足一整块的补零
            if len(chunk_data) < samples_per_chunk:
                chunk_data = np.pad(
                    chunk_data, (0, samples_per_chunk - len(chunk_data)),
                    mode='constant',
                )

            # 编码
            audio_feature, enc_time = self.encoder.encode(chunk_data)
            stats["encode_time"] += enc_time
            was_last = (i == num_chunks - 1)

            # 构造 Prompt Embedding
            prefix_text = "".join([m[1] for m in asr_memory])
            combined_audio = np.concatenate(
                [m[0] for m in asr_memory] + [audio_feature], axis=0
            )
            full_embd = self.prompt_builder.build(
                combined_audio, prefix_text, context, language,
            )

            # 解码
            res = self.decoder.safe_decode(
                full_embd, prefix_text, was_last, temperature,
            )

            # 更新记忆
            all_segments[i].text = res.text
            asr_memory.append((audio_feature, res.text))
            total_full_text += res.text

            stats["prefill_tokens"] += res.n_prefill
            stats["prefill_time"] += res.t_prefill
            stats["decode_tokens"] += res.n_generate
            stats["decode_time"] += res.t_generate

        t_total = time.time() - t_main_start
        stats["total_time"] = t_total
        stats["audio_duration"] = total_duration
        stats["rtf"] = t_total / total_duration if total_duration > 0 else 0

        return total_full_text, stats
