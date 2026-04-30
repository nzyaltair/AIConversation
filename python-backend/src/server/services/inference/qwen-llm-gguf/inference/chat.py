"""
chat.py - Qwen3.5 text chat engine
References ASR engine's architecture pattern, using llama.cpp for accelerated inference.
Supports Vulkan / CUDA / CPU backends, streaming/non-streaming generation,
thinking mode toggle, and performance metrics.
"""

import os
import time
import codecs
import numpy as np
from typing import List, Generator, Optional, Dict
from pathlib import Path

from .schema import ChatConfig, DecodeResult
from . import llama


class ChatEngine:
    """Qwen3.5 text chat engine (GGUF + llama.cpp acceleration)"""

    def __init__(self, config: ChatConfig):
        self.config = config
        model_path = os.path.join(config.model_dir, config.llm_fn)

        # Detect available GPU backends
        active_backend, available_backends = self._detect_backend(config.llm_backend)

        use_gpu = config.llm_backend != "cpu"
        gpu_layers = 0 if config.llm_backend == "cpu" else config.n_gpu_layers

        if config.verbose:
            llama.info(f"--- [ChatEngine] Initializing ---")
            llama.info(f"   Requested backend: {config.llm_backend}")
            llama.info(f"   Active GPU backend: {active_backend}")
            llama.info(f"   Available backends: {available_backends}")
            llama.info(f"   GPU layers: {gpu_layers} (-1=all)")
            llama.info(f"   Thinking mode: {'ON' if config.enable_thinking else 'OFF'}")

        self.model = llama.LlamaModel(
            model_path,
            n_gpu_layers=gpu_layers,
            use_gpu=use_gpu,
            backend=config.llm_backend
        )

        self.ctx = llama.LlamaContext(
            self.model,
            n_ctx=config.n_ctx,
            n_batch=config.n_batch,
            n_ubatch=config.n_ubatch,
            flash_attn=config.flash_attn,
        )

        self.ID_IM_START = self.model.token_to_id("<|im_start|>")
        self.ID_IM_END = self.model.token_to_id("<|im_end|>")
        self._active_backend = active_backend
        self._available_backends = available_backends
        self.last_stats: Dict = {}

        if config.verbose:
            llama.info(f"   Model dims: {self.model.n_embd}, ctx: {config.n_ctx}")
            llama.info(f"   im_start={self.ID_IM_START}, im_end={self.ID_IM_END}, eos={self.model.eos_token}")
            llama.info("--- [ChatEngine] Init done ---")

    @property
    def active_backend(self) -> str:
        return self._active_backend

    @property
    def available_backends(self) -> list:
        return list(self._available_backends)

    def _detect_backend(self, requested: str):
        """Detect available llama.cpp GPU backends from DLLs in bin/."""
        bin_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")
        available = []

        if os.path.exists(os.path.join(bin_dir, "ggml-vulkan.dll")) or \
           os.path.exists(os.path.join(bin_dir, "libggml-vulkan.so")):
            available.append("vulkan")
        if os.path.exists(os.path.join(bin_dir, "ggml-cuda.dll")) or \
           os.path.exists(os.path.join(bin_dir, "libggml-cuda.so")):
            available.append("cuda")
        if os.path.exists(os.path.join(bin_dir, "ggml-metal.dll")) or \
           os.path.exists(os.path.join(bin_dir, "libggml-metal.dylib")):
            available.append("metal")
        available.append("cpu")

        if requested == "auto":
            for b in ["cuda", "vulkan", "metal"]:
                if b in available:
                    return b, available
            return "cpu", available
        elif requested in available:
            return requested, available
        else:
            llama.warning(f"Backend '{requested}' not available (have: {available}), using auto")
            for b in ["cuda", "vulkan", "metal"]:
                if b in available:
                    return b, available
            return "cpu", available

    def _apply_chat_template(self, messages: List[dict], enable_thinking: bool = True) -> str:
        """Apply Qwen3.5 chat template.
        When enable_thinking=False, pre-close the <think> block to force
        the model to skip reasoning and answer directly."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        if not enable_thinking:
            parts.append("<think>\n\n<｜end▁of▁thinking｜>\n")
        return "".join(parts)

    def _build_prompt(self, prompt: str, system_prompt: str = "", enable_thinking: bool = True) -> str:
        """Build formatted prompt text from user input."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({"role": "system", "content": "You are a helpful assistant."})
        messages.append({"role": "user", "content": prompt})
        return self._apply_chat_template(messages, enable_thinking=enable_thinking)

    def chat(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_k: int = 40,
        top_p: float = 0.95,
        min_p: float = 0.05,
        repeat_penalty: float = 1.1,
        enable_thinking: Optional[bool] = None,
    ) -> str:
        """Non-streaming chat. Returns complete response string.
        Performance stats are stored in self.last_stats."""
        chunks = []
        for chunk in self.stream_chat(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repeat_penalty=repeat_penalty,
            enable_thinking=enable_thinking,
        ):
            chunks.append(chunk)
        return "".join(chunks)

    def stream_chat(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_k: int = 40,
        top_p: float = 0.95,
        min_p: float = 0.05,
        repeat_penalty: float = 1.1,
        enable_thinking: Optional[bool] = None,
    ) -> Generator[str, None, None]:
        """Streaming chat generation. Yields decoded text fragments.

        Performance stats stored in self.last_stats after exhaustion.
        enable_thinking: None = use config default, True/False = override.
        """
        do_think = self.config.enable_thinking if enable_thinking is None else enable_thinking

        formatted = self._build_prompt(prompt, system_prompt, enable_thinking=do_think)

        # Tokenize
        input_tokens = self.model.tokenize(formatted, add_special=False, parse_special=True)
        n_prefill = len(input_tokens)

        # Build batch with token IDs
        batch = llama.LlamaBatch(max(n_prefill + max_tokens, self.config.n_batch), 0, 1)
        batch.n_tokens = n_prefill
        for i, tid in enumerate(input_tokens):
            batch.token[i] = tid
            batch.pos[i] = i
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = 0
            batch.logits[i] = 1 if i == n_prefill - 1 else 0

        # Prefill
        self.ctx.clear_kv_cache()
        t_total_start = time.time()
        t_pre_start = time.time()
        self.ctx.decode(batch)
        prefill_time = time.time() - t_pre_start

        # Generation loop
        t_gen_start = time.time()
        ttft = prefill_time  # initial estimate, refined on first yielded token
        n_gen_tokens = 0
        text_decoder = codecs.getincrementaldecoder("utf-8")(errors='replace')
        sampler = llama.LlamaSampler(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repeat_penalty=repeat_penalty,
            seed=int(np.random.randint(0, 2**31 - 1)),
        )
        stable_tokens = []
        first_token_yielded = False

        last_token = sampler.sample(self.ctx)

        for _ in range(max_tokens):
            if last_token in [self.model.eos_token, self.ID_IM_END]:
                break

            if self.ctx.decode_token(last_token) != 0:
                break

            sampler.accept(last_token)
            stable_tokens.append(last_token)

            piece = text_decoder.decode(self.model.token_to_bytes(last_token))
            if piece:
                if not first_token_yielded:
                    first_token_yielded = True
                    ttft = time.time() - t_pre_start
                yield piece

            if len(stable_tokens) > 15:
                if len(set(stable_tokens[-15:])) <= 3:
                    if self.config.verbose:
                        llama.warning("[Abort] Repeat detected, aborting")
                    break

            last_token = sampler.sample(self.ctx)
            n_gen_tokens += 1

        gen_time = time.time() - t_gen_start
        total_time = time.time() - t_total_start

        # Flush remaining bytes
        final_piece = text_decoder.decode(b"", final=True)
        if final_piece:
            yield final_piece

        del sampler
        del batch

        # Store performance stats
        prefill_speed = n_prefill / prefill_time if prefill_time > 0 else 0
        gen_speed = n_gen_tokens / gen_time if gen_time > 0 else 0

        self.last_stats = {
            "backend": self._active_backend,
            "thinking": do_think,
            "n_prefill": n_prefill,
            "n_generate": n_gen_tokens,
            "t_prefill_s": round(prefill_time, 3),
            "t_generate_s": round(gen_time, 3),
            "t_total_s": round(total_time, 3),
            "ttft_ms": round(ttft * 1000, 1),
            "prefill_speed_tps": round(prefill_speed, 1),
            "gen_speed_tps": round(gen_speed, 1),
        }

        if self.config.verbose:
            llama.info(
                f"[Stats] prefill: {n_prefill}t ({prefill_time:.2f}s, {prefill_speed:.0f} t/s) | "
                f"generate: {n_gen_tokens}t ({gen_time:.2f}s, {gen_speed:.1f} t/s) | "
                f"TTFT: {ttft*1000:.0f}ms | total: {total_time:.2f}s"
            )
