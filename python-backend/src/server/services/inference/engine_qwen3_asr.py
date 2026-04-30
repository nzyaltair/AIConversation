"""
Qwen3-ASR 推理引擎（ONNX + GGUF 混合运行时）。

架构：音频 → FastWhisperMel → ONNX Frontend → ONNX Backend → 音频嵌入 →
      GGUF LLM 解码 → 文本转录（分块流式 + 滑动记忆）。
"""

from __future__ import annotations

import logging

import numpy as np

from server.services.inference import register_engine
from server.services.inference.base import AsrEngine, AsrResult
from server.services.inference.utils import preprocess_audio
from server.services.inference.qwen_asr import llama_binding as llama
from server.services.inference.qwen_asr.encoder import QwenAudioEncoder
from server.services.inference.qwen_asr.decoder import (
    AsrPromptBuilder,
    AsrDecoder,
    ChunkedAsrPipeline,
)

logger = logging.getLogger(__name__)


@register_engine("asr", "onnx+gguf")
class Qwen3AsrEngine(AsrEngine):
    """Qwen3-ASR 引擎（ONNX 编码器 + GGUF 解码器）。

    transcribe() 使用分块流式管线：每 40s 一块进行编码→解码，
    滑动窗口记忆跨块上下文。
    """

    def __init__(self, variant: str, model_dir: str) -> None:
        super().__init__(variant, model_dir)
        self._encoder: QwenAudioEncoder | None = None
        self._llm_model: llama.LlamaModel | None = None
        self._llm_ctx: llama.LlamaContext | None = None
        self._pipeline: ChunkedAsrPipeline | None = None
        # 模型文件名（与 seed_models.py 中注册的 repo 一致）
        self._frontend_fn = "qwen3_asr_encoder_frontend.int4.onnx"
        self._backend_fn = "qwen3_asr_encoder_backend.int4.onnx"
        self._llm_fn = "qwen3_asr_llm.q4_k.gguf"

    # ------------------------------------------------------------------
    async def load(self) -> None:
        import os

        frontend_path = str(self.model_dir / self._frontend_fn)
        backend_path = str(self.model_dir / self._backend_fn)
        llm_path = str(self.model_dir / self._llm_fn)

        for p in [frontend_path, backend_path, llm_path]:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"ASR 模型文件缺失: {p}")

        # 编码器
        self._encoder = QwenAudioEncoder(
            frontend_path=frontend_path,
            backend_path=backend_path,
            onnx_provider="AUTO",  # 自动选择 CUDA > DML > CPU
            dml_pad_to=40,
            verbose=False,
        )

        # LLM（默认 GPU，不可用时回退 CPU）
        try:
            self._llm_model = llama.LlamaModel(llm_path, n_gpu_layers=-1, use_gpu=True)
            logger.info("ASR LLM 解码器使用 GPU (Vulkan)")
        except Exception:
            logger.warning("ASR LLM GPU 加载失败，回退 CPU")
            self._llm_model = llama.LlamaModel(llm_path, n_gpu_layers=0, use_gpu=False)
        embd_table = llama.get_token_embeddings_gguf(llm_path)
        self._llm_ctx = llama.LlamaContext(self._llm_model, n_ctx=2048, n_batch=4096)

        # 解码管线
        prompt_builder = AsrPromptBuilder(self._llm_model, embd_table)
        decoder = AsrDecoder(self._llm_model, self._llm_ctx, rollback_num=5)
        self._pipeline = ChunkedAsrPipeline(
            encoder=self._encoder,
            prompt_builder=prompt_builder,
            decoder=decoder,
            chunk_size_sec=40.0,
            memory_chunks=1,
        )

        self._loaded = True
        logger.info("Qwen3-ASR 引擎加载完成: %s", self.variant)

    async def unload(self) -> None:
        self._encoder = None
        self._pipeline = None
        self._llm_ctx = None
        self._llm_model = None
        self._loaded = False
        logger.info("Qwen3-ASR 引擎已卸载: %s", self.variant)

    # ------------------------------------------------------------------
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000,
                   context: str = "", language: str = "",
                   temperature: float = 0.4) -> AsrResult:
        self._ensure_loaded()

        wav = preprocess_audio(audio, sample_rate, target_rate=16000)
        text, stats = self._pipeline.transcribe(  # type: ignore[union-attr]
            audio=wav,
            context=context,
            language=language or None,
            temperature=temperature,
        )

        # 统计信息放入首段 metadata
        segments: list[dict] = []
        if stats:
            segments.append({"performance": stats})

        return AsrResult(text=text, language=language or "zh", segments=segments)

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Qwen3AsrEngine 尚未加载，请先调用 load()")
