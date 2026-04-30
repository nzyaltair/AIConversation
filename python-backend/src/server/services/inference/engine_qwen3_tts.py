"""
Qwen3-TTS GGUF 引擎。

支持三种模型变体：
- Qwen3-TTS-0.6B-CustomVoice-gguf  (内置 9 种音色 + 可选 instruct)
- Qwen3-TTS-1.7B-CustomVoice-gguf  (内置 9 种音色 + 可选 instruct)
- Qwen3-TTS-1.7B-VoiceDesign-gguf  (纯 instruct 音色设计，无内置音色)
"""

from __future__ import annotations

import asyncio
import logging

from server.services.inference.base import TtsEngine, AudioResult
from server.services.inference import register_engine

logger = logging.getLogger(__name__)

# Qwen3-TTS CustomVoice 内置说话人（与 qwen3_tts_gguf 的 SPEAKER_MAP 一致）
_BUILTIN_SPEAKERS = [
    "Vivian", "Serena", "Uncle_Fu", "Ryan",
    "Aiden", "Ono_Anna", "Sohee", "Eric", "Dylan",
]


@register_engine("tts", "gguf")
class Qwen3TTSEngine(TtsEngine):
    """Qwen3-TTS GGUF 推理引擎。

    根据 variant 名称自动检测模型类型：
    - 包含 "VoiceDesign" → 音色设计模式（stream.design）
    - 否则 → 内置音色模式（stream.custom）
    """

    def __init__(self, variant: str, model_dir: str) -> None:
        super().__init__(variant, model_dir)
        self._tts_engine = None
        self._stream = None
        self._is_voice_design = "VoiceDesign" in variant

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def load(self) -> None:
        """加载 Qwen3-TTS 模型（GGUF + ONNX decoder）。"""

        def _load() -> None:
            from server.services.inference.qwen3_tts_gguf.inference import (
                TTSEngine,
                TTSConfig,
            )

            self._tts_engine = TTSEngine(
                model_dir=str(self.model_dir),
                onnx_provider="CPU",
                llm_use_gpu=True,
                chunk_size=12,
                verbose=False,
            )
            if not self._tts_engine.ready:
                raise RuntimeError(
                    f"Qwen3-TTS 引擎初始化失败: {self.variant}"
                )
            self._stream = self._tts_engine.create_stream(n_ctx=2048)

        await asyncio.to_thread(_load)
        self._loaded = True
        logger.info("Qwen3-TTS 引擎已加载: %s", self.variant)

    async def unload(self) -> None:
        """释放模型内存及子进程资源。"""

        def _unload() -> None:
            if self._stream is not None:
                self._stream.shutdown()
                self._stream = None
            if self._tts_engine is not None:
                self._tts_engine.shutdown()
                self._tts_engine = None

        await asyncio.to_thread(_unload)
        self._loaded = False
        logger.info("Qwen3-TTS 引擎已卸载: %s", self.variant)

    # ------------------------------------------------------------------
    # TtsEngine 接口
    # ------------------------------------------------------------------

    def list_voices(self) -> list[str]:
        self._ensure_loaded()
        if self._is_voice_design:
            return []
        return list(_BUILTIN_SPEAKERS)

    def synthesize(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        instruct: str | None = None,
    ) -> AudioResult:
        self._ensure_loaded()

        from server.services.inference.qwen3_tts_gguf.inference import TTSConfig

        config = TTSConfig(
            max_steps=400,
            temperature=0.6,
            sub_temperature=0.6,
            seed=42,
            sub_seed=45,
            streaming=False,
            enable_speaker=False,
        )

        if self._is_voice_design:
            instruct_text = instruct or voice
            result = self._stream.design(
                text=text,
                instruct=instruct_text,
                config=config,
            )
        else:
            speaker = voice if voice != "default" else _BUILTIN_SPEAKERS[0]
            result = self._stream.custom(
                text=text,
                speaker=speaker,
                instruct=instruct,
                config=config,
            )

        if result is None or result.audio is None:
            raise RuntimeError(
                f"Qwen3-TTS 合成失败: {self.variant}"
            )

        return AudioResult(audio=result.audio, sample_rate=24000)

    def synthesize_stream(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        instruct: str | None = None,
    ):
        """流式合成。当前实现将完整音频作为单个块输出。"""
        yield self.synthesize(text, voice, speed, instruct=instruct)

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Qwen3-TTS 引擎尚未加载")
