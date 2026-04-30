"""
VAD（语音活动检测）端点。

端点：POST /v1/audio/vad
当 VAD 引擎（FireRedVadEngine）已加载时使用真实推理。
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from fastapi import APIRouter, UploadFile, File, Form

from server.app_state import AppState
from server.api.audio.utils import decode_audio_bytes

logger = logging.getLogger(__name__)

_DEFAULT_VAD_MODEL = "FireRedVad-onnx"


def _resample_to_16khz(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """将音频重采样至 16kHz（线性插值），VAD 引擎要求 16kHz 输入。"""
    if orig_sr == 16000:
        return audio
    duration = len(audio) / orig_sr
    new_len = int(duration * 16000)
    old_idx = np.arange(len(audio), dtype=np.float64)
    new_idx = np.linspace(0, len(audio) - 1, new_len, dtype=np.float64)
    return np.interp(new_idx, old_idx, audio).astype(np.float32)


def create_router(state: AppState) -> APIRouter:
    """创建 VAD 检测路由。"""
    router = APIRouter()

    @router.post("/vad")
    async def detect_vad(
        file: UploadFile = File(...),
        model: str = Form(_DEFAULT_VAD_MODEL),
    ):
        audio_bytes = await file.read()

        if not audio_bytes:
            from server.error_handlers import invalid_request
            raise invalid_request("Empty audio file")

        try:
            audio_np, sr = decode_audio_bytes(audio_bytes, file.filename or "")
        except RuntimeError as exc:
            from server.error_handlers import invalid_request
            raise invalid_request(str(exc))

        audio_np = _resample_to_16khz(audio_np, sr)
        # FireRedVad 模型训练时使用 int16 幅值范围（±32768），
        # 而 decode_audio_bytes 返回 [-1, 1] 的 float32。
        # 需要恢复 int16 量级，使 FBank + CMVN 特征与训练分布匹配。
        audio_np = audio_np * 32767.0
        dur = len(audio_np) / 16000

        engine = state.get_vad_engine(model)
        if engine is None:
            await state.auto_load_engine(model)
            engine = state.get_vad_engine(model)

        if engine is None:
            from server.error_handlers import invalid_request
            raise invalid_request(
                f"VAD model '{model}' is not loaded. "
                f"Download and load the model first via /v1/admin/models/{model}/download "
                f"and /v1/admin/models/{model}/load."
            )

        logger.info("Running VAD on %s: %.1fs audio", file.filename or "?", dur)

        result = await asyncio.to_thread(engine.detect, audio_np, 16000)

        speech_total = sum(end - start for start, end in result.timestamps)
        speech_ratio = round(speech_total / dur, 4) if dur > 0 else 0.0

        return {
            "dur": result.dur,
            "timestamps": result.timestamps,
            "num_speech_segments": len(result.timestamps),
            "speech_duration_ratio": speech_ratio,
        }

    return router
