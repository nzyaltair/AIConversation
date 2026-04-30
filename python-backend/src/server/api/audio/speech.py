"""
TTS（文字转语音）端点。

端点：POST /v1/audio/speech
所有推理均使用本地 TTS 引擎（KokoroTtsEngine），引擎未加载时返回 400 错误。
"""

from __future__ import annotations

import base64
import io
import struct
import json
import asyncio
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse, Response

from server.app_state import AppState
from server.error_handlers import invalid_request
from server.models.schemas import TTSRequest
from server.services.inference.base import TtsEngine

logger = logging.getLogger(__name__)


def create_router(state: AppState) -> APIRouter:
    """创建 TTS 语音合成路由。"""
    router = APIRouter()

    @router.post("/speech")
    async def generate_speech(body: TTSRequest):
        engine = state.get_tts_engine(body.model)
        if engine is None:
            await state.auto_load_engine(body.model)
            engine = state.get_tts_engine(body.model)

        if engine is None:
            raise invalid_request(
                f"TTS 模型 '{body.model}' 未加载。请先通过 "
                f"/v1/admin/models/{body.model}/download 下载模型，"
                f"然后通过 /v1/admin/models/{body.model}/load 加载，"
                f"或直接调用 Generate Speech 自动加载。"
            )

        voice = body.voice or _default_voice(engine)
        speed = body.speed or 1.0
        instruct = body.instruct

        if body.stream:
            return await _real_stream_speech(engine, body, voice, speed, instruct)

        result = await asyncio.to_thread(
            engine.synthesize, body.input, voice, speed, instruct=instruct,
        )
        wav = _numpy_to_wav(result.audio, result.sample_rate)
        return Response(
            content=wav,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"},
        )

    @router.get("/voices")
    async def list_voices(model: str = ""):
        """列出可用的 TTS 语音。"""
        target = model or "Kokoro-82M-v1.1-zh-ONNX-q4"
        engine = state.get_tts_engine(target)
        if engine is None:
            await state.auto_load_engine(target)
            engine = state.get_tts_engine(target)

        if engine is None:
            return {"voices": []}

        return {"voices": engine.list_voices()}

    async def _real_stream_speech(
        engine: TtsEngine, body: TTSRequest, voice: str, speed: float, instruct: str | None,
    ):
        """流式 TTS：逐句合成，每句通过 SSE 推送 WAV base64 分块。"""

        async def _stream():
            yield f"data: {json.dumps({'event': 'start', 'sample_rate': 24000, 'audio_format': 'wav'})}\n\n"
            for audio_result in engine.synthesize_stream(body.input, voice, speed, instruct=instruct):
                wav_chunk = await asyncio.to_thread(
                    _numpy_to_wav, audio_result.audio, audio_result.sample_rate,
                )
                b64 = base64.b64encode(wav_chunk).decode()
                yield f"data: {json.dumps({'event': 'chunk', 'audio': b64, 'sample_rate': audio_result.sample_rate})}\n\n"
            yield f"data: {json.dumps({'event': 'done'})}\n\n"

        return StreamingResponse(_stream(), media_type="text/event-stream")

    return router


def _default_voice(engine: TtsEngine) -> str:
    voices = engine.list_voices()
    return voices[0] if voices else "default"


def _numpy_to_wav(audio: "np.ndarray", sample_rate: int) -> bytes:
    """numpy float32 音频 → WAV bytes (PCM 16-bit)。"""
    import numpy as np
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    data_size = len(pcm) * 2
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<H", 1))
    buf.write(struct.pack("<H", 1))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * 2))
    buf.write(struct.pack("<H", 2))
    buf.write(struct.pack("<H", 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm.tobytes())
    return buf.getvalue()
