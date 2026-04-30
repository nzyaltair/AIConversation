"""
ASR（语音转文字）端点。

端点：POST /v1/audio/transcriptions
格式：兼容 OpenAI Whisper API
所有推理均使用本地 ASR 引擎（Qwen3AsrEngine），引擎未加载时返回 400 错误。
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import Response

from server.app_state import AppState
from server.error_handlers import invalid_request
from server.services.inference.base import AsrEngine
from server.api.audio.utils import decode_audio_bytes

logger = logging.getLogger(__name__)


def create_router(state: AppState) -> APIRouter:
    """创建 ASR 转录路由。"""
    router = APIRouter()

    @router.post("/transcriptions")
    async def transcribe_audio(
        file: UploadFile = File(...),
        model: str = Form("whisper-base"),
        language: str = Form(""),
        response_format: str = Form("json"),
        timestamp_granularities: str = Form("[]"),
    ):
        engine = state.get_asr_engine(model)
        if engine is None:
            await state.auto_load_engine(model)
            engine = state.get_asr_engine(model)

        if engine is None:
            raise invalid_request(
                f"ASR 模型 '{model}' 未加载。请先通过 "
                f"/v1/admin/models/{model}/download 下载模型，"
                f"然后通过 /v1/admin/models/{model}/load 加载模型。"
            )

        return await _real_transcribe(engine, file, language, response_format)

    async def _real_transcribe(
        engine: AsrEngine,
        file: UploadFile,
        language: str,
        response_format: str,
    ):
        """使用真实 ASR 引擎转录音频。"""
        audio_bytes = await file.read()

        if not audio_bytes:
            raise invalid_request("Empty audio file")

        try:
            audio_np, sr = decode_audio_bytes(audio_bytes, file.filename or "")
        except RuntimeError as exc:
            raise invalid_request(str(exc))

        dur = len(audio_np) / sr if sr > 0 else 0.0
        logger.info("Transcribing %s: %.1fs audio @ %d Hz", file.filename or "?", dur, sr)

        result = await asyncio.to_thread(
            engine.transcribe, audio_np, sr, language=language,
        )
        return _format_response(result.text, language, dur, result, response_format)

    return router


def _format_response(text: str, language: str, duration: float, result, fmt: str):
    """根据 response_format 格式化输出。"""
    if fmt == "text":
        return Response(content=text, media_type="text/plain")
    return {
        "text": text,
        "language": language,
        "duration": duration,
        "segments": getattr(result, 'segments', []),
    }
