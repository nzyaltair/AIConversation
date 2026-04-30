"""
音频解码工具函数，供 ASR / VAD 等端点共享复用。
"""

from __future__ import annotations

import io
import logging

import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

logger = logging.getLogger(__name__)


def decode_audio_bytes(audio_bytes: bytes, filename: str = "") -> tuple[np.ndarray, int]:
    """将原始音频字节解码为 (float32 mono ndarray, sample_rate)。

    优先使用 soundfile（WAV/FLAC/OGG），
    失败时回退到 pydub+ffmpeg（webm/opus/mp3/m4a 等）。
    """
    # 策略1：soundfile（快速，原生支持 WAV/FLAC/OGG）
    try:
        audio_np, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        logger.debug("soundfile decoded %s: %d samples @ %d Hz", filename or "?", len(audio_np), sr)
        return np.asarray(audio_np, dtype=np.float32), sr
    except Exception as exc:
        logger.info(
            "soundfile cannot decode %s (%s), trying pydub fallback",
            filename or repr(audio_bytes[:16]),
            exc,
        )

    # 策略2：pydub（通过 ffmpeg 处理 webm/opus/mp3/m4a 等）
    try:
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        raw = np.array(seg.get_array_of_samples()).astype(np.float32)
        if seg.channels > 1:
            raw = raw.reshape(-1, seg.channels).mean(axis=1)
        max_int = 2 ** (8 * seg.sample_width - 1)
        raw = raw / max_int
        logger.debug(
            "pydub decoded %s: %d samples @ %d Hz (%d channels, %d-bit)",
            filename or "?", len(raw), seg.frame_rate, seg.channels, 8 * seg.sample_width,
        )
        return raw, seg.frame_rate
    except CouldntDecodeError as exc:
        logger.error("pydub failed to decode audio: %s. Is ffmpeg installed?", exc)
        raise RuntimeError(
            "Failed to decode audio format. Ensure ffmpeg is installed for webm/opus support."
        ) from exc
    except Exception as exc:
        logger.error("pydub fallback also failed: %s", exc)
        raise RuntimeError(f"Failed to decode audio: {exc}") from exc
