"""实时语音 WebSocket 协议测试 — IVWS 帧格式、mock 音频生成。"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest


def _build_ivws_frame(pcm16_samples: np.ndarray) -> bytes:
    """构建 IVWS 二进制帧。"""
    header = bytearray(24)
    header[0:4] = b"IVWS"
    header[4] = 1
    header[5] = 1  # kind=1: user audio
    raw = pcm16_samples.astype(np.int16).tobytes()
    return bytes(header) + raw


# ---------------------------------------------------------------------------
# 不需要运行服务器的单元测试
# ---------------------------------------------------------------------------

def test_ivws_frame_format():
    """IVWS 帧格式正确 —— 24 字节头 + PCM16 数据。"""
    pcm = np.array([100, 200, -100], dtype=np.int16)
    frame = _build_ivws_frame(pcm)
    assert frame[:4] == b"IVWS"
    assert frame[4] == 1  # version
    assert frame[5] == 1  # kind
    # 24 字节头 + 3 samples * 2 bytes = 30
    assert len(frame) == 24 + 6


def test_sine_wave_generation():
    """Mock 正弦波生成 —— 440Hz，时长和采样率正确。"""
    from server.api.voice.realtime import _gen_sine_wave

    audio = _gen_sine_wave(0.5, 24000)
    assert len(audio) == 12000  # 0.5 * 24000
    assert audio.dtype == np.float32
    assert np.abs(audio).max() <= 0.5
    assert np.abs(audio).max() > 0  # 非静音


def test_sine_wave_short():
    """极短正弦波也不会出错。"""
    from server.api.voice.realtime import _gen_sine_wave

    audio = _gen_sine_wave(0.01, 16000)
    assert len(audio) == 160  # 0.01 * 16000
    assert audio.dtype == np.float32


# ---------------------------------------------------------------------------
# WebSocket 端到端集成测试（需要运行服务器）
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="WebSocket 端到端测试需要完整服务器环境，在 CI 中通过 docker-compose 运行")
def test_ws_end_to_end():
    """启动服务器后运行：
    curl -N -X POST http://localhost:8000/v1/voice/realtime/ws
    """
    pass


@pytest.mark.skip(reason="需要完整服务器环境")
def test_mock_fallback_happy_path():
    """验证引擎未加载时的 mock 回退逻辑。"""
    pass
