"""
IVWS（Interactive Voice WebSocket）二进制帧协议。

帧格式:
  [0:4]   Magic: b"IVWS" (4 bytes)
  [4]     Version: uint8 (1 byte)
  [5]     Kind: uint8 (1=user_audio, 2=assistant_audio)
  [6:24]  Reserved: 18 zero bytes
  [24:]   Payload: PCM16 little-endian audio data
"""

from __future__ import annotations

import math
import struct

import numpy as np

# 帧头常量
IVWS_HEADER_SIZE = 24
IVWS_MAGIC = b"IVWS"
IVWS_VERSION = 1

# 帧类型常量
KIND_USER_AUDIO = 1
KIND_ASSISTANT_AUDIO = 2

# 音频常量
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2       # PCM16 = 2 bytes per sample
CHUNK_SIZE = 160       # 10ms at 16kHz


def build_ivws_frame(kind: int, pcm16_data: bytes) -> bytes:
    """构建完整 IVWS 帧（24 字节头 + PCM16 负载）。

    Args:
        kind: 帧类型（1=用户音频, 2=助手音频）
        pcm16_data: PCM16 little-endian 音频字节

    Returns:
        完整的 IVWS 二进制帧
    """
    header = struct.pack("<4sBB", IVWS_MAGIC, IVWS_VERSION, kind)
    # 补齐到 24 字节（Magic=4 + Version=1 + Kind=1 + Reserved=18）
    header = header.ljust(IVWS_HEADER_SIZE, b"\x00")
    return header + pcm16_data


def parse_ivws_frame(data: bytes) -> tuple[int, bytes]:
    """解析 IVWS 帧，返回 (kind, payload)。

    Args:
        data: 完整 IVWS 帧（至少 24 字节）

    Returns:
        (kind, payload) 元组

    Raises:
        ValueError: Magic 不匹配或数据过短
    """
    if len(data) < IVWS_HEADER_SIZE:
        raise ValueError(
            f"IVWS 帧数据过短: {len(data)} < {IVWS_HEADER_SIZE}"
        )

    magic, version, kind = struct.unpack_from("<4sBB", data, 0)
    if magic != IVWS_MAGIC:
        raise ValueError(
            f"IVWS Magic 不匹配: 期望 {IVWS_MAGIC!r}, 收到 {magic!r}"
        )
    if version != IVWS_VERSION:
        raise ValueError(
            f"不支持的 IVWS 版本: {version} (期望 {IVWS_VERSION})"
        )

    payload = data[IVWS_HEADER_SIZE:]
    return kind, payload


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """将 float32 音频 [-1, 1] 转换为 PCM16 little-endian 字节。

    Args:
        audio: float32 numpy 数组，值应在 [-1, 1] 范围内

    Returns:
        PCM16 little-endian 编码的字节数据
    """
    # 裁剪到 [-1, 1] 防止溢出
    clipped = np.clip(audio, -1.0, 1.0)
    # 量化到 int16
    pcm16 = (clipped * 32767).astype(np.int16)
    return pcm16.tobytes()


def pcm16_to_float32(data: bytes) -> np.ndarray:
    """将 PCM16 little-endian 字节转换为 float32 numpy 数组。

    Args:
        data: PCM16 little-endian 字节数据

    Returns:
        float32 numpy 数组，值在 [-1, 1] 范围内
    """
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    return samples / 32768.0


def resample_24k_to_16k(audio: np.ndarray) -> np.ndarray:
    """将 24kHz 音频使用多相 FIR 滤波器重采样到 16kHz（带 Kaiser 窗抗混叠）。

    与线性插值不同，多相 FIR 可防止高频混叠进入基带，
    并保留高频内容，消除"模糊"听感。

    Args:
        audio: 24kHz float32 音频数组

    Returns:
        16kHz float32 音频数组
    """
    if len(audio) == 0:
        return np.array([], dtype=np.float32)
    return _resample_poly(audio, up=2, down=3)


def _resample_poly(x: np.ndarray, up: int, down: int, window_size: int = 10) -> np.ndarray:
    """纯 numpy 多相 FIR 重采样，复刻 scipy.signal.resample_poly。

    使用 Kaiser 窗 (beta=5.0) 设计抗混叠 FIR 滤波器，
    通过多相分解高效实现有理数倍率重采样。
    """
    g = math.gcd(up, down)
    up //= g
    down //= g

    if up == down:
        return x.copy()

    max_rate = max(up, down)
    f_c = 1.0 / max_rate
    half_len = window_size * max_rate
    n_taps = 2 * half_len + 1

    t = np.arange(n_taps) - half_len
    h = np.sinc(f_c * t)

    beta = 5.0
    kaiser_win = np.i0(beta * np.sqrt(1 - (2 * t / (n_taps - 1)) ** 2)) / np.i0(beta)
    h = h * kaiser_win
    h = h * (up / np.sum(h))

    length_in = len(x)
    length_out = int(math.ceil(length_in * up / down))

    x_up = np.zeros(length_in * up + n_taps, dtype=np.float32)
    x_up[:length_in * up:up] = x

    y_full = np.convolve(x_up, h, mode='full')

    offset = (n_taps - 1) // 2
    y = y_full[offset: offset + length_in * up: down]

    return y[:length_out].astype(np.float32)
