"""
共享推理工具 — FastWhisperMel 频谱提取、音频预处理、语言检测。
"""

from __future__ import annotations

import math
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mel 频谱提取（纯 numpy，无 librosa/numba 启动延迟）
# ---------------------------------------------------------------------------

class FastWhisperMel:
    """纯 NumPy Mel 频谱提取器，行为对齐 Whisper / Qwen3-Audio 前端。

    n_mels=128, sr=16000, n_fft=400, hop_length=160, Slaney scale.
    """

    def __init__(
        self,
        n_mels: int = 128,
        sr: int = 16000,
        n_fft: int = 400,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        norm: Optional[str] = "slaney",
        mel_scale: str = "slaney",
    ) -> None:
        self.n_fft = n_fft
        self.hop_length = 160
        self.n_mels = n_mels
        self.filters = self._make_filters(sr, n_fft, n_mels, f_min, f_max, norm, mel_scale)
        # Hann 窗
        self.window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(self.n_fft) / self.n_fft)

    # -- filter bank ---------------------------------------------------------

    @staticmethod
    def _hz_to_mel(freq: np.ndarray | float, scale: str) -> np.ndarray | float:
        if scale == "htk":
            return 2595.0 * np.log10(1.0 + freq / 700.0)
        # Slaney: linear below 1kHz, log above
        f_sp = 200.0 / 3
        mels = (freq - 0.0) / f_sp
        min_log_hz = 1000.0
        logstep = math.log(6.4) / 27.0
        min_log_mel = (min_log_hz - 0.0) / f_sp
        if isinstance(freq, np.ndarray):
            mask = freq >= min_log_hz
            mels[mask] = min_log_mel + np.log(freq[mask] / min_log_hz) / logstep
        elif freq >= min_log_hz:
            mels = min_log_mel + math.log(freq / min_log_hz) / logstep
        return mels

    @staticmethod
    def _mel_to_hz(mels: np.ndarray | float, scale: str) -> np.ndarray | float:
        if scale == "htk":
            return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
        f_sp = 200.0 / 3
        freqs = 0.0 + f_sp * mels
        min_log_hz = 1000.0
        logstep = math.log(6.4) / 27.0
        min_log_mel = (min_log_hz - 0.0) / f_sp
        if isinstance(mels, np.ndarray):
            mask = mels >= min_log_mel
            freqs[mask] = min_log_hz * np.exp(logstep * (mels[mask] - min_log_mel))
        elif mels >= min_log_mel:
            freqs = min_log_hz * math.exp(logstep * (mels - min_log_mel))
        return freqs

    @classmethod
    def _make_filters(
        cls, sr: int, n_fft: int, n_mels: int,
        f_min: float, f_max: float, norm: Optional[str], mel_scale: str,
    ) -> np.ndarray:
        n_freqs = n_fft // 2 + 1
        all_freqs = np.linspace(0, sr / 2, n_freqs)
        m_pts = np.linspace(
            cls._hz_to_mel(f_min, mel_scale),
            cls._hz_to_mel(f_max, mel_scale),
            n_mels + 2,
        )
        f_pts = cls._mel_to_hz(m_pts, mel_scale)
        f_diff = f_pts[1:] - f_pts[:-1]
        slopes = f_pts[np.newaxis, :] - all_freqs[:, np.newaxis]
        down = (-slopes[:, :-2]) / f_diff[:-1]
        up = slopes[:, 2:] / f_diff[1:]
        fb = np.maximum(0, np.minimum(down, up))
        if norm == "slaney":
            enorm = 2.0 / (f_pts[2:n_mels + 2] - f_pts[:n_mels])
            fb *= enorm[np.newaxis, :]
        return fb.astype(np.float32)

    # -- call ----------------------------------------------------------------

    def __call__(self, audio: np.ndarray, dtype: type = np.float32) -> np.ndarray:
        pad_len = self.n_fft // 2
        y = np.pad(audio, pad_len, mode="reflect")

        num_frames = 1 + (len(y) - self.n_fft) // self.hop_length
        shape = (self.n_fft, num_frames)
        strides = (y.itemsize, self.hop_length * y.itemsize)
        frames = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)

        spec = np.fft.rfft(frames * self.window[:, np.newaxis], axis=0)
        power = np.abs(spec) ** 2
        mel = np.dot(self.filters.T, power)
        log_mel = np.log10(np.maximum(mel, 1e-10))
        log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
        log_mel = (log_mel + 4.0) / 4.0

        n_frames_out = audio.shape[-1] // self.hop_length
        log_mel = log_mel[:, :n_frames_out]
        return log_mel.astype(dtype)


def get_feat_extract_output_lengths(input_lengths: int) -> int:
    """Qwen3 前端输出帧数计算 — 复刻官方 C++ 逻辑。"""
    leave = input_lengths % 100
    feat_len = (leave - 1) // 2 + 1
    out_len = ((feat_len - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return out_len


# ---------------------------------------------------------------------------
# 音频工具
# ---------------------------------------------------------------------------

def _apply_low_pass(audio: np.ndarray, cutoff_ratio: float) -> np.ndarray:
    """Windowed-sinc low-pass filter for anti-aliasing before downsampling."""
    window_size = 63
    kernel = np.sinc(2 * cutoff_ratio * (np.arange(window_size) - (window_size - 1) / 2))
    kernel *= np.hamming(window_size)
    kernel /= kernel.sum()
    return np.convolve(audio, kernel, mode='same').astype(np.float32)


def resample_linear(audio: np.ndarray, src_rate: int, tgt_rate: int) -> np.ndarray:
    """线性插值重采样，降采样前应用抗混叠低通滤波。"""
    if src_rate == tgt_rate:
        return audio.copy()
    if src_rate > tgt_rate:
        audio = _apply_low_pass(audio, (tgt_rate / src_rate) * 0.9)
    ratio = tgt_rate / src_rate
    target_len = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, target_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def preprocess_audio(audio: np.ndarray, sample_rate: int, target_rate: int = 16000) -> np.ndarray:
    """标准音频预处理：转为单声道 float32、重采样到 target_rate、归一化。"""
    wav = np.asarray(audio, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=0)
    wav = wav.flatten()
    if sample_rate != target_rate:
        wav = resample_linear(wav, sample_rate, target_rate)
    peak = np.abs(wav).max()
    if peak > 0:
        wav = wav / peak * 0.9
    return wav


# ---------------------------------------------------------------------------
# 语言工具
# ---------------------------------------------------------------------------

_SUPPORTED_LANGUAGES = frozenset({
    "zh", "en", "ja", "ko", "yue", "de", "fr", "es", "ru", "pt",
    "it", "ar", "nl", "sv", "pl", "tr", "uk", "vi", "th",
})

_LANG_ALIASES: dict[str, str] = {
    "chinese": "zh", "english": "en", "japanese": "ja", "korean": "ko",
    "cantonese": "yue", "german": "de", "french": "fr",
    "spanish": "es", "russian": "ru", "portuguese": "pt",
    "italian": "it", "arabic": "ar", "dutch": "nl", "swedish": "sv",
    "polish": "pl", "turkish": "tr", "ukrainian": "uk",
    "vietnamese": "vi", "thai": "th",
    "zh-cn": "zh", "zh-tw": "zh",
}


def normalize_language_name(name: str) -> str:
    """将语言名称归一化为 ISO 639-1 代码。"""
    key = name.strip().lower()
    return _LANG_ALIASES.get(key, key)


def validate_language(lang: str) -> None:
    """验证语言代码是否在支持的列表中，不支持则记录警告。"""
    if lang not in _SUPPORTED_LANGUAGES:
        logger.warning("不支持的语言代码 '%s'，推理可能降级", lang)


def detect_language(text: str) -> str:
    """启发式语言检测：含 CJK 字符的判定为中文，其余英文。"""
    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
            return "zh"
        if 0x3040 <= cp <= 0x30FF:
            return "ja"
        if 0xAC00 <= cp <= 0xD7AF:
            return "ko"
    return "en"
