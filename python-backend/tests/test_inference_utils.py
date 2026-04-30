"""shared inference utilities 单元测试 — FastWhisperMel、音频预处理、语言检测。"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest


class TestFastWhisperMel:
    @pytest.fixture
    def mel(self):
        from server.services.inference.utils import FastWhisperMel
        return FastWhisperMel()

    def test_output_shape(self, mel):
        """已知输入长度，验证输出 (128, expected_frames)。"""
        duration_sec = 2.0
        sr = 16000
        audio = np.random.randn(int(duration_sec * sr)).astype(np.float32)
        spec = mel(audio)
        assert spec.dtype == np.float32
        assert spec.shape[0] == 128
        expected_frames = len(audio) // 160  # hop_length=160
        assert spec.shape[1] == expected_frames

    def test_no_nan_no_inf(self, mel):
        """随机音频输入不产生 NaN 或 Inf。"""
        audio = np.random.randn(16000).astype(np.float32)
        spec = mel(audio)
        assert not np.any(np.isnan(spec))
        assert not np.any(np.isinf(spec))

    def test_silence_input(self, mel):
        """全零输入不产生 NaN。"""
        audio = np.zeros(16000, dtype=np.float32)
        spec = mel(audio)
        assert not np.any(np.isnan(spec))

    def test_short_audio(self, mel):
        """极短音频 (小于 n_fft) 仍能正常输出。"""
        audio = np.random.randn(200).astype(np.float32)
        spec = mel(audio)
        assert spec.shape[0] == 128
        assert spec.shape[1] > 0

    def test_value_range(self, mel):
        """输出值范围有限（已做归一化）。"""
        audio = np.random.randn(32000).astype(np.float32)
        spec = mel(audio)
        assert spec.min() >= -2.5
        assert spec.max() <= 2.5


class TestFeatExtractOutputLengths:
    def test_known_values(self):
        from server.services.inference.utils import get_feat_extract_output_lengths
        # 100 帧 → 13 帧输出
        assert get_feat_extract_output_lengths(100) == 13
        # 200 帧 → 26 帧
        assert get_feat_extract_output_lengths(200) == 26
        # 0 在取模时产生 0
        assert get_feat_extract_output_lengths(0) == 0


class TestPreprocessAudio:
    def test_mono_conversion(self):
        from server.services.inference.utils import preprocess_audio
        stereo = np.random.randn(2, 1600).astype(np.float32)
        result = preprocess_audio(stereo, 16000)
        assert result.ndim == 1

    def test_resample(self):
        from server.services.inference.utils import preprocess_audio
        audio = np.zeros(8000, dtype=np.float32)
        result = preprocess_audio(audio, 8000, target_rate=16000)
        assert len(result) == 16000

    def test_normalization(self):
        from server.services.inference.utils import preprocess_audio
        loud = np.ones(1600, dtype=np.float32) * 5.0
        result = preprocess_audio(loud, 16000)
        assert np.abs(result).max() <= 0.9 + 1e-5


class TestLanguageDetection:
    def test_chinese(self):
        from server.services.inference.utils import detect_language
        assert detect_language("你好世界") == "zh"
        assert detect_language("Hello，你好") == "zh"

    def test_english(self):
        from server.services.inference.utils import detect_language
        assert detect_language("Hello world") == "en"
        assert detect_language("12345") == "en"

    def test_japanese(self):
        from server.services.inference.utils import detect_language
        assert detect_language("こんにちは") == "ja"

    def test_korean(self):
        from server.services.inference.utils import detect_language
        assert detect_language("안녕하세요") == "ko"

    def test_normalize(self):
        from server.services.inference.utils import normalize_language_name
        assert normalize_language_name("Chinese") == "zh"
        assert normalize_language_name("zh-cn") == "zh"
        assert normalize_language_name("en") == "en"
