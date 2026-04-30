"""Kokoro TTS 推理引擎单元测试 — G2P 增强版本。"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class TestKokoroTtsEngine:

    @pytest.fixture
    def engine(self, tmp_path):
        from server.services.inference.engine_kokoro_tts import KokoroTtsEngine
        return KokoroTtsEngine("Kokoro-82M", str(tmp_path))

    def test_engine_not_loaded_raises(self, engine):
        with pytest.raises(RuntimeError, match="尚未加载"):
            engine.synthesize("hello", "zf_001")

    def test_empty_text_raises(self, engine):
        engine._loaded = True
        engine._voices = {"zf_001": np.zeros(256, dtype=np.float32)}
        with pytest.raises(ValueError, match="不能为空"):
            engine.synthesize("  ", "zf_001")

    def test_unknown_voice_raises(self, engine):
        engine._loaded = True
        engine._voices = {"zf_001": np.zeros(256, dtype=np.float32)}
        with pytest.raises(ValueError, match="未知音色"):
            engine.synthesize("hello", "nonexistent")

    def test_list_voices(self, engine):
        engine._loaded = True
        engine._voices = {"zf_001": np.array([]), "zm_009": np.array([])}
        voices = engine.list_voices()
        assert "zf_001" in voices
        assert "zm_009" in voices

    def test_tokenize_phonemes(self, engine):
        """G2P 输出音素后，tokenizer 编码为 token IDs。"""
        engine._tokenizer_vocab = {"n": 1, " ": 2, "i": 3, "h": 4, "a": 5, "o": 6}
        tokens = engine._tokenize_phonemes("n i h ao")
        assert tokens.tolist() == [1, 2, 3, 2, 4, 2, 5, 6]

    def test_build_feed_dict_by_name(self, engine):
        engine._input_names = ["input_ids", "style_embedding", "speed"]
        tokens = np.array([[1, 2, 3]], dtype=np.int64)
        voice = np.zeros((1, 256), dtype=np.float32)
        speed_arr = np.array([1.0], dtype=np.float32)
        mask = np.ones((1, 3), dtype=np.int64)
        feed = engine._build_feed_dict(tokens, voice, speed_arr, mask)
        assert "input_ids" in feed
        assert "style_embedding" in feed
        assert "speed" in feed

    def test_build_feed_dict_fallback(self, engine):
        engine._input_names = ["a", "b", "c"]
        tokens = np.array([[1, 2, 3]], dtype=np.int64)
        voice = np.zeros((1, 256), dtype=np.float32)
        speed_arr = np.array([1.0], dtype=np.float32)
        mask = np.ones((1, 3), dtype=np.int64)
        feed = engine._build_feed_dict(tokens, voice, speed_arr, mask)
        assert len(feed) == 3
        assert "a" in feed
        assert "b" in feed
        assert "c" in feed

    def test_run_onnx(self, engine):
        engine._sess = MagicMock()
        engine._sess.run.return_value = [np.zeros(24000, dtype=np.float32)]
        engine._output_names = ["waveform"]
        engine._input_names = ["input_ids", "style_embedding", "speed"]

        tokens = [0, 1, 2, 0]
        voice = np.zeros(256, dtype=np.float32)
        audio = engine._run_onnx_sentence(tokens, voice, 1.0)
        assert audio.ndim == 1

    @patch("tokenizers.Tokenizer.from_file")
    @patch("onnxruntime.InferenceSession")
    def test_load_missing_model_file(self, mock_ort, mock_tok, engine):
        import asyncio
        with pytest.raises(FileNotFoundError):
            asyncio.run(engine.load())
