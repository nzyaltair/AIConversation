"""Kokoro TTS 引擎增强功能测试 — G2P、语言检测、动态语速。"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class TestG2PPipeline:
    @pytest.fixture
    def mock_misaki(self):
        """模拟 misaki 库 G2P 功能。"""
        with patch("server.services.inference.engine_kokoro_tts.logger"):
            from server.services.inference.engine_kokoro_tts import KokoroTtsEngine
            engine = KokoroTtsEngine("test", "/mock")
            engine._zh_g2p = MagicMock()
            engine._zh_g2p.return_value = "n i3 h ao3 sh ih4 j ie4"
            engine._en_g2p = MagicMock()
            engine._en_g2p.return_value = "həˈloʊ wɜrld"
            yield engine

    def test_chinese_g2p(self, mock_misaki):
        result = mock_misaki._g2p("你好世界", "zh")
        assert "n i3" in result

    def test_english_g2p(self, mock_misaki):
        result = mock_misaki._g2p("hello world", "en")
        assert "həˈloʊ" in result

    def test_fallback_no_g2p(self):
        """G2P 不可用时回退到原始文本。"""
        with patch("server.services.inference.engine_kokoro_tts.logger"):
            from server.services.inference.engine_kokoro_tts import KokoroTtsEngine
            engine = KokoroTtsEngine("test", "/mock")
            engine._zh_g2p = None
            engine._en_g2p = None
            assert engine._g2p("test", "en") == "test"
            assert engine._g2p("测试", "zh") == "测试"


class TestComputeSpeed:
    def test_short_text(self):
        from server.services.inference.engine_kokoro_tts import KokoroTtsEngine
        assert KokoroTtsEngine._compute_speed(30) == pytest.approx(1.1, rel=1e-6)

    def test_medium_text(self):
        from server.services.inference.engine_kokoro_tts import KokoroTtsEngine
        assert KokoroTtsEngine._compute_speed(80) == pytest.approx(1.1, rel=1e-6)

    def test_long_text(self):
        from server.services.inference.engine_kokoro_tts import KokoroTtsEngine
        assert KokoroTtsEngine._compute_speed(150) == pytest.approx(0.9526, rel=1e-4)

    def test_very_long_text(self):
        from server.services.inference.engine_kokoro_tts import KokoroTtsEngine
        assert KokoroTtsEngine._compute_speed(300) == pytest.approx(0.88, rel=1e-6)


class TestSynthesize:
    def test_synthesize_with_mock_onnx(self, tmp_path):
        from server.services.inference.engine_kokoro_tts import KokoroTtsEngine
        from unittest.mock import patch

        (tmp_path / "model_q4.onnx").write_text("mock")
        (tmp_path / "tokenizer.json").write_text('{"model":{"vocab":{"n":1," ":2,"i":3,"h":4,"a":5,"o":6}}}')
        voices_dir = tmp_path / "voices"
        voices_dir.mkdir()
        emb = np.random.randn(512).astype(np.float32)
        emb.tofile(str(voices_dir / "zf_001.bin"))

        engine = KokoroTtsEngine("test", str(tmp_path))

        with patch.object(engine, "_init_g2p"), \
             patch("onnxruntime.InferenceSession") as mock_ort, \
             patch("tokenizers.Tokenizer") as mock_tok:
            mock_sess = MagicMock()
            mock_sess.run.return_value = [np.zeros(24000, dtype=np.float32)]
            mock_sess.get_inputs.return_value = [
                MagicMock(name="input", type="tensor(int64)"),
            ]
            mock_ort.return_value = mock_sess

            mock_tok_inst = MagicMock()
            mock_tok_inst.encode.return_value = MagicMock(ids=[1, 2, 3, 4, 5])
            mock_tok.from_file.return_value = mock_tok_inst

            import asyncio
            asyncio.run(engine.load())

            engine._zh_g2p = MagicMock(return_value="n i3 h ao3")
            engine._en_g2p = MagicMock(return_value="hello")

            result = engine.synthesize("你好", "zf_001")
            assert result.sample_rate == 24000
            assert result.audio.ndim == 1


class TestVoiceList:
    def test_empty_when_no_voices(self):
        with patch("server.services.inference.engine_kokoro_tts.logger"):
            from server.services.inference.engine_kokoro_tts import KokoroTtsEngine
            engine = KokoroTtsEngine("test", "/mock")
            engine._loaded = True
            assert engine.list_voices() == []
