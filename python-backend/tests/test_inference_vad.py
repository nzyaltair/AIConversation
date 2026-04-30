"""FireRedVad 推理引擎单元测试"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class TestFireRedVadConfig:
    def test_default_config(self):
        from server.services.inference.engine_fire_red_vad import FireRedVadConfig
        cfg = FireRedVadConfig()
        assert cfg.speech_threshold == 0.4
        assert cfg.min_speech_frame == 20
        assert cfg.chunk_max_frame == 30000

    def test_invalid_threshold(self):
        from server.services.inference.engine_fire_red_vad import FireRedVadConfig
        with pytest.raises(ValueError):
            FireRedVadConfig(speech_threshold=1.5)
        with pytest.raises(ValueError):
            FireRedVadConfig(speech_threshold=-0.1)

    def test_invalid_min_speech_frame(self):
        from server.services.inference.engine_fire_red_vad import FireRedVadConfig
        with pytest.raises(ValueError):
            FireRedVadConfig(min_speech_frame=0)


class TestVadPostprocessor:
    @pytest.fixture
    def post(self):
        from server.services.inference.engine_fire_red_vad import (
            FireRedVadConfig, _VadPostprocessor,
        )
        return _VadPostprocessor(FireRedVadConfig())

    def test_silence_only(self, post):
        """全静音帧 → 全 0 决策。"""
        probs = np.zeros(100, dtype=np.float32)
        decisions = post.process(probs)
        assert all(d == 0 for d in decisions)

    def test_speech_only(self, post):
        """全语音帧 → 全 1 决策。"""
        probs = np.ones(100, dtype=np.float32)
        decisions = post.process(probs)
        assert any(d == 1 for d in decisions)

    def test_min_speech_filter(self, post):
        """过短的语音段被过滤。"""
        # 仅 3 帧语音，远低于 min_speech_frame=20
        probs = np.zeros(500, dtype=np.float32)
        probs[100:103] = 0.9
        decisions = post.process(probs)
        assert all(d == 0 for d in decisions)

    def test_decisions_to_timestamps(self, post):
        decisions = [0] * 100
        decisions[20:50] = [1] * 30  # 0.20s - 0.50s
        ts = post.decisions_to_timestamps(decisions)
        assert len(ts) == 1
        assert ts[0] == [0.2, 0.5]

    def test_multiple_segments(self, post):
        decisions = [0] * 300
        decisions[30:80] = [1] * 50
        decisions[150:200] = [1] * 50
        ts = post.decisions_to_timestamps(decisions)
        assert len(ts) == 2


class TestFireRedVadEngine:
    @pytest.fixture
    def mock_onnx(self):
        with patch("onnxruntime.InferenceSession") as mock:
            sess = MagicMock()
            inp1 = MagicMock()
            inp1.name = "feat"
            inp1.shape = (1, "T", 80)
            inp2 = MagicMock()
            inp2.name = "cache_0"
            inp2.shape = (1, 128, "C")
            sess.get_inputs.return_value = [inp1, inp2]
            out = MagicMock()
            out.name = "probs"
            sess.get_outputs.return_value = [out, MagicMock(name="out_cache_0")]
            sess.run.return_value = [np.random.rand(1, 1).astype(np.float32),
                                      np.zeros((1, 128, 19), dtype=np.float32)]
            mock.return_value = sess
            yield mock

    def test_engine_not_loaded_raises(self, tmp_path):
        from server.services.inference.engine_fire_red_vad import FireRedVadEngine
        engine = FireRedVadEngine("test", str(tmp_path))
        with pytest.raises(RuntimeError, match="尚未加载"):
            engine.detect(np.zeros(1600, dtype=np.float32))

    def test_detect_empty_audio(self, tmp_path, mock_onnx):
        from server.services.inference.engine_fire_red_vad import FireRedVadEngine
        import asyncio

        (tmp_path / "model.onnx").write_text("mock")
        (tmp_path / "cmvn.ark").write_bytes(
            b"\x02\x00\x00\x00\x02\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00"
        )

        engine = FireRedVadEngine("test", str(tmp_path))
        # patch _AudioFeat to return empty feats
        with patch.object(engine, '_feat') as mock_feat:
            mock_feat.extract.return_value = (np.zeros((0, 80), dtype=np.float32), 0.0)
            engine._loaded = True
            engine._sess = MagicMock()
            engine._post = MagicMock()
            result = engine.detect(np.zeros(0, dtype=np.float32))
            assert result.dur == 0.0
            assert result.timestamps == []

    def test_process_chunk(self, tmp_path, mock_onnx):
        from server.services.inference.engine_fire_red_vad import FireRedVadEngine

        feats = np.random.randn(100, 80).astype(np.float32)

        engine = FireRedVadEngine("test", str(tmp_path))
        engine._loaded = True
        engine._sess = MagicMock()
        engine._sess.run.return_value = [
            np.random.rand(1, 100).astype(np.float32),
            np.zeros((1, 128, 19), dtype=np.float32),
        ]
        engine._R = 1
        engine._cache_shape = (1, 128, 19)
        engine._input_names = ["feat", "cache_0"]
        engine._output_names = ["probs", "out_cache_0"]

        probs, cache = engine.process_chunk(feats)
        assert probs.shape == (100,)
        assert len(cache) == 1
