"""Qwen3-ASR 推理引擎单元测试 — 重构后版本。"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class TestQwen3AsrEngine:

    @pytest.fixture
    def engine(self, tmp_path):
        from server.services.inference.engine_qwen3_asr import Qwen3AsrEngine
        return Qwen3AsrEngine("Qwen3-ASR", str(tmp_path))

    def test_engine_not_loaded_raises(self, engine):
        """引擎未加载时抛出 RuntimeError。"""
        with pytest.raises(RuntimeError, match="尚未加载"):
            engine.transcribe(np.zeros(16000, dtype=np.float32))

    def test_engine_registration(self):
        """验证引擎正确注册 (asr, onnx+gguf)。"""
        from server.services.inference import get_engine_class
        from server.services.inference.engine_qwen3_asr import Qwen3AsrEngine
        cls = get_engine_class("asr", "onnx+gguf")
        assert cls is Qwen3AsrEngine

    def test_load_missing_files(self, engine):
        """模型文件缺失时抛出 FileNotFoundError。"""
        import asyncio
        with pytest.raises(FileNotFoundError):
            asyncio.run(engine.load())

    def test_transcribe_with_mock_pipeline(self, engine, tmp_path):
        """使用 mock 管线验证 transcribe 调用链。"""
        # 创建假模型文件
        for fn in ["qwen3_asr_encoder_frontend.int4.onnx",
                    "qwen3_asr_encoder_backend.int4.onnx",
                    "qwen3_asr_llm.q4_k.gguf"]:
            (tmp_path / fn).write_text("mock")

        engine._encoder = MagicMock()
        engine._encoder.encode.return_value = (np.zeros((50, 896), dtype=np.float32), 0.1)
        engine._pipeline = MagicMock()
        engine._pipeline.transcribe.return_value = ("测试结果", {"rtf": 0.5})
        engine._loaded = True

        audio = np.random.randn(32000).astype(np.float32)
        result = engine.transcribe(audio)
        assert "测试结果" in result.text
