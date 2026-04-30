"""Qwen3.5 LLM GGUF 推理引擎单元测试"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from unittest.mock import MagicMock, patch


class TestQwen35LlmEngine:

    @pytest.fixture
    def engine(self, tmp_path):
        from server.services.inference.engine_qwen35_llm import Qwen35LlmEngine
        return Qwen35LlmEngine("Qwen3.5-0.8B.Q4_K_M", str(tmp_path))

    def test_engine_not_loaded_raises(self, engine):
        with pytest.raises(RuntimeError, match="尚未加载"):
            engine.generate([{"role": "user", "content": "Hello"}])

    def test_extract_messages_single_user(self, engine):
        messages = [{"role": "user", "content": "你好"}]
        sys_prompt, prompt = engine._extract_messages(messages)
        assert sys_prompt == ""
        assert "你好" in prompt

    def test_extract_messages_with_system(self, engine):
        messages = [
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "Hello"},
        ]
        sys_prompt, prompt = engine._extract_messages(messages)
        assert sys_prompt == "你是助手"
        assert "Hello" in prompt

    def test_extract_messages_multi_turn(self, engine):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "How are you?"},
        ]
        _sys, prompt = engine._extract_messages(messages)
        assert "Hi" in prompt
        assert "Hello" in prompt
        assert "How are you?" in prompt

    def test_sync_generate_returns_chat_result(self, engine):
        mock_chat_engine = MagicMock()
        mock_chat_engine.chat.return_value = "这是回复"
        engine._engine = mock_chat_engine
        engine._loaded = True

        result = engine._sync_generate(
            "你好", "", 512, 1.0, 0.9, True,
        )
        assert result.object == "chat.completion"
        assert len(result.choices) == 1
        assert result.choices[0]["message"]["role"] == "assistant"
        assert result.choices[0]["message"]["content"] == "这是回复"

    def test_stream_generate_yields_chunks(self, engine):
        mock_chat_engine = MagicMock()
        mock_chat_engine.stream_chat.return_value = iter(["你", "好", ""])
        engine._engine = mock_chat_engine
        engine._loaded = True

        gen = engine._stream_generate(
            "你好", "", 512, 1.0, 0.9, True,
        )
        chunks = list(gen)
        # 有内容块 + 结束块
        assert len(chunks) >= 2
        for c in chunks:
            assert "choices" in c
            assert c["object"] == "chat.completion.chunk"

    def test_split_thinking(self, engine):
        text = "<think>这是思考内容</think>这是实际回复"
        thinking, content = engine._split_thinking(text)
        assert thinking == "这是思考内容"
        assert content == "这是实际回复"

    def test_split_thinking_no_tags(self, engine):
        text = "直接回复，没有思考过程"
        thinking, content = engine._split_thinking(text)
        assert thinking == ""
        assert content == "直接回复，没有思考过程"

    @patch(
        "server.services.inference.engine_qwen35_llm.import_module",
        autospec=True,
    )
    def test_load_creates_chat_engine(self, mock_import, tmp_path):
        from server.services.inference.engine_qwen35_llm import Qwen35LlmEngine
        import asyncio

        mock_pkg = MagicMock()
        mock_pkg.ChatConfig = MagicMock()
        mock_pkg.ChatEngine = MagicMock(return_value=MagicMock(active_backend="cpu"))
        mock_import.return_value = mock_pkg

        # 创建空模型文件
        model_file = tmp_path / "Qwen3.5-0.8B.Q4_K_M.gguf"
        model_file.write_text("mock gguf")

        engine = Qwen35LlmEngine("Qwen3.5-0.8B.Q4_K_M", str(tmp_path))
        asyncio.run(engine.load())
        assert engine.is_loaded
        mock_pkg.ChatConfig.assert_called_once()
        mock_pkg.ChatEngine.assert_called_once()
