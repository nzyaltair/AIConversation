"""ASR pipeline 单元测试 — encoder, prompt builder, decoder, chunked pipeline。"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# FastWhisperMel (via encoder)
# ---------------------------------------------------------------------------

class TestFastWhisperMelInEncoder:
    def test_via_encoder_import(self):
        from server.services.inference.utils import FastWhisperMel
        mel = FastWhisperMel()
        audio = np.random.randn(32000).astype(np.float32)
        spec = mel(audio)
        assert spec.shape == (128, 200)  # 32000 // 160 = 200 frames


# ---------------------------------------------------------------------------
# QwenAudioEncoder
# ---------------------------------------------------------------------------

class TestQwenAudioEncoder:
    @pytest.fixture
    def mock_onnx_sessions(self):
        with patch("onnxruntime.InferenceSession") as mock_sess:
            fe_sess = MagicMock()
            fe_sess.run.return_value = [np.zeros((1, 13, 896), dtype=np.float32)]
            fe_sess.get_inputs.return_value = [MagicMock(type="tensor(float)")]
            be_sess = MagicMock()
            be_sess.run.return_value = [np.zeros((1, 13, 896), dtype=np.float32)]
            be_sess.get_inputs.return_value = [MagicMock(type="tensor(float)")]
            mock_sess.side_effect = [fe_sess, be_sess]
            yield mock_sess

    def test_encode_output_shape(self, tmp_path, mock_onnx_sessions):
        """编码后输出 (T, D) 形状正确。"""
        from server.services.inference.qwen_asr.encoder import QwenAudioEncoder

        # 写入假模型文件
        frontend = tmp_path / "frontend.onnx"
        backend = tmp_path / "backend.onnx"
        frontend.write_text("mock")
        backend.write_text("mock")

        with patch("server.services.inference.qwen_asr.encoder.FastWhisperMel") as mock_mel:
            mock_mel.return_value = MagicMock()
            mock_mel.return_value.return_value = np.random.randn(128, 200).astype(np.float32)

            encoder = QwenAudioEncoder(
                frontend_path=str(frontend),
                backend_path=str(backend),
                onnx_provider="CPU",
                dml_pad_to=0,
                verbose=False,
            )
            audio = np.random.randn(32000).astype(np.float32)
            embd, elapsed = encoder.encode(audio)
            assert embd.ndim == 2
            assert embd.shape[1] == 896


# ---------------------------------------------------------------------------
# AsrPromptBuilder
# ---------------------------------------------------------------------------

class TestAsrPromptBuilder:
    @pytest.fixture
    def builder(self):
        from server.services.inference.qwen_asr.decoder import AsrPromptBuilder
        from server.services.inference.qwen_asr import llama_binding as llama

        mock_model = MagicMock(spec=llama.LlamaModel)
        mock_model.n_embd = 896
        mock_model.tokenize.return_value = [101, 102, 103]
        mock_model.token_to_id.side_effect = lambda x: {"<|im_start|>": 1,
            "<|im_end|>": 2, "<|audio_start|>": 3, "<|audio_end|>": 4,
            "<asr_text>": 5}.get(x, 0)

        mock_table = MagicMock(spec=llama.LlamaEmbeddingTable)
        mock_table.__getitem__.return_value = np.zeros((10, 896), dtype=np.float32)

        return AsrPromptBuilder(mock_model, mock_table)

    def test_build_output_shape(self, builder):
        """构建的嵌入序列形状 = n_pre + n_aud + n_post。"""
        from unittest.mock import patch as m_patch
        with m_patch.object(builder, "table") as tbl:
            tbl.__getitem__.return_value = np.zeros((10, 896), dtype=np.float32)
            audio_embd = np.zeros((50, 896), dtype=np.float32)
            embd = builder.build(audio_embd)
            # pre + 50 + post
            assert embd.shape[0] > 50
            assert embd.shape[1] == 896

    def test_language_param(self, builder):
        audio_embd = np.zeros((50, 896), dtype=np.float32)
        embd = builder.build(audio_embd, language="zh")
        assert embd.shape[1] == 896


# ---------------------------------------------------------------------------
# AsrDecoder
# ---------------------------------------------------------------------------

class TestAsrDecoder:
    @pytest.fixture
    def mock_decoder_components(self):
        from server.services.inference.qwen_asr import llama_binding as llama

        model = MagicMock(spec=llama.LlamaModel)
        model.n_embd = 896
        model.eos_token = 2
        model.token_to_id.return_value = 2  # <|im_end|>
        model.token_to_bytes.return_value = b"test"

        ctx = MagicMock(spec=llama.LlamaContext)
        ctx.decode.return_value = 0
        ctx.decode_token.return_value = 0

        return model, ctx

    def test_decode_terminates_on_eos(self, mock_decoder_components):
        from server.services.inference.qwen_asr.decoder import AsrDecoder
        from server.services.inference.qwen_asr import llama_binding as llama

        model, ctx = mock_decoder_components
        # 配置采样器第一次就返回 EOS
        with patch.object(llama, "LlamaBatch") as mock_batch_cls, \
             patch.object(llama, "LlamaSampler") as mock_sampler_cls:
            mock_batch = MagicMock()
            mock_batch_cls.return_value = mock_batch
            mock_sampler = MagicMock()
            mock_sampler.sample.return_value = model.eos_token
            mock_sampler_cls.return_value = mock_sampler

            decoder = AsrDecoder(model, ctx, rollback_num=3)
            embd = np.zeros((100, 896), dtype=np.float32)
            result = decoder.decode(embd, temperature=0.4)

            assert not result.is_aborted
            assert result.n_generate == 0  # 第一个 token 就是 EOS

    def test_safe_decode_retry(self, mock_decoder_components):
        from server.services.inference.qwen_asr.decoder import AsrDecoder
        from server.services.inference.qwen_asr import llama_binding as llama

        model, ctx = mock_decoder_components

        with patch.object(llama, "LlamaBatch") as mock_batch_cls, \
             patch.object(llama, "LlamaSampler") as mock_s_cls:
            mock_batch = MagicMock()
            mock_batch_cls.return_value = mock_batch

            # 第一次返回 aborted, 第二次正常
            call_count = [0]
            def mock_decode(*a, **kw):
                from server.services.inference.qwen_asr.decoder import DecodeResult
                call_count[0] += 1
                res = DecodeResult()
                if call_count[0] < 3:
                    res.is_aborted = True
                else:
                    res.text = "final"
                return res

            decoder = AsrDecoder(model, ctx)
            with patch.object(decoder, "decode", side_effect=mock_decode):
                result = decoder.safe_decode(
                    np.zeros((100, 896), dtype=np.float32),
                    temperature=0.4, max_retries=4,
                )
                assert not result.is_aborted
                assert result.text == "final"


# ---------------------------------------------------------------------------
# ChunkedAsrPipeline
# ---------------------------------------------------------------------------

class TestChunkedAsrPipeline:
    def test_single_chunk(self):
        from server.services.inference.qwen_asr.decoder import (
            ChunkedAsrPipeline, DecodeResult,
        )

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = (np.zeros((50, 896), dtype=np.float32), 0.1)

        mock_builder = MagicMock()
        mock_builder.build.return_value = np.zeros((200, 896), dtype=np.float32)

        mock_decoder = MagicMock()
        res = DecodeResult(text="hello", is_aborted=False)
        mock_decoder.safe_decode.return_value = res

        pipeline = ChunkedAsrPipeline(
            encoder=mock_encoder,
            prompt_builder=mock_builder,
            decoder=mock_decoder,
            chunk_size_sec=40.0,
            memory_chunks=1,
        )

        audio = np.random.randn(16000 * 2).astype(np.float32)  # 2 秒
        text, stats = pipeline.transcribe(audio, context="test")
        assert "hello" in text
        assert stats["rtf"] >= 0
