# coding=utf-8
"""
Qwen3-ASR ONNX 音频编码器 — 桥接到 qwen_asr_gguf.inference.encoder。

提供 QwenAudioEncoder：Mel 频谱提取 + ONNX Frontend + ONNX Backend。
"""

from server.services.inference.qwen_asr_gguf.inference.encoder import (
    QwenAudioEncoder,
    FastWhisperMel,
    get_feat_extract_output_lengths,
)

__all__ = [
    "QwenAudioEncoder",
    "FastWhisperMel",
    "get_feat_extract_output_lengths",
]
