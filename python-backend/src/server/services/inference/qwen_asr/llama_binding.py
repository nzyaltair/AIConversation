# coding=utf-8
"""
llama.cpp GGUF 推理绑定 — 桥接到 qwen_asr_gguf.inference.llama。

提供统一的高层接口：LlamaModel、LlamaContext、LlamaBatch、
LlamaSampler、以及 get_token_embeddings_gguf。
"""

from server.services.inference.qwen_asr_gguf.inference.llama import (
    LlamaModel,
    LlamaContext,
    LlamaBatch,
    LlamaSampler,
    LlamaEmbeddingTable,
    get_token_embeddings_gguf,
    token_to_bytes,
    text_to_tokens,
)

__all__ = [
    "LlamaModel",
    "LlamaContext",
    "LlamaBatch",
    "LlamaSampler",
    "LlamaEmbeddingTable",
    "get_token_embeddings_gguf",
    "token_to_bytes",
    "text_to_tokens",
]
