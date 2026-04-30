from dataclasses import dataclass, field
from typing import List


@dataclass
class DecodeResult:
    """LLM decode output"""
    text: str = ""
    new_text: str = ""
    stable_tokens: List[int] = field(default_factory=list)
    t_prefill: float = 0.0
    t_generate: float = 0.0
    n_prefill: int = 0
    n_generate: int = 0
    is_aborted: bool = False


@dataclass
class ChatConfig:
    """Text chat engine configuration"""
    model_dir: str
    llm_fn: str = "Qwen3.5-0.8B.Q4_K_M.gguf"

    llm_backend: str = "auto"  # auto, vulkan, cuda, cpu
    n_ctx: int = 2048
    n_batch: int = 2048
    n_ubatch: int = 512
    flash_attn: bool = True
    n_gpu_layers: int = -1  # -1 = all layers on GPU
    verbose: bool = True
    enable_thinking: bool = True  # False = skip think block, direct answer
