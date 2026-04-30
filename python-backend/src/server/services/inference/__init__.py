"""
推理引擎包。

提供基于 (category, runtime) 分发的引擎注册机制。新增模型只需添加引擎文件并
使用 @register_engine 装饰器，无需修改调度代码。
"""

from __future__ import annotations

from server.services.inference.base import (
    InferenceEngine,
    VadEngine,
    TtsEngine,
    AsrEngine,
    LlmEngine,
    VadResult,
    AudioResult,
    AsrResult,
    ChatResult,
)

# ---------------------------------------------------------------------------
# 引擎注册表
# ---------------------------------------------------------------------------

_ENGINE_REGISTRY: dict[tuple[str, str], type[InferenceEngine]] = {}


def register_engine(category: str, runtime: str):
    """装饰器：将引擎类按 (category, runtime) 注册到全局分发表。"""
    def decorator(cls):
        _ENGINE_REGISTRY[(category, runtime)] = cls
        return cls
    return decorator


def get_engine_class(category: str, runtime: str) -> type[InferenceEngine] | None:
    """根据类别和运行时查找已注册的引擎类。"""
    return _ENGINE_REGISTRY.get((category, runtime))


# ---------------------------------------------------------------------------
# 导入引擎模块以触发 @register_engine 自注册
# ---------------------------------------------------------------------------

from server.services.inference import engine_fire_red_vad       # noqa: E402,F401
from server.services.inference import engine_kokoro_tts         # noqa: E402,F401
from server.services.inference import engine_qwen3_llm_onnx    # noqa: E402,F401

import logging as _logging
_logger = _logging.getLogger(__name__)

try:
    from server.services.inference import engine_qwen3_asr      # noqa: E402,F401
except Exception:
    _logger.warning("ASR 引擎导入失败（不影响其他引擎运行）", exc_info=True)

try:
    from server.services.inference import engine_qwen35_llm     # noqa: E402,F401
except Exception:
    _logger.warning("GGUF LLM 引擎导入失败（不影响其他引擎运行）", exc_info=True)

try:
    from server.services.inference import engine_qwen3_tts      # noqa: E402,F401
except Exception:
    _logger.warning("Qwen3-TTS 引擎导入失败（不影响其他引擎运行）", exc_info=True)

try:
    from server.services.inference import engine_external_api_llm  # noqa: E402,F401
except Exception:
    _logger.warning("外部 API LLM 引擎导入失败（不影响其他引擎运行）", exc_info=True)


__all__ = [
    "InferenceEngine",
    "VadEngine",
    "TtsEngine",
    "AsrEngine",
    "LlmEngine",
    "VadResult",
    "AudioResult",
    "AsrResult",
    "ChatResult",
    "register_engine",
    "get_engine_class",
]
