"""
推理引擎基类模块。

定义 InferenceEngine 抽象基类及四个类别级接口（VadEngine / TtsEngine /
AsrEngine / LlmEngine），所有具体引擎实现均继承对应类别基类。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# 结果类型
# ---------------------------------------------------------------------------

@dataclass
class VadResult:
    """VAD 检测结果"""
    dur: float
    timestamps: list[list[float]]       # [[start, end], ...]


@dataclass
class AudioResult:
    """音频合成结果"""
    audio: "np.ndarray"                 # PCM float32 单声道
    sample_rate: int                    # 采样率


@dataclass
class AsrResult:
    """语音识别结果"""
    text: str
    language: str = ""
    segments: list[dict] = field(default_factory=list)


@dataclass
class ChatResult:
    """对话生成结果（非流式）"""
    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class InferenceEngine(ABC):
    """所有推理引擎的抽象基类。

    子类必须实现 load() 和 unload() 以管理模型生命周期。
    """

    def __init__(self, variant: str, model_dir: str) -> None:
        self.variant = variant
        self.model_dir = Path(model_dir)
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    async def load(self) -> None:
        """加载模型权重到内存。失败时应抛出异常。"""
        ...

    @abstractmethod
    async def unload(self) -> None:
        """从内存释放模型权重。"""
        ...


class VadEngine(InferenceEngine, ABC):
    """VAD 类别引擎接口"""

    @abstractmethod
    def detect(self, audio: "np.ndarray", sample_rate: int = 16000) -> VadResult:
        """完整音频的 VAD 检测，返回语音段时间戳。"""
        ...

    @abstractmethod
    def process_chunk(
        self, chunk: "np.ndarray", cache_state: list["np.ndarray"] | None = None
    ) -> tuple["np.ndarray", list["np.ndarray"]]:
        """流式逐块 VAD 处理，返回当前块的语音概率和更新的缓存状态。"""
        ...


class TtsEngine(InferenceEngine, ABC):
    """TTS 类别引擎接口"""

    @abstractmethod
    def list_voices(self) -> list[str]:
        """返回可用音色名称列表。"""
        ...

    @abstractmethod
    def synthesize(self, text: str, voice: str = "default",
                   speed: float = 1.0, instruct: str | None = None) -> AudioResult:
        """文本转语音，返回 PCM 音频数据。"""
        ...

    def synthesize_stream(self, text: str, voice: str = "default",
                          speed: float = 1.0, instruct: str | None = None):
        """流式文本转语音。子类可覆盖以按句/按块逐段 yield AudioResult。
        默认实现回退为一次性合成完整音频。
        """
        yield self.synthesize(text, voice, speed, instruct=instruct)


class AsrEngine(InferenceEngine, ABC):
    """ASR 类别引擎接口"""

    @abstractmethod
    def transcribe(self, audio: "np.ndarray", sample_rate: int = 16000) -> AsrResult:
        """音频转文字，返回识别文本及分段信息。"""
        ...


class LlmEngine(InferenceEngine, ABC):
    """LLM 类别引擎接口"""

    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        stream: bool = False,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """对话生成。

        非流式返回 ChatResult；流式返回同步生成器，每次 yield 一个 dict。
        """
        ...
