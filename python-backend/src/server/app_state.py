"""
应用共享状态模块。

AppState 是整个应用的运行时容器，持有所有 Store 实例、推理引擎及请求并发控制信号量。
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from server.config import ServeConfig
from server.services.inference.base import (
    InferenceEngine,
    VadEngine,
    TtsEngine,
    AsrEngine,
    LlmEngine,
)
from server.stores import (
    ChatStore,
    ModelStore,
    TranscriptionStore,
    SpeechHistoryStore,
    VoiceStore,
    VoiceObservationStore,
    SavedVoiceStore,
    OnboardingStore,
)

logger = logging.getLogger(__name__)


class AppState:
    """应用共享状态容器。

    职责：
    - 持有配置和请求并发信号量
    - 管理所有持久化 Store 的生命周期（初始化/销毁）
    - 管理推理引擎的加载/卸载/查询
    - 启动时自动种子模型目录（容错：种子失败不阻止启动）
    """

    def __init__(self, config: ServeConfig) -> None:
        self.config = config
        self.request_semaphore = asyncio.Semaphore(config.max_concurrent_requests)

        db = config.db_path
        self.chat_store = ChatStore(db)
        self.model_store = ModelStore(db)
        self.transcription_store = TranscriptionStore(db)
        self.speech_history_store = SpeechHistoryStore(db)
        self.voice_store = VoiceStore(db)
        self.voice_observation_store = VoiceObservationStore(db)
        self.saved_voice_store = SavedVoiceStore(db)
        self.onboarding_store = OnboardingStore(db)

        # 推理引擎: variant → InferenceEngine
        self._engines: dict[str, InferenceEngine] = {}

    async def initialize(self) -> None:
        """启动时初始化所有 Store，并种子模型目录。

        所有 Store 并行初始化（asyncio.gather），提高启动速度。
        种子模型目录仅添加不存在的记录（幂等），失败时静默忽略以保障服务可用性。
        """
        results = await asyncio.gather(
            self.chat_store.initialize(),
            self.model_store.initialize(),
            self.transcription_store.initialize(),
            self.speech_history_store.initialize(),
            self.voice_store.initialize(),
            self.voice_observation_store.initialize(),
            self.saved_voice_store.initialize(),
            self.onboarding_store.initialize(),
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, Exception):
                raise r

        try:
            from server.services.model_downloader import ModelDownloader
            from server.services.seed_models import seed_models
            downloader = ModelDownloader(self.config.models_dir)
            await seed_models(self.model_store, downloader, self.config.models_dir)
        except Exception:
            pass  # 种子模型目录失败不阻止启动——确保服务可用性优先于模型目录完整性

    async def shutdown(self) -> None:
        """关闭时释放所有引擎和 Store 的数据库连接。"""
        # 先卸载所有引擎
        for variant in list(self._engines.keys()):
            try:
                await self.unload_engine(variant)
            except Exception:
                logger.warning("卸载引擎失败: %s", variant, exc_info=True)

        await asyncio.gather(
            self.chat_store.dispose(),
            self.model_store.dispose(),
            self.transcription_store.dispose(),
            self.speech_history_store.dispose(),
            self.voice_store.dispose(),
            self.voice_observation_store.dispose(),
            self.saved_voice_store.dispose(),
            self.onboarding_store.dispose(),
        )

    # ------------------------------------------------------------------
    # 引擎管理
    # ------------------------------------------------------------------

    def has_engine(self, variant: str) -> bool:
        """检查指定引擎是否已加载。"""
        return variant in self._engines

    def get_engine(self, variant: str) -> InferenceEngine | None:
        """按 variant 获取已加载的引擎。"""
        return self._engines.get(variant)

    def get_vad_engine(self, variant: str) -> VadEngine | None:
        eng = self._engines.get(variant)
        return eng if isinstance(eng, VadEngine) else None

    def get_tts_engine(self, variant: str) -> TtsEngine | None:
        eng = self._engines.get(variant)
        return eng if isinstance(eng, TtsEngine) else None

    def get_asr_engine(self, variant: str) -> AsrEngine | None:
        eng = self._engines.get(variant)
        return eng if isinstance(eng, AsrEngine) else None

    def get_llm_engine(self, variant: str) -> LlmEngine | None:
        eng = self._engines.get(variant)
        return eng if isinstance(eng, LlmEngine) else None

    async def load_engine(self, variant: str) -> None:
        """按 variant 动态实例化并加载推理引擎。

        从数据库读取模型元数据（category, runtime），通过注册表查找匹配的
        引擎类，实例化后加载模型权重。新增模型只需在 SEED_MODELS 中配置
        正确的 category 和 runtime，无需修改此方法。
        """
        from server.services.inference import get_engine_class

        if variant in self._engines:
            raise RuntimeError(f"引擎已加载: {variant}")

        model = await self.model_store.get_model(variant)
        if model is None:
            raise KeyError(f"未知模型: {variant}")

        category = model.get("category", "")
        runtime = model.get("runtime", "")
        engine_class = get_engine_class(category, runtime)
        if engine_class is None:
            raise RuntimeError(
                f"未找到 ({category}, {runtime}) 对应的推理引擎，"
                f"请检查 @register_engine 注册"
            )

        model_dir = str(Path(self.config.models_dir) / variant)
        engine = engine_class(variant, model_dir)
        await engine.load()
        self._engines[variant] = engine
        logger.info("引擎加载成功: %s (%s/%s)", variant, category, runtime)

    async def unload_engine(self, variant: str) -> None:
        """卸载并移除指定引擎。"""
        engine = self._engines.pop(variant, None)
        if engine is not None:
            await engine.unload()
            logger.info("引擎已卸载: %s", variant)

    async def auto_load_engine(self, variant: str) -> bool:
        """如果模型已下载但未加载，自动加载到内存。返回是否已加载（或加载成功）。"""
        import os
        from pathlib import Path

        if variant in self._engines:
            return True
        model = await self.model_store.get_model(variant)
        if model is None:
            return False
        # External API engines skip the local directory check
        runtime = model.get("runtime", "")
        if runtime == "external":
            try:
                await self.load_engine(variant)
                await self.model_store.update_status(variant, status="ready")
                return True
            except Exception:
                logger.exception("自动加载失败: %s", variant)
                return False

        if model["status"] not in ("downloaded", "ready"):
            model_path = Path(self.config.models_dir) / variant
            if model_path.is_dir() and any(model_path.iterdir()):
                await self.model_store.update_status(variant, status="downloaded")
            else:
                return False
        try:
            await self.load_engine(variant)
            await self.model_store.update_status(variant, status="ready")
            return True
        except Exception:
            logger.exception("自动加载失败: %s", variant)
            return False

    async def get_best_llm_engine(self) -> LlmEngine | None:
        """返回当前最优的 LLM 引擎（GGUF 优先 > 任意已加载 > 自动加载 GGUF > 自动加载 ONNX）。"""
        # 1) 优先返回已加载的 GGUF LLM 引擎
        for _variant, engine in self._engines.items():
            if isinstance(engine, LlmEngine) and "gguf" in _variant.lower():
                return engine  # type: ignore[return-value]

        # 2) 返回任意已加载的 LLM 引擎
        for _variant, engine in self._engines.items():
            if isinstance(engine, LlmEngine):
                return engine  # type: ignore[return-value]

        # 3) 尝试自动加载 GGUF LLM 模型
        _gguf_variant = await self._find_llm_variant_by_runtime("gguf")
        if _gguf_variant:
            await self.auto_load_engine(_gguf_variant)
            engine = self._engines.get(_gguf_variant)
            if isinstance(engine, LlmEngine):
                return engine

        # 4) 尝试自动加载 ONNX LLM 模型
        _onnx_variant = await self._find_llm_variant_by_runtime("onnx")
        if _onnx_variant:
            await self.auto_load_engine(_onnx_variant)
            engine = self._engines.get(_onnx_variant)
            if isinstance(engine, LlmEngine):
                return engine

        return None

    async def _find_llm_variant_by_runtime(self, runtime: str) -> str | None:
        """在模型目录中查找指定运行时的 LLM 模型。"""
        try:
            models = await self.model_store.list_models(category="llm")
            for m in models:
                if m.get("runtime") == runtime:
                    return m["variant"]
        except Exception:
            pass
        return None
