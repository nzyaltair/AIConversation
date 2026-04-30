"""
模型种子数据模块。

应用启动时自动将预定义模型目录（SEED_MODELS）写入数据库，并清理不在目录中的废弃条目。
与前端 `model-metadata.ts` 中的 MODEL_DETAILS 一一对应。
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# 预定义模型种子目录 —— 与前端 MODEL_DETAILS 保持同步
# 每个模型包含 variant（唯一标识）、repo_id（ModelScope 仓库）、category（类别）
SEED_MODELS = [
    {
        "variant": "Qwen3.5-0.8B.Q8_0",
        "repo_id": "nzyaltair/Qwen3.5-0.8B.Q8_0",
        "category": "llm",
        "runtime": "gguf",
    },
    {
        "variant": "Qwen3.5-0.8B.Q4_K_M",
        "repo_id": "nzyaltair/Qwen3.5-0.8B.Q4_K_M",
        "category": "llm",
        "runtime": "gguf",
    },
    {
        "variant": "FireRedVad-onnx",
        "repo_id": "nzyaltair/FireRedVad-onnx",
        "category": "vad",
        "runtime": "onnx",
    },
    {
        "variant": "Kokoro-82M-v1.1-zh-ONNX-q4",
        "repo_id": "nzyaltair/Kokoro-82M-v1.1-zh-ONNX-q4",
        "category": "tts",
        "runtime": "onnx",
    },
    {
        "variant": "Qwen3-TTS-0.6B-CustomVoice-gguf",
        "repo_id": "nzyaltair/Qwen3-TTS-12Hz-0.6B-CustomVoice-gguf",
        "category": "tts",
        "runtime": "gguf",
    },
    {
        "variant": "Qwen3-TTS-1.7B-CustomVoice-gguf",
        "repo_id": "nzyaltair/Qwen3-TTS-12Hz-1.7B-CustomVoice-gguf",
        "category": "tts",
        "runtime": "gguf",
    },
    {
        "variant": "Qwen3-TTS-1.7B-VoiceDesign-gguf",
        "repo_id": "nzyaltair/Qwen3-TTS-12Hz-1.7B-VoiceDesign-gguf",
        "category": "tts",
        "runtime": "gguf",
    },
    {
        "variant": "Qwen3-ASR-0.6B-gguf",
        "repo_id": "nzyaltair/Qwen3-ASR-0.6B-gguf",
        "category": "asr",
        "runtime": "onnx+gguf",
    },
    {
        "variant": "Qwen3-0.6B-onnx",
        "repo_id": "nzyaltair/Qwen3-0.6B-onnx",
        "category": "llm",
        "runtime": "onnx",
    },
    {
        "variant": "Qwen3-ASR-1.7B-gguf",
        "repo_id": "nzyaltair/Qwen3-ASR-1.7B-gguf",
        "category": "asr",
        "runtime": "onnx+gguf",
    },
    {
        "variant": "external-api",
        "repo_id": "external/api",
        "category": "llm",
        "runtime": "external",
    },
]


async def seed_models(model_store, downloader, models_dir: str) -> None:
    """种子模型目录并清理废弃条目。

    1. 将 SEED_MODELS 中不存在的模型插入数据库（幂等）
    2. 对于已存在记录的模型，检查是否需要重置或迁移 runtime 字段
    3. 清理数据库中不在 SEED_MODELS 内的废弃条目（仅删除状态为
       not_downloaded 或 error 的条目，保护用户已下载的模型数据）

    容错策略：查询文件大小失败时 size_bytes 设为 None（不阻止启动）；
    清理失败时静默忽略以确保服务可用性。
    """
    # ── 1. 种子新模型 & 更新现有模型 ──
    seed_variants = {m["variant"] for m in SEED_MODELS}

    for m in SEED_MODELS:
        existing = await model_store.get_model(m["variant"])
        if existing is not None:
            needs_reset = False
            if existing.get("status") in ("downloaded", "ready", "error"):
                model_path = Path(models_dir) / m["variant"]
                if not model_path.exists() or not any(model_path.iterdir()):
                    needs_reset = True

            needs_runtime_migration = (
                not existing.get("runtime") and m.get("runtime")
            )

            if needs_reset or needs_runtime_migration:
                await model_store.upsert_model(
                    variant=m["variant"],
                    repo_id=m["repo_id"],
                    category=m["category"],
                    runtime=m.get("runtime", existing.get("runtime", "")),
                    size_bytes=existing.get("size_bytes"),
                    status="not_downloaded" if needs_reset else existing.get("status", "not_downloaded"),
                    enabled=bool(existing.get("enabled", 1)),
                    storage_path=None if needs_reset else existing.get("storage_path"),
                    downloaded_bytes=0 if needs_reset else existing.get("downloaded_bytes", 0),
                    error_message=None if needs_reset else existing.get("error_message"),
                )
            continue
        if m.get("repo_id", "").startswith("external/"):
            # External API models have no local files
            size_bytes = None
        else:
            try:
                files = await downloader.get_repo_file_list(m["repo_id"])
                size_bytes = sum(f["size"] for f in files) if files else None
            except Exception:
                size_bytes = None
        await model_store.upsert_model(
            variant=m["variant"],
            repo_id=m["repo_id"],
            category=m["category"],
            runtime=m.get("runtime", ""),
            size_bytes=size_bytes,
            status="not_downloaded",
        )
        await asyncio.sleep(0.5)  # 避免 ModelScope API 限流

    # ── 2. 清理废弃的模型条目 ──
    try:
        all_models = await model_store.list_models()
        for entry in all_models:
            variant = entry["variant"]
            if variant in seed_variants:
                continue  # 保留 SEED_MODELS 中的条目
            status = entry.get("status", "")
            # 只清理未下载或出错状态的废弃条目，保护用户已下载的模型
            if status in ("not_downloaded", "error"):
                await model_store.delete_model(variant)
                logger.info("已清理废弃模型条目: %s (status=%s)", variant, status)
            else:
                logger.info(
                    "保留非种子模型条目: %s (status=%s，已下载/就绪不删除)", variant, status
                )
    except Exception:
        pass  # 清理失败不阻止启动
