"""
模型管理 API 路由。

端点前缀：/v1/admin/models

功能：
  - 模型列表/详情查询（GET /, /{variant}）
  - 模型下载/取消下载（POST /{variant}/download, /cancel）
  - 下载进度 SSE 订阅（GET /{variant}/download/progress）
  - 模型删除（DELETE /{variant}）
  - 磁盘扫描检测孤立模型（GET /scan-disk）

模型状态生命周期：not_downloaded → downloading → downloaded → ready（推理时自动加载）
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse

from server.app_state import AppState
from server.api.dependencies import get_app_state
from server.error_handlers import invalid_request, not_found, server_error
from server.models.schemas import ModelInfoResponse
from server.services.model_downloader import (
    DownloadCancelledError,
    ModelDownloader,
)

logger = logging.getLogger(__name__)


def create_router(state: AppState) -> APIRouter:
    """创建模型管理路由。

    管理模型下载/删除的完整生命周期，通过 SSE 推送下载进度，
    并提供磁盘扫描检测孤立模型。推理引擎在首次使用时自动加载。
    """
    router = APIRouter()
    downloader = ModelDownloader(state.config.models_dir)
    _active_tasks: dict[str, asyncio.Task] = {}       # 活跃下载任务映射
    _active_count: dict[str, int] = {"count": 0}       # 当前并发下载计数
    _max_concurrent = 2                                 # 最大并发下载数

    def _model_to_response(m: dict) -> dict:
        """将数据库行字典映射为 API 响应格式。"""
        return {
            "variant": m["variant"],
            "repo_id": m["repo_id"],
            "category": m["category"],
            "status": m["status"],
            "enabled": bool(m["enabled"]),
            "size_bytes": m["size_bytes"],
            "downloaded_bytes": m["downloaded_bytes"],
            "error_message": m["error_message"],
        }

    @router.get("/")
    async def list_models(
        category: str | None = Query(None),
        sort_by: str | None = Query(None),
    ):
        models = await state.model_store.list_models(category, sort_by)
        return [_model_to_response(m) for m in models]

    @router.get("/{variant}")
    async def get_model(variant: str):
        m = await state.model_store.get_model(variant)
        if m is None:
            raise not_found(f"Model {variant} not found")
        return _model_to_response(m)

    @router.post("/{variant}/download")
    async def download_model(variant: str):
        m = await state.model_store.get_model(variant)
        if m is None:
            raise not_found(f"Model {variant} not found")
        if m["status"] == "downloading":
            raise invalid_request("Already downloading")
        if _active_count["count"] >= _max_concurrent:
            raise invalid_request("Max concurrent downloads reached")

        _active_count["count"] += 1

        async def _progress_callback(percent, current_file, downloaded_bytes, total_bytes):
            await state.model_store.update_status(
                variant,
                status="downloading",
                downloaded_bytes=downloaded_bytes,
            )

        async def _download_task():
            """后台下载任务。

            流程：更新 status=downloading → 调用 ModelDownloader.download_model
            → 成功时 status=downloaded → 失败（DownloadCancelledError）清理目录并重置
            → 其他异常设置 status=error → finally 清理并发计数和任务引用
            """
            try:
                await state.model_store.update_status(
                    variant, status="downloading", downloaded_bytes=0, error_message=None
                )
                storage_path = await downloader.download_model(
                    variant, m["repo_id"], _progress_callback
                )
                await state.model_store.update_status(
                    variant,
                    status="downloaded",
                    downloaded_bytes=m["size_bytes"] or 0,
                    storage_path=storage_path,
                )
            except DownloadCancelledError:
                # 下载被取消：清理已下载的部分文件目录，状态重置为 not_downloaded
                storage_path = os.path.join(state.config.models_dir, variant)
                if os.path.isdir(storage_path):
                    try:
                        shutil.rmtree(storage_path)
                    except OSError:
                        pass
                await state.model_store.update_status(
                    variant, status="not_downloaded", downloaded_bytes=0
                )
            except Exception as exc:
                # 下载异常：记录日志并标记模型为 error 状态
                logger.exception("Download failed for %s", variant)
                await state.model_store.update_status(
                    variant, status="error", error_message=str(exc)
                )
            finally:
                _active_count["count"] -= 1
                _active_tasks.pop(variant, None)

        task = asyncio.create_task(_download_task())
        _active_tasks[variant] = task
        return {"status": "ok"}

    @router.post("/{variant}/download/cancel")
    async def cancel_download(variant: str):
        m = await state.model_store.get_model(variant)
        if m is None:
            raise not_found(f"Model {variant} not found")
        downloader.cancel(variant)
        # Clean up partial download directory
        storage_path = os.path.join(state.config.models_dir, variant)
        if os.path.isdir(storage_path):
            try:
                shutil.rmtree(storage_path)
            except OSError:
                pass
        await state.model_store.update_status(
            variant, status="not_downloaded", downloaded_bytes=0
        )
        return {"status": "ok"}

    @router.delete("/{variant}")
    async def delete_model(variant: str):
        """删除模型文件并将状态重置为 not_downloaded。

        仅清除磁盘上的模型文件目录和下载进度，
        保留模型目录记录以便重新下载。
        """
        m = await state.model_store.get_model(variant)
        if m is None:
            raise not_found(f"Model {variant} not found")
        storage_path = m.get("storage_path")
        if storage_path and os.path.isdir(storage_path):
            try:
                shutil.rmtree(storage_path)
            except OSError:
                pass
        await state.model_store.upsert_model(
            variant=m["variant"],
            repo_id=m["repo_id"],
            category=m["category"],
            runtime=m.get("runtime", ""),
            size_bytes=m["size_bytes"],
            status="not_downloaded",
            enabled=bool(m["enabled"]),
            storage_path=None,
            downloaded_bytes=0,
            error_message=None,
        )
        return {"status": "ok"}

    @router.get("/scan-disk")
    async def scan_disk():
        """扫描模型目录，检测孤立模型（磁盘上存在但不在目录中的模型）。

        遍历 models_dir 下的所有非空子目录，与数据库中的模型记录比对，
        返回在磁盘上找到但未在目录中注册的模型列表。
        扫描失败时不抛出异常，返回空列表以确保页面可用性。
        """
        models_dir = state.config.models_dir
        try:
            dirs = os.listdir(models_dir)
        except OSError:
            logger.exception("无法读取模型目录: %s", models_dir)
            return {"disk_models": [], "orphaned": [], "catalog_count": 0}

        disk_models: list[str] = []
        for name in dirs:
            full = os.path.join(models_dir, name)
            if not os.path.isdir(full):
                continue
            try:
                has_files = any(
                    entry.is_file() for entry in os.scandir(full)
                )
            except OSError:
                continue
            if has_files:
                disk_models.append(name)

        catalog = await state.model_store.list_models()
        catalog_variants = {m["variant"] for m in catalog}
        orphaned = [d for d in disk_models if d not in catalog_variants]

        logger.debug(
            "磁盘扫描: %d 个目录, %d 个在目录中, %d 个孤立: %s",
            len(disk_models), len(catalog_variants), len(orphaned), orphaned,
        )
        return {
            "disk_models": disk_models,
            "orphaned": orphaned,
            "catalog_count": len(catalog_variants),
        }

    @router.post("/{variant}/load")
    async def load_model(variant: str):
        """将已下载的模型加载到内存中用于推理。"""
        if state.has_engine(variant):
            return {"status": "ok", "message": f"Model '{variant}' already loaded"}
        m = await state.model_store.get_model(variant)
        if m is None:
            raise not_found(f"Model '{variant}' not found")
        if m["status"] not in ("downloaded", "ready"):
            # External API engines have no local files — allow loading directly
            if m.get("runtime") != "external":
                raise invalid_request(
                    f"Model '{variant}' has status '{m['status']}', "
                    f"must be downloaded first"
                )
        try:
            await state.load_engine(variant)
            await state.model_store.update_status(variant, status="ready")
            return {"status": "ok", "message": f"Model '{variant}' loaded"}
        except Exception as exc:
            raise server_error(f"Failed to load model '{variant}': {exc}")

    @router.post("/{variant}/unload")
    async def unload_model(variant: str):
        """从内存中卸载模型，释放 GPU/CPU 资源。"""
        if not state.has_engine(variant):
            return {"status": "ok", "message": f"Model '{variant}' not loaded"}
        try:
            await state.unload_engine(variant)
            await state.model_store.update_status(variant, status="downloaded")
            return {"status": "ok", "message": f"Model '{variant}' unloaded"}
        except Exception as exc:
            raise server_error(f"Failed to unload model '{variant}': {exc}")

    @router.get("/{variant}/download/progress")
    async def download_progress(variant: str):
        """SSE 端点：实时推送模型下载进度。

        每 0.5 秒轮询一次数据库中的 downloaded_bytes 字段，
        直至模型状态不再为 'downloading' 时停止推送。
        """
        async def generate():
            while True:
                m = await state.model_store.get_model(variant)
                if m is None:
                    break
                data = json.dumps({
                    "variant": variant,
                    "percent": (
                        round(m["downloaded_bytes"] / m["size_bytes"] * 100, 1)
                        if m["size_bytes"]
                        else 0.0
                    ),
                    "current_file": "",
                    "status": m["status"],
                    "downloaded_bytes": m["downloaded_bytes"],
                    "total_bytes": m["size_bytes"] or 0,
                })
                yield f"event: message\ndata: {data}\n\n"
                if m["status"] != "downloading":
                    break
                await asyncio.sleep(0.5)

        return StreamingResponse(generate(), media_type="text/event-stream")

    return router
