"""
ModelScope 模型下载服务。

通过 modelscope SDK 从 ModelScope Hub 下载模型文件，
支持进度回调、取消下载。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from modelscope.hub.api import HubApi
from modelscope.hub.file_download import model_file_download


class DownloadCancelledError(Exception):
    """取消下载时抛出的异常信号，调用方据此进行清理。"""


class ModelDownloader:
    """ModelScope Hub 模型下载服务。

    职责：
    1. 查询 ModelScope 仓库文件列表及大小
    2. 逐个下载模型文件，通过 progress_callback 报告进度
    3. 支持通过 asyncio.Event 取消下载
    """

    def __init__(self, models_dir: str) -> None:
        self._models_dir = Path(models_dir)
        self._api = HubApi()
        self._cancel_events: dict[str, asyncio.Event] = {}

    def cancel(self, variant: str) -> None:
        """触发指定 variant 的取消信号。"""
        event = self._cancel_events.get(variant)
        if event:
            event.set()

    def get_cancel_event(self, variant: str) -> asyncio.Event:
        """获取或创建 variant 的取消事件。"""
        event = self._cancel_events.get(variant)
        if event is None:
            event = asyncio.Event()
            self._cancel_events[variant] = event
        return event

    def _cleanup_cancel_event(self, variant: str) -> None:
        """下载完成后清理取消事件，释放内存。"""
        self._cancel_events.pop(variant, None)

    async def get_repo_file_list(self, model_id: str) -> list[dict]:
        """查询 ModelScope 仓库中的文件列表。

        在线程池中调用 HubApi.get_model_files（避免阻塞事件循环），
        返回包含 path 和 size 的文件信息列表。
        """
        files = await asyncio.to_thread(
            self._api.get_model_files, model_id, recursive=True
        )
        result = []
        for f in files:
            size = f.get("Size") or 0
            if size > 0:
                result.append({"path": f["Path"], "size": size})
        return result

    async def download_model(
        self,
        variant: str,
        repo_id: str,
        progress_callback,
    ) -> str:
        """下载模型的所有文件，返回存储路径。

        流程：
        1. 获取仓库文件列表及总大小
        2. 逐个下载文件
        3. 每完成一个文件即回调 progress_callback 报告进度
        4. 每下载一个文件前检查取消信号
        5. 下载完成后清理取消事件

        Raises:
            DownloadCancelledError: 在下载过程中检测到取消信号
        """
        cancel_event = self.get_cancel_event(variant)
        storage_path = self._models_dir / variant
        storage_path.mkdir(parents=True, exist_ok=True)

        files = await self.get_repo_file_list(repo_id)
        if not files:
            raise FileNotFoundError(f"No files found for {repo_id}")

        total_bytes = sum(f["size"] for f in files)
        downloaded_so_far = 0

        for file_info in files:
            if cancel_event.is_set():
                raise DownloadCancelledError()

            def _download():
                return model_file_download(
                    model_id=repo_id,
                    file_path=file_info["path"],
                    local_dir=str(storage_path),
                )

            await asyncio.to_thread(_download)
            downloaded_so_far += file_info["size"]
            percent = (downloaded_so_far / total_bytes * 100) if total_bytes > 0 else 0
            await progress_callback(
                percent=round(percent, 1),
                current_file=file_info["path"],
                downloaded_bytes=downloaded_so_far,
                total_bytes=total_bytes,
            )

        self._cleanup_cancel_event(variant)
        return str(storage_path)
