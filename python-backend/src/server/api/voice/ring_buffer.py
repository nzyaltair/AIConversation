"""
线程安全的 float32 音频环形缓冲区。

支持并发写入和提取，用于 WebSocket 实时语音管线。
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

logger = logging.getLogger(__name__)


class AudioRingBuffer:
    """线程安全的 float32 音频环形缓冲区。

    内部维护一个固定大小的 numpy 数组。当缓冲区写满时，新数据会覆盖最旧的数据。
    使用 asyncio.Lock() 保证线程安全。
    """

    def __init__(self, capacity_seconds: float = 30.0, sample_rate: int = 16000) -> None:
        capacity_samples = int(capacity_seconds * sample_rate)
        self._buffer = np.zeros(capacity_samples, dtype=np.float32)
        self._capacity = capacity_samples
        self._write_pos = 0
        self._count = 0
        self._lock = asyncio.Lock()
        self._armed = True

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    async def write(self, samples: np.ndarray) -> None:
        """写入 float32 音频样本。如果已 disarm 则静默忽略。"""
        if not self._armed:
            return
        async with self._lock:
            n = len(samples)
            if n == 0:
                return
            if n >= self._capacity:
                # 写入数据超过整个缓冲区大小，仅保留最后 capacity 个样本
                self._buffer[:] = samples[-self._capacity:]
                self._write_pos = 0
                self._count = self._capacity
            else:
                # 环形写入
                end = self._write_pos + n
                if end <= self._capacity:
                    self._buffer[self._write_pos:end] = samples
                else:
                    first_part = self._capacity - self._write_pos
                    self._buffer[self._write_pos:] = samples[:first_part]
                    self._buffer[:n - first_part] = samples[first_part:]
                self._write_pos = end % self._capacity
                self._count = min(self._count + n, self._capacity)

    async def drain(self) -> np.ndarray:
        """提取所有有效样本（按时间顺序）。如果缓冲区为空则返回空数组。"""
        async with self._lock:
            if self._count == 0:
                return np.array([], dtype=np.float32)
            if self._count == self._capacity:
                # 缓冲区已满，数据可能回绕
                data = np.concatenate([
                    self._buffer[self._write_pos:],
                    self._buffer[:self._write_pos],
                ])
            else:
                # 数据在 [0, count) 线性排列
                data = self._buffer[:self._count].copy()
            self._count = 0
            self._write_pos = 0
            return data

    async def clear(self) -> None:
        """清空缓冲区。"""
        async with self._lock:
            self._count = 0
            self._write_pos = 0

    async def disarm(self) -> None:
        """禁止写入。"""
        self._armed = False

    async def rearm(self) -> None:
        """允许写入。"""
        self._armed = True

    @property
    def is_empty(self) -> bool:
        return self._count == 0

    async def size(self) -> int:
        async with self._lock:
            return self._count
