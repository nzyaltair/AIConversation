"""
对话状态机。

管理实时语音对话中 IDLE / LISTENING / PROCESSING / SPEAKING 四个状态的
原子性转换和广播回调。所有操作通过 asyncio.Lock 保证线程安全。
"""

from __future__ import annotations

import asyncio
import enum
import logging
from collections.abc import Coroutine
from typing import Callable

logger = logging.getLogger(__name__)


class ConversationState(str, enum.Enum):
    """对话状态枚举。继承 str 以便 JSON 序列化。"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"


# 可用的转换表：源状态 → 允许的目标状态集合
_VALID_TRANSITIONS: dict[ConversationState, set[ConversationState]] = {
    ConversationState.IDLE: {ConversationState.LISTENING},
    ConversationState.LISTENING: {ConversationState.PROCESSING, ConversationState.IDLE},
    ConversationState.PROCESSING: {
        ConversationState.SPEAKING,
        ConversationState.LISTENING,
        ConversationState.IDLE,
    },
    ConversationState.SPEAKING: {ConversationState.LISTENING, ConversationState.IDLE},
}


class StateMachine:
    """对话状态机。

    使用方式:
        sm = StateMachine()
        await sm.transition(ConversationState.LISTENING)  # IDLE -> LISTENING
        current = await sm.get()  # ConversationState.LISTENING

    on_change 回调接收 (old_state, new_state) 参数。
    """

    def __init__(self) -> None:
        self._state = ConversationState.IDLE
        self._lock = asyncio.Lock()
        self._callbacks: list[
            Callable[[ConversationState, ConversationState], Coroutine]
        ] = []

    async def get(self) -> ConversationState:
        async with self._lock:
            return self._state

    async def transition(self, target: ConversationState) -> bool:
        """尝试合法转换。成功返回 True，非法转换返回 False。"""
        async with self._lock:
            source = self._state
            allowed = _VALID_TRANSITIONS.get(source, set())
            if target not in allowed:
                logger.warning(
                    "状态转换被拒绝: %s -> %s (允许从 %s 转换到 %s)",
                    source.value, target.value, source.value,
                    {s.value for s in allowed},
                )
                return False
            self._state = target
        await self._fire_callbacks(source, target)
        return True

    async def force_transition(self, target: ConversationState) -> None:
        """强制转换（跳过验证），用于中断、错误恢复等场景。"""
        async with self._lock:
            source = self._state
            self._state = target
        await self._fire_callbacks(source, target)

    def on_change(
        self, cb: Callable[[ConversationState, ConversationState], Coroutine]
    ) -> None:
        """注册状态变化回调。每次状态转换后依次调用所有已注册的回调。"""
        self._callbacks.append(cb)

    async def _fire_callbacks(
        self, old_state: ConversationState, new_state: ConversationState,
    ) -> None:
        for cb in self._callbacks:
            try:
                await cb(old_state, new_state)
            except Exception:
                logger.exception("状态转换回调异常: %s -> %s", old_state, new_state)
