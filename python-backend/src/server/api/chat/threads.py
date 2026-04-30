from __future__ import annotations

import json
import time
import uuid
import asyncio
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from server.app_state import AppState
from server.models.schemas import (
    ThreadResponse,
    CreateThreadRequest,
    UpdateThreadRequest,
    MessageResponse,
    SendMessageRequest,
    BatchSaveMessagesRequest,
)
from server.services.inference.base import LlmEngine

logger = logging.getLogger(__name__)


def _mock_reply(user_content: str) -> str:
    return (
        f"Thank you for your message. You said: \"{user_content[:100]}\". "
        "This is a simulated response from the AI assistant. "
        "I am here to help with conversations, answer questions, and provide information. "
        "Feel free to ask me anything!"
    )


def create_router(state: AppState) -> APIRouter:
    router = APIRouter()
    chat = state.chat_store

    @router.get("/")
    async def list_threads() -> list[dict]:
        rows = await chat.list_threads()
        return [dict(r) for r in rows]

    @router.post("/")
    async def create_thread(body: CreateThreadRequest) -> dict:
        row = await chat.create_thread(title=body.title, model_id=body.model_id or "")
        return dict(row)

    @router.get("/{thread_id}")
    async def get_thread(thread_id: str):
        row = await chat.get_thread(thread_id)
        if not row:
            from server.error_handlers import not_found
            raise not_found("Thread not found")
        return dict(row)

    @router.patch("/{thread_id}")
    async def update_thread(thread_id: str, body: UpdateThreadRequest):
        row = await chat.update_thread(thread_id, title=body.title)
        if not row:
            from server.error_handlers import not_found
            raise not_found("Thread not found")
        return dict(row)

    @router.delete("/{thread_id}")
    async def delete_thread(thread_id: str):
        ok = await chat.delete_thread(thread_id)
        if not ok:
            from server.error_handlers import not_found
            raise not_found("Thread not found")
        return {"status": "ok"}

    @router.get("/{thread_id}/messages")
    async def list_messages(thread_id: str) -> list[dict]:
        rows = await chat.list_messages(thread_id)
        return [dict(r) for r in rows]

    @router.post("/{thread_id}/messages")
    async def send_message(thread_id: str, body: SendMessageRequest):
        thread = await chat.get_thread(thread_id)
        if not thread:
            from server.error_handlers import not_found
            raise not_found("Thread not found")

        # 保存用户消息
        user_msg = await chat.append_message(
            thread_id, "user", body.content, model_id=body.model or ""
        )

        if getattr(body, "stream", False):
            return await _stream_message(chat, thread_id, body, state)

        return await _non_stream_message(chat, thread_id, body, state)

    @router.post("/{thread_id}/messages/batch")
    async def batch_save_messages(thread_id: str, body: BatchSaveMessagesRequest):
        """批量保存消息到线程（不触发推理）。"""
        thread = await chat.get_thread(thread_id)
        if not thread:
            from server.error_handlers import not_found
            raise not_found("Thread not found")

        saved: list[dict] = []
        for msg in body.messages:
            row = await chat.append_message(
                thread_id, msg.role, msg.content, model_id=msg.model_id,
            )
            saved.append(dict(row))
        return {"status": "ok", "messages": saved}

    return router


# ---------------------------------------------------------------------------
# 非流式消息回复
# ---------------------------------------------------------------------------

async def _non_stream_message(store, thread_id: str, body: SendMessageRequest, state: AppState):
    engine = await _get_llm_engine(state, body.model)
    if engine is not None:
        reply = await _generate_real_reply(engine, body)
    else:
        reply = _mock_reply(body.content)

    assistant_msg = await store.append_message(
        thread_id, "assistant", reply, model_id=body.model or ""
    )
    return dict(assistant_msg)


async def _generate_real_reply(engine: LlmEngine, body: SendMessageRequest) -> str:
    """使用真实 LLM 引擎生成回复文本。"""
    messages = [{"role": "user", "content": body.content}]
    gen_kwargs: dict = dict(
        messages=messages,
        stream=False,
        max_tokens=body.max_tokens or 2048,
        temperature=body.temperature or 0.7,
        top_p=0.9,
    )
    if body.thinking is not None:
        gen_kwargs["enable_thinking"] = body.thinking
    result = await asyncio.to_thread(engine.generate, **gen_kwargs)
    if hasattr(result, 'choices') and result.choices:
        return result.choices[0].get("message", {}).get("content", "")
    return str(result)


# ---------------------------------------------------------------------------
# 流式消息回复
# ---------------------------------------------------------------------------

async def _stream_message(store, thread_id: str, body: SendMessageRequest, state: AppState):
    engine = await _get_llm_engine(state, body.model)

    async def generate():
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        yield f"data: {json.dumps({'event': 'start', 'thread_id': thread_id})}\n\n"

        reply_parts: list[str] = []
        if engine is not None:
            async for delta_text in _stream_real_reply(engine, body):
                reply_parts.append(delta_text)
                yield f"data: {json.dumps({'event': 'delta', 'text': delta_text})}\n\n"
            reply = "".join(reply_parts)
        else:
            reply = _mock_reply(body.content)
            for char in reply:
                yield f"data: {json.dumps({'event': 'delta', 'text': char})}\n\n"
                await asyncio.sleep(0.015)

        # 保存助手消息到数据库
        msg_id = uuid.uuid4().hex
        await store._execute(
            "INSERT INTO chat_messages (id, thread_id, role, content, created_at, model_id) "
            "VALUES (:id, :tid, :r, :c, :ts, :m)",
            {"id": msg_id, "tid": thread_id, "r": "assistant", "c": reply,
             "ts": ts, "m": body.model or ""},
        )
        new_count = await store._fetch_scalar(
            "SELECT COUNT(*) FROM chat_messages WHERE thread_id = :tid",
            {"tid": thread_id},
        )
        preview = reply[:120]
        await store._execute(
            "UPDATE chat_threads SET message_count = :c, last_message_preview = :p, "
            "updated_at = :u WHERE id = :tid",
            {"c": new_count, "p": preview, "u": ts, "tid": thread_id},
        )
        yield f"data: {json.dumps({'event': 'done', 'thread_id': thread_id, 'model_id': body.model or '', 'assistant_message': {'id': msg_id, 'content': reply}})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


async def _stream_real_reply(engine: LlmEngine, body: SendMessageRequest):
    """流式调用 LLM 引擎，逐块 yield delta 文本。"""
    messages = [{"role": "user", "content": body.content}]
    loop = asyncio.get_event_loop()
    gen_kwargs: dict = dict(
        messages=messages,
        stream=True,
        max_tokens=body.max_tokens or 2048,
        temperature=body.temperature or 0.7,
        top_p=0.9,
    )
    if body.thinking is not None:
        gen_kwargs["enable_thinking"] = body.thinking
    gen = await loop.run_in_executor(
        None,
        lambda: engine.generate(**gen_kwargs),
    )
    try:
        while True:
            chunk = await loop.run_in_executor(None, next, gen, None)
            if chunk is None:
                break
            delta = (
                chunk.get("choices", [{}])[0]
                .get("delta", {})
                .get("content", "")
            )
            if delta:
                yield delta
    except StopIteration:
        pass



# ---------------------------------------------------------------------------
# 引擎获取
# ---------------------------------------------------------------------------

async def _get_llm_engine(state: AppState, model: str | None) -> LlmEngine | None:
    """尝试获取 LLM 引擎，必要时自动加载。"""
    model_variant = model or ""
    engine = state.get_llm_engine(model_variant)
    if engine is None and model_variant:
        try:
            await state.auto_load_engine(model_variant)
        except Exception:
            pass
        engine = state.get_llm_engine(model_variant)
    return engine
