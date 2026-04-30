"""
Chat Completions API — OpenAI 兼容的 LLM 推理端点。

端点：POST /v1/chat/completions
格式：兼容 OpenAI Chat Completions API（支持 stream 和非 stream 模式）

所有推理均使用本地 LLM 引擎，引擎未加载时回退到 mock 响应。
"""

from __future__ import annotations

import json
import time
import uuid
import asyncio
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from server.app_state import AppState
from server.models.schemas import ChatCompletionRequest
from server.services.inference.base import LlmEngine

logger = logging.getLogger(__name__)


def _mock_reply(body: ChatCompletionRequest) -> str:
    """当 LLM 引擎未加载时，生成模拟回复文本。"""
    last_user_msg = ""
    for m in body.messages:
        if m.get("role") == "user":
            last_user_msg = m.get("content", "")
    preview = last_user_msg[:100]
    return (
        f'Thank you for your message. You said: "{preview}". '
        "This is a simulated response from the AI assistant. "
        "I am here to help with conversations, answer questions, and provide information. "
        "Feel free to ask me anything!"
    )


def _mock_completion_result(body: ChatCompletionRequest) -> dict:
    """当引擎不可用时，返回 OpenAI 兼容的非流式 mock 响应。"""
    created = int(time.time())
    reply = _mock_reply(body)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": created,
        "model": body.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": reply},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


async def _mock_stream_completion(body: ChatCompletionRequest):
    """SSE 流式 mock 回复，逐字符输出，格式兼容 OpenAI SSE chunk。"""
    created = int(time.time())
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    reply = _mock_reply(body)
    for char in reply:
        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': body.model, 'choices': [{'index': 0, 'delta': {'content': char}, 'finish_reason': None}]})}\n\n"
        await asyncio.sleep(0.015)
    yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': body.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
    yield "data: [DONE]\n\n"


def create_router(state: AppState) -> APIRouter:
    """创建 Chat Completions 路由，遵循 OpenAI API 格式。"""

    router = APIRouter()

    @router.post("/completions")
    async def completions(body: ChatCompletionRequest):
        engine = state.get_llm_engine(body.model)
        if engine is None:
            await state.auto_load_engine(body.model)
            engine = state.get_llm_engine(body.model)

        if engine is None:
            # 请求的模型无法加载，尝试回退到其他可用的 LLM 引擎
            fallback = await state.get_best_llm_engine()
            if fallback is not None:
                logger.warning(
                    "LLM 引擎 '%s' 加载失败，回退到 '%s'",
                    body.model, fallback.variant,
                )
                engine = fallback
                body.model = fallback.variant

        if engine is None:
            logger.warning("LLM 引擎 '%s' 未加载，使用 mock 回复", body.model)
            if not body.stream:
                return _mock_completion_result(body)
            return StreamingResponse(
                _mock_stream_completion(body), media_type="text/event-stream",
            )

        if not body.stream:
            return await _real_completion(engine, body)

        return StreamingResponse(
            _stream_completion(engine, body), media_type="text/event-stream",
        )

    async def _real_completion(engine: LlmEngine, body: ChatCompletionRequest):
        """使用真实 LLM 引擎进行非流式推理。"""
        messages = [{"role": m["role"], "content": m["content"]} for m in body.messages]

        gen_kwargs: dict = dict(
            messages=messages,
            stream=False,
            max_tokens=body.max_tokens or 2048,
            temperature=body.temperature if body.temperature is not None else 1.0,
            top_p=body.top_p if body.top_p is not None else 1.0,
        )
        if body.thinking is not None:
            gen_kwargs["enable_thinking"] = body.thinking

        result = await asyncio.to_thread(engine.generate, **gen_kwargs)
        return result.__dict__ if hasattr(result, '__dict__') else result

    async def _stream_completion(engine: LlmEngine, body: ChatCompletionRequest):
        """SSE 流式推理：asyncio.Queue + 单后台线程，避免 next(gen, None) 哨兵歧义。"""
        messages = [{"role": m["role"], "content": m["content"]} for m in body.messages]
        queue: asyncio.Queue[dict | None] = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _producer() -> None:
            try:
                gen_kwargs: dict = dict(
                    messages=messages,
                    stream=True,
                    max_tokens=body.max_tokens or 2048,
                    temperature=body.temperature if body.temperature is not None else 1.0,
                    top_p=body.top_p if body.top_p is not None else 1.0,
                )
                if body.thinking is not None:
                    gen_kwargs["enable_thinking"] = body.thinking
                gen = engine.generate(**gen_kwargs)
                for chunk in gen:
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as exc:
                logger.exception("LLM 流式生成失败")
                loop.call_soon_threadsafe(queue.put_nowait, {"error": str(exc)})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        asyncio.get_event_loop().run_in_executor(None, _producer)

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            if "error" in chunk:
                yield f"data: {json.dumps({'error': chunk['error']})}\n\n"
                break
            yield f"data: {json.dumps(chunk)}\n\n"

        yield "data: [DONE]\n\n"

    return router
