"""
实时语音 WebSocket 端点。

管线: VAD (process_chunk) → ASR (to_thread) → LLM (to_thread) → TTS (to_thread)
通过 register_ws() 挂载到 FastAPI 应用。

WebSocket 路径: /v1/voice/realtime/ws
协议: IVWS 二进制帧 (用户音频 → 服务端) + JSON 文本帧 (控制消息)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from server.app_state import AppState
from server.api.voice.ivws_protocol import (
    KIND_ASSISTANT_AUDIO,
    KIND_USER_AUDIO,
    build_ivws_frame,
    float32_to_pcm16,
    parse_ivws_frame,
    pcm16_to_float32,
)
from server.api.voice.ring_buffer import AudioRingBuffer
from server.api.voice.state_machine import ConversationState, StateMachine
from server.api.voice.vad_stream_helper import VadStreamHelper

logger = logging.getLogger(__name__)

# 音频常量
_SAMPLE_RATE = 16000
_DEFAULT_VAD_MODEL = "FireRedVad-onnx"
_DEFAULT_ASR_MODEL = "Qwen3-ASR-0.6B-gguf"
_DEFAULT_LLM_MODEL = "Qwen3.5-0.8B.Q8_0"
_DEFAULT_TTS_MODEL = "Qwen3-TTS-1.7B-VoiceDesign-gguf"


# ---------------------------------------------------------------------------
# 公开 API - 供 router.py 导入
# ---------------------------------------------------------------------------


def register_ws(app: FastAPI, state: AppState) -> None:
    """注册实时语音 WebSocket 端点。

    挂载到 /v1/voice/realtime/ws，与 router.py 中的约定一致。
    """
    @app.websocket("/v1/voice/realtime/ws")
    async def voice_realtime_endpoint(ws: WebSocket):
        await handle_websocket(ws, state)


# ---------------------------------------------------------------------------
# 纯函数辅助
# ---------------------------------------------------------------------------


def _gen_sine_wave(duration: float, sample_rate: int = _SAMPLE_RATE) -> np.ndarray:
    """生成指定时长的 440Hz 正弦波音频（float32，用于 mock 回退）。"""
    n = int(duration * sample_rate)
    t = np.arange(n, dtype=np.float64) / sample_rate
    return (np.sin(2.0 * math.pi * 440.0 * t) * 0.3).astype(np.float32)



def _adjust_speed(audio: np.ndarray, sample_rate: int, speed: float) -> np.ndarray:
    """通过线性插值重采样调整音频速度（会略微改变音高）。

    对不支持 speed 参数的 TTS 引擎（如 Qwen3-TTS）进行后处理，
    确保 speed 控制在所有引擎上一致生效。
    """
    if speed == 1.0:
        return audio
    new_len = int(len(audio) / speed)
    old_idx = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(old_idx, np.arange(len(audio)), audio).astype(np.float32)



def _build_messages(
    user_text: str,
    history: list[dict],
    system_prompt: str = "",
) -> list[dict]:
    """构建 LLM 消息列表（带可选的系统提示和历史记录）。"""
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    # 保留最近 100 条消息
    for msg in history[-100:]:
        messages.append(msg)
    messages.append({"role": "user", "content": user_text})
    return messages


def _get_mock_llm_response(user_text: str) -> str:
    """生成 mock LLM 回复（引擎未加载时使用）。"""
    return (
        f"你好！你刚才说：「{user_text}」。"
        "我现在处于模拟模式，暂时无法生成真实回复。"
    )


import re

_SENTENCE_RE = re.compile(r"[^。！？\n]+[。！？\n]+")


def _split_sentences(text: str) -> list[str]:
    """按句子边界（。！？换行）拆分文本，用于逐句 TTS。

    保留分隔符，短句合并至前一句。
    """
    parts = _SENTENCE_RE.findall(text)
    if not parts:
        return [text] if text.strip() else []
    # 合并过短的片段到前一句（最小 8 字符）
    merged: list[str] = []
    for part in parts:
        if merged and len(part.strip()) < 8:
            merged[-1] += part
        else:
            merged.append(part)
    # 追加末尾未匹配的剩余文本
    last_end = 0
    for part in parts:
        last_end = text.index(part, last_end) + len(part)
    remainder = text[last_end:].strip()
    if remainder:
        if merged:
            merged[-1] += remainder
        else:
            merged.append(remainder)
    return merged


def _validate_and_apply_tts_settings(raw: dict, target: dict) -> None:
    """验证并应用客户端提供的 TTS 设置（带边界检查）。"""
    if "speaker" in raw:
        target["speaker"] = str(raw["speaker"])
    if "speed" in raw:
        v = raw["speed"]
        if isinstance(v, (int, float)):
            target["speed"] = max(0.5, min(2.0, float(v)))
    if "voice_design_instruct" in raw:
        target["voice_design_instruct"] = str(raw["voice_design_instruct"])


def _validate_and_apply_llm_settings(raw: dict, target: dict) -> None:
    """验证并应用客户端提供的 LLM 设置（带边界检查）。"""
    if "temperature" in raw:
        v = raw["temperature"]
        if isinstance(v, (int, float)) and 0.0 <= v <= 2.0:
            target["temperature"] = float(v)
    if "max_tokens" in raw:
        v = raw["max_tokens"]
        if isinstance(v, int) and 64 <= v <= 8192:
            target["max_tokens"] = v
    if "thinking_enabled" in raw:
        target["thinking_enabled"] = bool(raw["thinking_enabled"])
    if "system_prompt" in raw:
        sp = str(raw["system_prompt"])
        target["system_prompt"] = sp[:4096]  # 截断超长提示词


def _validate_and_apply_vad_config(raw: dict, target: dict) -> None:
    """验证并应用客户端提供的 VAD 配置（带边界检查）。"""
    if "threshold" in raw:
        v = raw["threshold"]
        if isinstance(v, (int, float)):
            target["threshold"] = max(0.1, min(0.99, float(v)))
    if "min_speech_ms" in raw:
        v = raw["min_speech_ms"]
        if isinstance(v, (int, float)):
            target["min_speech_ms"] = max(50, min(1000, int(v)))
    if "silence_duration_ms" in raw:
        v = raw["silence_duration_ms"]
        if isinstance(v, (int, float)):
            target["silence_duration_ms"] = max(100, min(5000, int(v)))
    if "max_utterance_ms" in raw:
        v = raw["max_utterance_ms"]
        if isinstance(v, (int, float)):
            target["max_utterance_ms"] = max(1000, min(60000, int(v)))


async def _send_audio_frames_async(
    ws: WebSocket,
    audio: np.ndarray,
    sample_rate: int,
) -> None:
    """将 float32 音频分 IVWS 帧发送到客户端。"""
    # 转换为 PCM16
    pcm_data = float32_to_pcm16(audio)

    # 分割为帧发送（每帧约 100ms，基于实际采样率动态计算）
    frame_size = sample_rate // 10 * 2
    offset = 0
    while offset < len(pcm_data):
        chunk = pcm_data[offset:offset + frame_size]
        if chunk:
            frame = build_ivws_frame(KIND_ASSISTANT_AUDIO, chunk)
            try:
                await ws.send_bytes(frame)
            except Exception:
                return
        offset += frame_size


# ---------------------------------------------------------------------------
# WebSocket 会话管理
# ---------------------------------------------------------------------------


async def handle_websocket(ws: WebSocket, state: AppState) -> None:
    """管理单个 WebSocket 连接的全生命周期。"""
    await ws.accept()
    logger.info("WebSocket 连接已接受 — 客户端: %s", ws.client)

    # ── 会话局部状态（在闭包中共享） ──
    state_machine = StateMachine()
    ring_buffer = AudioRingBuffer()
    vad_helper: VadStreamHelper | None = None
    cancel_event = asyncio.Event()

    vad_task: asyncio.Task | None = None
    turn_task: asyncio.Task | None = None
    speech_audio_chunks: list[np.ndarray] = []

    conversation_history: list[dict] = []

    model_variants: dict[str, str] = {
        "vad": _DEFAULT_VAD_MODEL,
        "asr": _DEFAULT_ASR_MODEL,
        "llm": _DEFAULT_LLM_MODEL,
        "tts": _DEFAULT_TTS_MODEL,
    }

    llm_settings: dict = {
        "temperature": 0.7,
        "max_tokens": 2048,
        "thinking_enabled": False,
        "system_prompt": "你是一个热情又贴心的小伙伴，说话自然亲切，像朋友一样。尽量用口语化的中文回答，语气轻松活泼，不要长篇大论，像日常聊天那样回应就好。",
    }
    tts_settings: dict = {
        "speaker": "Vivian",
        "speed": 1.0,
        "voice_design_instruct": "",
    }
    vad_config: dict = {
        "threshold": 0.5,
        "min_speech_ms": 400,
        "silence_duration_ms": 800,
        "max_utterance_ms": 20000,
    }
    api_config: dict = {}

    # ── 内部辅助（闭包引用 session 变量） ──

    async def _send_error(code: str, message: str) -> None:
        try:
            await ws.send_json({
                "type": "error",
                "code": code,
                "message": message,
            })
        except Exception:
            pass

    async def _broadcast_state_change(old: ConversationState, new: ConversationState) -> None:
        try:
            await ws.send_json({
                "type": "state_change",
                "state": new.value,
                "previous": old.value,
            })
        except Exception:
            pass

    state_machine.on_change(_broadcast_state_change)

    async def sleep_and_rearm() -> None:
        """暖机延迟后重新启用 ring_buffer 写入。"""
        await asyncio.sleep(0.1)
        await ring_buffer.rearm()

    def cancel_tasks() -> None:
        """取消所有正在运行的任务（非等待）。"""
        nonlocal vad_task, turn_task
        for t in [vad_task, turn_task]:
            if t is not None and not t.done():
                t.cancel()

    async def await_tasks() -> None:
        """等待所有任务完成（忽略取消和异常）。"""
        nonlocal vad_task, turn_task
        for t in [vad_task, turn_task]:
            if t is not None and not t.done():
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

    # ── VAD 循环 ──

    async def vad_loop() -> None:
        """VAD 循环：从 ring_buffer 读取音频，送入 vad_helper，触发 ASR 管线。"""
        nonlocal speech_audio_chunks, turn_task

        while not cancel_event.is_set():
            current_state = await state_machine.get()
            if current_state != ConversationState.LISTENING:
                await asyncio.sleep(0.01)
                continue

            if vad_helper is None:
                await asyncio.sleep(0.01)
                continue

            buffer_data = await ring_buffer.drain()
            if len(buffer_data) == 0:
                await asyncio.sleep(0.01)
                continue

            was_speaking = vad_helper.is_speaking
            events = vad_helper.add_audio(buffer_data)

            # 始终累积音频，避免丢失 speech_start 前的语音帧
            speech_audio_chunks.append(buffer_data)
            # 未开始说话时仅保留最近 3 秒，防止静音期间内存膨胀
            if not was_speaking:
                total_samples = sum(len(c) for c in speech_audio_chunks)
                max_samples = _SAMPLE_RATE * 3
                while total_samples > max_samples and len(speech_audio_chunks) > 1:
                    removed = speech_audio_chunks.pop(0)
                    total_samples -= len(removed)

            for ev in events:
                ev_type = ev.get("event")

                if ev_type == "speech_start":
                    logger.info("[VAD] speech_start — 检测到语音开始")

                elif ev_type == "speech_end":
                    # 在结束前 drain 环形缓冲区中的残余数据
                    remainder = await ring_buffer.drain()
                    if len(remainder) > 0:
                        speech_audio_chunks.append(remainder)

                    # GUARD: 空缓冲检查
                    if len(speech_audio_chunks) == 0:
                        continue

                    turn_audio = np.concatenate(speech_audio_chunks)
                    logger.info("[VAD] speech_end — 检测到语音结束 (缓冲时长: %.2fs)",
                                len(turn_audio) / _SAMPLE_RATE)
                    speech_audio_chunks = []

                    # 取消现有 turn 任务
                    if turn_task is not None and not turn_task.done():
                        turn_task.cancel()
                        try:
                            await turn_task
                        except (asyncio.CancelledError, Exception):
                            pass

                    turn_task = asyncio.create_task(run_turn(turn_audio))

    # ── 管线：ASR → LLM → TTS ──

    async def run_turn(speech_audio: np.ndarray) -> None:
        """执行一轮 ASR→LLM→TTS 管线。"""
        # GUARD: 空缓冲保护（最后一道防线）
        if speech_audio is None or len(speech_audio) == 0:
            logger.warning("run_turn: 空语音缓冲，跳过")
            await state_machine.transition(ConversationState.LISTENING)
            return

        if not await state_machine.transition(ConversationState.PROCESSING):
            logger.warning("run_turn: 无法转换到 PROCESSING 状态")
            return

        logger.info(
            "[TURN] === run_turn 开始 — %d 样本 (%.2fs) ===",
            len(speech_audio),
            len(speech_audio) / _SAMPLE_RATE,
        )
        logger.info("[TURN] 当前模型: vad=%s asr=%s llm=%s tts=%s",
                     model_variants.get("vad", "-"),
                     model_variants.get("asr", "-"),
                     model_variants.get("llm", "-"),
                     model_variants.get("tts", "-"))

        try:
            # ═══════════════════════════════════════════════
            # 1. ASR
            # ═══════════════════════════════════════════════
            user_text = ""
            asr_variant = model_variants.get("asr", "")
            logger.info("[TURN] ASR: variant=%s", asr_variant)
            asr_engine = state.get_asr_engine(asr_variant) if asr_variant else None

            if asr_engine is None and asr_variant:
                try:
                    if await state.auto_load_engine(asr_variant):
                        asr_engine = state.get_asr_engine(asr_variant)
                except Exception:
                    logger.warning("ASR 自动加载失败", exc_info=True)

            if asr_engine is not None:
                try:
                    result = await asyncio.to_thread(
                        asr_engine.transcribe, speech_audio, _SAMPLE_RATE,
                    )
                    user_text = result.text
                except Exception:
                    logger.warning("ASR 转录失败，重试一次", exc_info=True)
                    try:
                        result = await asyncio.to_thread(
                            asr_engine.transcribe, speech_audio, _SAMPLE_RATE,
                        )
                        user_text = result.text
                    except Exception as exc2:
                        logger.error("ASR 重试仍然失败", exc_info=True)
                        await _send_error("asr_failed", "语音识别失败，请稍后重试")
                        await state_machine.transition(ConversationState.LISTENING)
                        return
            else:
                user_text = "这是一段测试语音识别文本。"
                logger.info("ASR 引擎未加载，使用 mock 文本")

            # 发送用户转录
            logger.info("[TURN] ASR 结果: \"%s\"", user_text[:80])
            try:
                await ws.send_json({
                    "type": "user_transcript_final",
                    "text": user_text,
                })
            except Exception:
                return

            # ═══════════════════════════════════════════════
            # 2. LLM
            # ═══════════════════════════════════════════════
            messages = _build_messages(user_text, conversation_history, llm_settings["system_prompt"])

            llm_variant = model_variants.get("llm", "")
            logger.info("[TURN] LLM: variant=%s", llm_variant)
            llm_engine = state.get_llm_engine(llm_variant) if llm_variant else None
            logger.info("[TURN] LLM: engine=%s (loaded=%s)", llm_engine.__class__.__name__ if llm_engine else "None", llm_engine is not None)

            if llm_engine is None and llm_variant:
                # For external-api, explicitly load it — never fall back to local engines
                if llm_variant == "external-api":
                    logger.info("[TURN] LLM: 尝试加载 external-api 引擎")
                    try:
                        ok = await state.auto_load_engine(llm_variant)
                        logger.info("[TURN] LLM: auto_load_engine(external-api) → %s", ok)
                        if ok:
                            llm_engine = state.get_llm_engine(llm_variant)
                    except Exception:
                        logger.warning("外部 API LLM 引擎加载失败", exc_info=True)
                else:
                    # For local engines, fall back to any available LLM
                    logger.info("[TURN] LLM: 尝试 get_best_llm_engine() 回退")
                    try:
                        llm_engine = await state.get_best_llm_engine()
                        logger.info("[TURN] LLM: get_best_llm_engine() → %s", llm_engine.__class__.__name__ if llm_engine else "None")
                    except Exception:
                        logger.warning("LLM 引擎获取失败", exc_info=True)

            full_text = ""
            full_thinking = ""

            try:
                await ws.send_json({"type": "assistant_text_start"})
            except Exception:
                return

            if llm_engine is not None:
                try:
                    # Whitelist api_config keys to prevent client from
                    # overriding internal engine parameters (messages, stream, etc.)
                    _ALLOWED_API_CONFIG_KEYS = {"base_url", "api_key", "model", "reasoning_effort"}
                    _safe_api_config = {k: v for k, v in api_config.items() if k in _ALLOWED_API_CONFIG_KEYS}

                    gen_kwargs = dict(
                        messages=messages,
                        stream=True,
                        max_tokens=llm_settings["max_tokens"],
                        temperature=llm_settings["temperature"],
                        top_p=0.9,
                        enable_thinking=llm_settings.get("thinking_enabled"),
                        **_safe_api_config,
                    )

                    # Use asyncio.Queue + thread executor to avoid blocking
                    # the event loop on each streaming chunk (critical for
                    # external API engines where each chunk is an HTTP call).
                    queue: asyncio.Queue[dict | None] = asyncio.Queue()
                    loop = asyncio.get_event_loop()

                    def _producer() -> None:
                        try:
                            gen = llm_engine.generate(**gen_kwargs)
                            for chunk in gen:
                                loop.call_soon_threadsafe(queue.put_nowait, chunk)
                        except Exception as exc:
                            logger.warning("LLM 流式生成失败", exc_info=True)
                            loop.call_soon_threadsafe(
                                queue.put_nowait,
                                {"error": str(exc)},
                            )
                        finally:
                            loop.call_soon_threadsafe(queue.put_nowait, None)

                    loop.run_in_executor(None, _producer)

                    while not cancel_event.is_set():
                        chunk = await queue.get()
                        if chunk is None:
                            break

                        if "error" in chunk:
                            err_msg = chunk["error"]
                            if "[API_KEY_MISSING]" in err_msg:
                                await _send_error("api_key_missing", "API key not configured. Please open Settings and enter your API key.")
                            else:
                                await _send_error("api_error", err_msg)
                            full_text = f"[API Error] {err_msg}"
                            break

                        choices = chunk.get("choices", [])
                        if not choices:
                            continue

                        delta = choices[0].get("delta", {})
                        content_delta = delta.get("content", "")
                        thinking_delta = delta.get("thinking", "")

                        if thinking_delta:
                            full_thinking += thinking_delta
                            if llm_settings["thinking_enabled"]:
                                try:
                                    await ws.send_json({
                                        "type": "thinking_delta",
                                        "text": thinking_delta,
                                    })
                                except Exception:
                                    return

                        if content_delta:
                            full_text += content_delta
                            try:
                                await ws.send_json({
                                    "type": "assistant_text_delta",
                                    "text": content_delta,
                                })
                            except Exception:
                                return

                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("LLM 生成失败")
                    full_text = "（LLM 生成出错，请稍后重试）"
            else:
                # Mock LLM
                mock_text = _get_mock_llm_response(user_text)
                for char in mock_text:
                    if cancel_event.is_set():
                        break
                    full_text += char
                    try:
                        await ws.send_json({
                            "type": "assistant_text_delta",
                            "text": char,
                        })
                    except Exception:
                        return
                    await asyncio.sleep(0.02)

            # 发送最终文本
            logger.info("[TURN] LLM 完成: full_text=%d chars, thinking=%d chars", len(full_text), len(full_thinking))
            try:
                await ws.send_json({
                    "type": "assistant_text_final",
                    "text": full_text,
                    "thinking": full_thinking if full_thinking else None,
                })
            except Exception:
                return

            # 记录历史
            conversation_history.append({"role": "user", "content": user_text})
            conversation_history.append({"role": "assistant", "content": full_text})

            # ═══════════════════════════════════════════════
            # 3. TTS（逐句合成，尽早发送首段音频）
            # ═══════════════════════════════════════════════
            if not await state_machine.transition(ConversationState.SPEAKING):
                logger.warning("run_turn: 无法转换到 SPEAKING 状态")
                return

            tts_variant = model_variants.get("tts", "")
            logger.info("[TURN] TTS: variant=%s", tts_variant)
            tts_engine = state.get_tts_engine(tts_variant) if tts_variant else None
            logger.info("[TURN] TTS: engine=%s (loaded=%s)", tts_engine.__class__.__name__ if tts_engine else "None", tts_engine is not None)
            if tts_engine is None and tts_variant:
                try:
                    if await state.auto_load_engine(tts_variant):
                        tts_engine = state.get_tts_engine(tts_variant)
                except Exception:
                    logger.warning("TTS 自动加载失败", exc_info=True)

            if tts_engine is not None and full_text.strip():
                try:
                    instruct = tts_settings.get("voice_design_instruct") or None
                    speaker = tts_settings["speaker"]
                    # 按句子边界拆分，尽早合成并发送首段音频
                    sentences = _split_sentences(full_text)
                    logger.info("[TURN] TTS: 逐句合成 %d 个句子", len(sentences))
                    for i, sentence in enumerate(sentences):
                        if cancel_event.is_set():
                            break
                        if not sentence.strip():
                            continue
                        speed = tts_settings.get("speed", 1.0)
                        tts_result = await asyncio.to_thread(
                            tts_engine.synthesize,
                            sentence,
                            speaker,
                            1.0,
                            instruct=instruct,
                        )
                        if speed != 1.0:
                            tts_result.audio = _adjust_speed(
                                tts_result.audio, tts_result.sample_rate, speed,
                            )
                        await _send_audio_frames_async(
                            ws, tts_result.audio, tts_result.sample_rate,
                        )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.warning("TTS 合成失败，降级纯文本模式", exc_info=True)
            elif tts_engine is None:
                # Mock TTS
                mock_audio = _gen_sine_wave(1.0)
                await _send_audio_frames_async(ws, mock_audio, _SAMPLE_RATE)

            # ═══════════════════════════════════════════════
            # 4. 完成
            # ═══════════════════════════════════════════════
            logger.info("[TURN] === run_turn 完成，回到 LISTENING ===")
            try:
                await ws.send_json({"type": "turn_done"})
            except Exception:
                return

            await state_machine.transition(ConversationState.LISTENING)

        except asyncio.CancelledError:
            logger.debug("run_turn: 被取消")
            raise
        except Exception as exc:
            logger.exception("run_turn: 管线异常")
            await _send_error("pipeline_error", "处理请求时发生内部错误")
            await state_machine.force_transition(ConversationState.LISTENING)

    # ── 中断处理 ──

    async def handle_interrupt() -> None:
        """从中断恢复。安全地从任意状态调用。"""
        nonlocal turn_task, vad_task, speech_audio_chunks

        logger.debug("handle_interrupt: 开始")

        # 取消 turn_task
        if turn_task is not None and not turn_task.done():
            turn_task.cancel()
            try:
                await turn_task
            except (asyncio.CancelledError, Exception):
                pass
            turn_task = None

        # 取消 vad_task
        if vad_task is not None and not vad_task.done():
            vad_task.cancel()
            try:
                await vad_task
            except (asyncio.CancelledError, Exception):
                pass
            vad_task = None

        # 清空状态
        speech_audio_chunks = []
        await ring_buffer.clear()
        await ring_buffer.disarm()

        if vad_helper is not None:
            vad_helper.reset_with_warmup()

        # 强制进入 LISTENING
        await state_machine.force_transition(ConversationState.LISTENING)

        # 100ms 后重新启用写入
        asyncio.create_task(sleep_and_rearm())

        # 重启 VAD
        vad_task = asyncio.create_task(vad_loop())
        logger.debug("handle_interrupt: 完成")

    # ── JSON 消息分发 ──

    async def handle_json_message(data: dict) -> None:
        """处理 JSON 控制消息。"""
        nonlocal vad_task, vad_helper, model_variants, llm_settings, tts_settings, vad_config, api_config

        msg_type = data.get("type", "")

        if msg_type == "session_start":
            logger.info("[WS] ← session_start — 会话初始化")
            # 无需特殊操作，连接即开始

        elif msg_type == "input_stream_start":
            logger.info("[WS] ← input_stream_start — 模型变体: %s", data.get("model_variants", {}))
            logger.info("[WS]   api_config keys: %s", list(data.get("api_config", {}).keys()))
            # Guard: cancel existing VAD task to prevent zombie tasks
            if vad_task is not None and not vad_task.done():
                vad_task.cancel()
                try:
                    await vad_task
                except (asyncio.CancelledError, Exception):
                    pass
                vad_task = None

            mv = data.get("model_variants", {})
            if mv:
                model_variants.update(mv)

            # 解析并验证 LLM/TTS/VAD 设置
            ls = data.get("llm_settings", {})
            if ls:
                _validate_and_apply_llm_settings(ls, llm_settings)
            ts = data.get("tts_settings", {})
            if ts:
                _validate_and_apply_tts_settings(ts, tts_settings)
            vc = data.get("vad_config", {})
            if vc:
                _validate_and_apply_vad_config(vc, vad_config)

            # 保存外部 API 配置（base_url, api_key, model, reasoning_effort）
            ac = data.get("api_config", {})
            if ac:
                api_config = dict(ac)

            # 加载引擎（阻塞等待，避免首轮 run_turn 中内联加载阻塞管线）
            try:
                await asyncio.wait_for(
                    _load_required_engines_async(state, model_variants),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                logger.warning("引擎加载超时，回退到懒加载")
            except Exception:
                logger.warning("引擎加载异常", exc_info=True)

            # 创建 VAD helper（使用客户端提供的配置）
            vad_variant = model_variants.get("vad", _DEFAULT_VAD_MODEL)
            logger.info("[WS] 获取 VAD 引擎: variant=%s", vad_variant)
            vad_engine = state.get_vad_engine(vad_variant)
            if vad_engine is None and vad_variant:
                logger.info("[WS] VAD 引擎未加载，尝试 auto_load_engine(%s)", vad_variant)
                try:
                    ok = await state.auto_load_engine(vad_variant)
                    logger.info("[WS] auto_load_engine(%s) → %s", vad_variant, ok)
                    if ok:
                        vad_engine = state.get_vad_engine(vad_variant)
                except Exception:
                    logger.warning("VAD 自动加载失败", exc_info=True)

            if vad_engine is None:
                logger.error("[WS] VAD 引擎不可用，发送 vad_unavailable 错误")
                await _send_error("vad_unavailable", "VAD 引擎未加载")
                return

            vad_helper = VadStreamHelper(
                vad_engine,
                speech_threshold=vad_config["threshold"],
                min_speech_frames=max(1, vad_config["min_speech_ms"] // 10),
                min_silence_frames=max(1, vad_config["silence_duration_ms"] // 10),
                max_utterance_frames=max(1, vad_config["max_utterance_ms"] // 10),
            )
            cancel_event.clear()
            await state_machine.transition(ConversationState.LISTENING)
            vad_task = asyncio.create_task(vad_loop())
            logger.info("[WS] input_stream_start 完成 — VAD 循环已启动 (threshold=%.2f)", vad_config["threshold"])

        elif msg_type == "input_stream_stop":
            logger.info("[WS] ← input_stream_stop — 停止输入流")
            cancel_event.set()
            cancel_tasks()
            await await_tasks()
            vad_task = None
            turn_task = None
            await state_machine.force_transition(ConversationState.IDLE)

        elif msg_type == "interrupt":
            logger.info("[WS] ← interrupt — 用户中断")
            await handle_interrupt()

        elif msg_type == "update_settings":
            mv = data.get("model_variants", {})
            logger.info("[WS] ← update_settings — 模型变体: %s", mv)
            if mv:
                model_variants.update(mv)
            ls = data.get("llm_settings", {})
            if ls:
                _validate_and_apply_llm_settings(ls, llm_settings)
            ts = data.get("tts_settings", {})
            if ts:
                _validate_and_apply_tts_settings(ts, tts_settings)
            vc = data.get("vad_config", {})
            if vc:
                _validate_and_apply_vad_config(vc, vad_config)
                # 使用公共方法更新 VAD helper 参数
                if vad_helper is not None:
                    vad_helper.update_config(
                        threshold=vad_config["threshold"],
                        min_speech_ms=vad_config["min_speech_ms"],
                        silence_duration_ms=vad_config["silence_duration_ms"],
                        max_utterance_ms=vad_config["max_utterance_ms"],
                    )
            ac = data.get("api_config", {})
            if ac:
                api_config = dict(ac)
            logger.info("update_settings: 设置已更新")

        elif msg_type == "ping":
            try:
                await ws.send_json({"type": "pong"})
            except Exception:
                pass

        else:
            logger.warning("未知消息类型: %s", msg_type)

    # ── 主接收循环 ──
    try:
        while True:
            raw = await ws.receive()

            if raw.get("type") == "websocket.disconnect":
                break

            text_data = raw.get("text")
            bytes_data = raw.get("bytes")

            if text_data is not None:
                try:
                    data = json.loads(text_data)
                    await handle_json_message(data)
                except json.JSONDecodeError:
                    logger.warning("收到无效 JSON 消息")

            elif bytes_data is not None:
                try:
                    kind, payload = parse_ivws_frame(bytes_data)
                    if kind == KIND_USER_AUDIO:
                        audio_float32 = pcm16_to_float32(payload)
                        await ring_buffer.write(audio_float32)
                except ValueError as exc:
                    logger.warning("无效 IVWS 帧: %s", exc)

    except WebSocketDisconnect:
        logger.info("[WS] WebSocket 客户端断开连接")
    except Exception:
        logger.exception("[WS] WebSocket 处理异常")
    finally:
        logger.info("[WS] 开始清理会话...")
        cancel_event.set()
        cancel_tasks()
        await await_tasks()
        try:
            await ws.close()
        except Exception:
            pass
        logger.info("[WS] 会话清理完成")


# ---------------------------------------------------------------------------
# 引擎加载辅助（模块级函数）
# ---------------------------------------------------------------------------


async def _load_required_engines_async(state: AppState, variants: dict[str, str]) -> None:
    """后台加载所有必需的推理引擎。失败仅记录警告。"""
    logger.info("[ENGINE] _load_required_engines_async 开始 — variants: %s", variants)
    for key in ("vad", "asr", "llm", "tts"):
        variant = variants.get(key, "")
        if not variant:
            logger.info("[ENGINE]   %s: 未指定 variant，跳过", key)
            continue
        if state.has_engine(variant):
            logger.info("[ENGINE]   %s: %s 已加载，跳过", key, variant)
            continue
        try:
            logger.info("[ENGINE]   %s: 尝试 auto_load_engine(%s)...", key, variant)
            ok = await state.auto_load_engine(variant)
            if ok:
                logger.info("[ENGINE]   %s: %s 自动加载成功", key, variant)
            else:
                logger.warning("[ENGINE]   %s: %s 未就绪（模型文件不存在或未下载）", key, variant)
        except Exception:
            logger.warning("[ENGINE]   %s: %s 加载失败", key, variant, exc_info=True)
    logger.info("[ENGINE] _load_required_engines_async 完成")
