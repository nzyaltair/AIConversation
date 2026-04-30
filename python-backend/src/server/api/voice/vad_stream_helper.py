"""
流式 VAD 辅助模块。

使用 FireRedVadEngine.process_chunk() 和持久化的 kaldi_native_fbank.OnlineFbank
进行逐帧语音活动检测。OnlineFbank 流式累积音频特征，持续追踪说话状态。
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)


class VadStreamHelper:
    """流式 VAD 辅助类。

    创建持久的 kaldi OnlineFbank 实例，持续接收音频块并检测语音活动。
    通过 process_chunk() 逐帧调用 VadEngine，返回结构化事件。

    使用方式:
        helper = VadStreamHelper(vad_engine)
        events = helper.add_audio(pcm_float32_chunk)
        for ev in events:
            if ev["event"] == "speech_start":
                ...
    """

    def __init__(
        self,
        vad_engine: object,
        speech_threshold: float = 0.4,
        min_speech_frames: int = 10,
        min_silence_frames: int = 25,
        max_utterance_frames: int = 2000,
    ) -> None:
        """初始化流式 VAD 辅助器。

        Args:
            vad_engine: FireRedVadEngine 实例，须实现 process_chunk(chunk, cache_state)
            speech_threshold: 语音概率阈值 [0, 1]
            min_speech_frames: 触发 speech_start 的最小连续语音帧数
            min_silence_frames: 触发 speech_end 的最小连续静音帧数
            max_utterance_frames: 单次话语的最大帧数（超过后强制结束）
        """
        self._vad = vad_engine
        self._threshold = speech_threshold
        self._min_speech = min_speech_frames
        self._min_silence = min_silence_frames
        self._max_utterance = max_utterance_frames

        # 持久化 FBank，跨 add_audio() 调用保持状态
        import kaldi_native_fbank as knf
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = 16000
        opts.frame_opts.frame_length_ms = 25
        opts.frame_opts.frame_shift_ms = 10
        opts.frame_opts.dither = 0.0
        opts.frame_opts.snip_edges = True
        opts.mel_opts.num_bins = 80
        opts.mel_opts.debug_mel = False
        self._fbank = knf.OnlineFbank(opts)

        # 推理缓存状态
        self._cache_state: list[np.ndarray] | None = None

        # 帧计数器
        self._last_frame_count = 0
        self._consecutive_speech = 0
        self._consecutive_silence = 0
        self._utterance_frames = 0

        # 状态标志
        self._speaking = False
        self._warmup_remaining = 0  # 暖机帧计数

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def add_audio(self, pcm_float32: np.ndarray) -> list[dict]:
        """处理新的音频数据块，返回事件列表。

        Args:
            pcm_float32: float32 音频数据，16kHz

        Returns:
            事件字典列表，可能包含:
            - {event: "speech_start"}
            - {event: "speech_continue", probability: float}
            - {event: "speech_frame", probability: float}
            - {event: "speech_end"}
        """
        events: list[dict] = []

        if len(pcm_float32) == 0:
            return events

        # 将音频送入持久化 FBank
        import kaldi_native_fbank as knf  # noqa: F811
        self._fbank.accept_waveform(16000, pcm_float32.tolist())

        # 处理新帧
        n_ready = self._fbank.num_frames_ready
        for i in range(self._last_frame_count, n_ready):
            feat = np.array(self._fbank.get_frame(i), dtype=np.float32)  # (80,)

            # 调用 VAD 引擎
            probs, self._cache_state = self._vad.process_chunk(
                feat[np.newaxis, :],  # (1, 80)
                self._cache_state,
            )
            prob = float(probs[-1]) if hasattr(probs, "__len__") and len(probs) > 0 else float(probs)

            # 暖机期：跳过前若干帧的检测
            if self._warmup_remaining > 0:
                self._warmup_remaining -= 1
                events.append({"event": "speech_frame", "probability": prob})
                continue

            # 阈值判定
            is_speech = prob > self._threshold

            if is_speech:
                self._consecutive_speech += 1
                self._consecutive_silence = 0
            else:
                self._consecutive_silence += 1
                self._consecutive_speech = 0

            if not self._speaking:
                if self._consecutive_speech >= self._min_speech:
                    # 语音开始
                    self._speaking = True
                    self._utterance_frames = self._consecutive_speech
                    events.append({"event": "speech_start"})
                    logger.debug("VAD: speech_start (prob=%.3f)", prob)
                events.append({"event": "speech_frame", "probability": prob})
            else:
                self._utterance_frames += 1
                events.append({"event": "speech_continue", "probability": prob})

                # 检查结束条件：连续静音或已达最大长度
                should_end = False
                if self._consecutive_silence >= self._min_silence:
                    should_end = True
                    reason = "silence"
                elif self._utterance_frames >= self._max_utterance:
                    should_end = True
                    reason = "max_length"

                if should_end:
                    self._speaking = False
                    self._consecutive_speech = 0
                    self._consecutive_silence = 0
                    self._utterance_frames = 0
                    events.append({"event": "speech_end"})
                    logger.debug("VAD: speech_end (reason=%s, frames=%d)",
                                 reason, self._utterance_frames)

        self._last_frame_count = n_ready
        return events

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    def reset(self) -> None:
        """重置 VAD 状态（清空 FBank 缓冲、帧计数器和缓存）。"""
        import kaldi_native_fbank as knf  # noqa: F811
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = 16000
        opts.frame_opts.frame_length_ms = 25
        opts.frame_opts.frame_shift_ms = 10
        opts.frame_opts.dither = 0.0
        opts.frame_opts.snip_edges = True
        opts.mel_opts.num_bins = 80
        opts.mel_opts.debug_mel = False
        self._fbank = knf.OnlineFbank(opts)
        self._cache_state = None
        self._last_frame_count = 0
        self._consecutive_speech = 0
        self._consecutive_silence = 0
        self._utterance_frames = 0
        self._speaking = False
        self._warmup_remaining = 0

    def reset_with_warmup(self) -> None:
        """重置 VAD 状态并插入 100ms 暖机期（10 帧）。"""
        self.reset()
        self._warmup_remaining = 10  # 10 frames = 100ms at 10ms/frame

    def update_config(
        self,
        threshold: float | None = None,
        min_speech_ms: int | None = None,
        silence_duration_ms: int | None = None,
        max_utterance_ms: int | None = None,
    ) -> None:
        """运行时更新 VAD 参数（带边界验证）。

        注意：参数更新在下一个 add_audio() 调用时生效，
        当前正在处理的话语不受影响。
        """
        if threshold is not None:
            self._threshold = max(0.1, min(0.99, float(threshold)))
        if min_speech_ms is not None:
            self._min_speech = max(1, min(100, int(min_speech_ms) // 10))
        if silence_duration_ms is not None:
            self._min_silence = max(1, min(500, int(silence_duration_ms) // 10))
        if max_utterance_ms is not None:
            self._max_utterance = max(1, min(6000, int(max_utterance_ms) // 10))
