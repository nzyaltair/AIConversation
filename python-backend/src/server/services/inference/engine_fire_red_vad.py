"""
FireRedVad 推理引擎（ONNX 运行时）。

适配自 models/FireRedVad-onnx/fireredvad_onnx_inference.py 参考实现。
"""

from __future__ import annotations

import logging

import numpy as np

from server.services.inference import register_engine
from server.services.inference.base import VadEngine, VadResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

class FireRedVadConfig:
    """VAD 后处理参数"""
    def __init__(
        self,
        smooth_window_size: int = 5,
        speech_threshold: float = 0.4,
        min_speech_frame: int = 20,
        max_speech_frame: int = 2000,
        min_silence_frame: int = 20,
        merge_silence_frame: int = 20,
        extend_speech_frame: int = 20,
        chunk_max_frame: int = 30000,
    ):
        if not 0 <= speech_threshold <= 1:
            raise ValueError("speech_threshold must be in [0, 1]")
        if min_speech_frame <= 0:
            raise ValueError("min_speech_frame must be positive")
        self.smooth_window_size = smooth_window_size
        self.speech_threshold = speech_threshold
        self.min_speech_frame = min_speech_frame
        self.max_speech_frame = max_speech_frame
        self.min_silence_frame = min_silence_frame
        self.merge_silence_frame = merge_silence_frame
        self.extend_speech_frame = extend_speech_frame
        self.chunk_max_frame = chunk_max_frame


# ---------------------------------------------------------------------------
# 特征提取辅助类
# ---------------------------------------------------------------------------

class _CMVN:
    """读取 Kaldi 格式 CMVN 文件并应用归一化"""

    def __init__(self, kaldi_cmvn_file: str):
        import kaldiio
        stats = kaldiio.load_mat(kaldi_cmvn_file)
        if stats.shape[0] != 2:
            raise ValueError(f"Unexpected CMVN stats shape: {stats.shape}")
        dim = stats.shape[-1] - 1
        count = stats[0, dim]
        if count < 1:
            raise ValueError(f"CMVN count too small: {count}")
        self.dim = dim
        self.means = stats[0, :dim] / count
        variance = stats[1, :dim] / count - self.means * self.means
        variance = np.maximum(variance, 1e-20)
        self.inverse_std = 1.0 / np.sqrt(variance)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.shape[-1] != self.dim:
            raise ValueError(f"CMVN dim mismatch: {x.shape[-1]} vs {self.dim}")
        return (x - self.means) * self.inverse_std


class _FbankExtractor:
    """kaldi-native-fbank FBank 特征提取器"""

    def __init__(self, num_mel_bins: int = 80, frame_length: int = 25,
                 frame_shift: int = 10, dither: float = 0.0):
        import kaldi_native_fbank as knf
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = 16000
        opts.frame_opts.frame_length_ms = frame_length
        opts.frame_opts.frame_shift_ms = frame_shift
        opts.frame_opts.dither = dither
        opts.frame_opts.snip_edges = True
        opts.mel_opts.num_bins = num_mel_bins
        opts.mel_opts.debug_mel = False
        self._opts = opts
        self._num_bins = num_mel_bins

    def extract(self, sample_rate: int, waveform: np.ndarray) -> np.ndarray:
        import kaldi_native_fbank as knf
        fbank = knf.OnlineFbank(self._opts)
        fbank.accept_waveform(sample_rate, waveform.tolist())
        n = fbank.num_frames_ready
        if n == 0:
            return np.zeros((0, self._num_bins), dtype=np.float32)
        return np.vstack([fbank.get_frame(i) for i in range(n)])


class _AudioFeat:
    """音频特征提取流水线：FBank + CMVN"""

    def __init__(self, cmvn_path: str | None = None):
        self._cmvn = _CMVN(cmvn_path) if cmvn_path else None
        self._fbank = _FbankExtractor(num_mel_bins=80, frame_length=25,
                                       frame_shift=10, dither=0.0)

    def extract(self, audio: np.ndarray, sample_rate: int = 16000
                ) -> tuple[np.ndarray, float]:
        """返回 (fbank_feat, duration_seconds)"""
        if sample_rate != 16000:
            raise ValueError(f"Expected 16kHz audio, got {sample_rate}Hz")
        wav = np.asarray(audio, dtype=np.float32).flatten()
        dur = len(wav) / sample_rate
        feat = self._fbank.extract(sample_rate, wav)
        if self._cmvn is not None:
            feat = self._cmvn(feat)
        return feat.astype(np.float32), dur


# ---------------------------------------------------------------------------
# 后处理
# ---------------------------------------------------------------------------

class _VadPostprocessor:
    """VAD 后处理：平滑、阈值判定、语音段合并"""

    def __init__(self, config: FireRedVadConfig):
        self._cfg = config

    def process(self, probs: np.ndarray) -> list[int]:
        """逐帧概率 → 0/1 决策序列"""
        probs = np.asarray(probs, dtype=np.float32).flatten()
        if self._cfg.smooth_window_size > 1:
            kernel = np.ones(self._cfg.smooth_window_size) / self._cfg.smooth_window_size
            probs = np.convolve(probs, kernel, mode="same")
        decisions = (probs > self._cfg.speech_threshold).astype(int).tolist()
        return self._merge_segments(decisions)

    def decisions_to_timestamps(self, decisions: list[int]) -> list[list[float]]:
        """0/1 决策序列 → [[start_sec, end_sec], ...]（帧移 10ms）"""
        segments: list[list[float]] = []
        start: int | None = None
        for i, d in enumerate(decisions):
            if d == 1 and start is None:
                start = i
            elif d == 0 and start is not None:
                segments.append([start * 0.01, i * 0.01])
                start = None
        if start is not None:
            segments.append([start * 0.01, len(decisions) * 0.01])
        return segments

    def _merge_segments(self, decisions: list[int]) -> list[int]:
        """合并过短的语音段及过短的静音间隔"""
        cfg = self._cfg
        segments: list[tuple[int, int]] = []
        start: int | None = None
        for i, d in enumerate(decisions):
            if d == 1 and start is None:
                start = i
            elif d == 0 and start is not None:
                segments.append((start, i - 1))
                start = None
        if start is not None:
            segments.append((start, len(decisions) - 1))

        # 过滤短语音段、截断超长段
        merged: list[tuple[int, int]] = []
        for s, e in segments:
            length = e - s + 1
            if length < cfg.min_speech_frame:
                continue
            if length > cfg.max_speech_frame:
                e = s + cfg.max_speech_frame - 1
            merged.append((s, e))

        # 合并短静音间隔
        if cfg.merge_silence_frame > 0 and len(merged) > 1:
            final = [merged[0]]
            for i in range(1, len(merged)):
                gap = merged[i][0] - final[-1][1] - 1
                if gap <= cfg.merge_silence_frame:
                    final[-1] = (final[-1][0], merged[i][1])
                else:
                    final.append(merged[i])
            merged = final

        # 扩展语音段前后
        if cfg.extend_speech_frame > 0:
            n = len(decisions)
            merged = [
                (max(0, s - cfg.extend_speech_frame),
                 min(n - 1, e + cfg.extend_speech_frame))
                for s, e in merged
            ]

        new = [0] * len(decisions)
        for s, e in merged:
            for i in range(s, e + 1):
                new[i] = 1
        return new


# ---------------------------------------------------------------------------
# 引擎
# ---------------------------------------------------------------------------

@register_engine("vad", "onnx")
class FireRedVadEngine(VadEngine):
    """FireRedVad 语音活动检测引擎（ONNX）。

    输入: 16kHz 单声道 PCM16 音频
    输出: VadResult（时长 + 语音段时间戳列表）
    """

    def __init__(self, variant: str, model_dir: str) -> None:
        super().__init__(variant, model_dir)
        self._config = FireRedVadConfig()
        self._feat: _AudioFeat | None = None
        self._post: _VadPostprocessor | None = None
        self._sess: "ort.InferenceSession" | None = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        self._R: int = 0  # cache 数量
        self._cache_shape: tuple = (1, 128, 19)  # 已知固定值

    async def load(self) -> None:
        import onnxruntime as ort

        onnx_path = str(self.model_dir / "model.onnx")
        cmvn_path = str(self.model_dir / "cmvn.ark")

        if not self.model_dir.exists():
            raise FileNotFoundError(f"模型目录不存在: {self.model_dir}")

        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        logger.info("FireRedVad ONNX providers: %s", providers)

        self._sess = ort.InferenceSession(onnx_path, providers=providers)
        self._feat = _AudioFeat(cmvn_path)
        self._post = _VadPostprocessor(self._config)

        self._input_names = [inp.name for inp in self._sess.get_inputs()]
        self._output_names = [out.name for out in self._sess.get_outputs()]
        self._R = sum(1 for n in self._input_names if n.startswith("cache_"))

        cache_input = self._sess.get_inputs()[1]
        raw_shape = cache_input.shape
        if any(isinstance(d, str) for d in raw_shape):
            logger.info("FireRedVad cache 形状含动态维度 %s，使用固定值 %s",
                        raw_shape, self._cache_shape)
        else:
            self._cache_shape = tuple(raw_shape)

        self._loaded = True
        logger.info("FireRedVad 引擎加载完成 (R=%d, cache_shape=%s)",
                    self._R, self._cache_shape)

    async def unload(self) -> None:
        self._sess = None
        self._feat = None
        self._post = None
        self._loaded = False

    def detect(self, audio: np.ndarray, sample_rate: int = 16000) -> VadResult:
        """对完整音频执行 VAD 检测。"""
        self._ensure_loaded()
        feats, dur = self._feat.extract(audio, sample_rate)  # type: ignore[union-attr]
        T = feats.shape[0]
        logger.debug("VAD 特征提取: frames=%d, dur=%.2fs", T, dur)
        if T == 0:
            return VadResult(dur=round(dur, 3), timestamps=[])

        probs = self._run_onnx(feats)
        logger.debug("VAD 概率: min=%.4f, max=%.4f, mean=%.4f, over_threshold=%d/%d",
                     probs.min(), probs.max(), probs.mean(),
                     (probs > self._config.speech_threshold).sum(), len(probs))
        decisions = self._post.process(probs)  # type: ignore[union-attr]
        timestamps = self._post.decisions_to_timestamps(decisions)  # type: ignore[union-attr]
        logger.debug("VAD 检测到 %d 个语音段: %s", len(timestamps), timestamps)
        return VadResult(dur=round(dur, 3), timestamps=timestamps)

    def process_chunk(
        self, chunk: np.ndarray, cache_state: list[np.ndarray] | None = None
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """逐块流式 VAD。chunk 形状 (T, 80)，返回 (probs, new_cache)。"""
        self._ensure_loaded()
        if cache_state is None:
            cache_state = [
                np.zeros(self._cache_shape, dtype=np.float32)
                for _ in range(self._R)
            ]
        feats = np.expand_dims(chunk.astype(np.float32), axis=0)
        feed_dict = {"feat": feats}
        for i, cache in enumerate(cache_state):
            feed_dict[f"cache_{i}"] = cache
        outputs = self._sess.run(self._output_names, feed_dict)  # type: ignore[union-attr]
        return outputs[0].flatten(), [outputs[1 + i] for i in range(self._R)]

    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("FireRedVadEngine 尚未加载，请先调用 load()")

    def _run_onnx(self, feats: np.ndarray) -> np.ndarray:
        """分块运行 ONNX 推理，返回逐帧概率。"""
        chunk_size = self._config.chunk_max_frame
        T = feats.shape[0]
        all_probs: list[np.ndarray] = []
        caches = [
            np.zeros(self._cache_shape, dtype=np.float32)
            for _ in range(self._R)
        ]

        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            chunk_feat = feats[start:end]
            chunk_feat = np.expand_dims(chunk_feat, axis=0).astype(np.float32)

            feed_dict = {"feat": chunk_feat}
            for i, cache in enumerate(caches):
                feed_dict[f"cache_{i}"] = cache

            outputs = self._sess.run(self._output_names, feed_dict)  # type: ignore[union-attr]
            all_probs.append(outputs[0].flatten())
            for i in range(self._R):
                caches[i] = outputs[1 + i]

        return np.concatenate(all_probs)
