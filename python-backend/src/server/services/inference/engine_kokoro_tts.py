"""
Kokoro-82M TTS 推理引擎（ONNX Q4 运行时）。

基于 StyleTTS 2 架构的中文/英文语音合成。使用 misaki 库进行 G2P 字素→音素转换，
支持 37 种中英文音色，动态语速调整，按句分割推理。
"""

from __future__ import annotations

import logging
import os
import re

import numpy as np

from server.services.inference import register_engine
from server.services.inference.base import TtsEngine, AudioResult
from server.services.inference.utils import detect_language

logger = logging.getLogger(__name__)

# 英文专有名词映射（与 make_zh.py 保持一致）
EN_SPECIAL_WORDS = {
    'Kokoro': 'kˈOkəɹO',
    'Sol': 'sˈOl',
    'Vale': 'vˈAɪl',
    'Maple': 'mˈAːpəl',
}


@register_engine("tts", "onnx")
class KokoroTtsEngine(TtsEngine):
    """Kokoro-82M TTS 引擎（ONNX Q4 + G2P 音素转换）。

    输入文本 → G2P（字素→音素）→ 音素分词 → 结合音色嵌入 → ONNX → PCM float32。
    支持按句分割推理，段落间自动插入静音。
    """

    def __init__(self, variant: str, model_dir: str) -> None:
        super().__init__(variant, model_dir)
        self._sess: "ort.InferenceSession" | None = None
        self._tokenizer_vocab: dict[str, int] = {}
        self._voices: dict[str, np.ndarray] = {}  # name -> (N, 256) float32
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        self._sample_rate: int = 24000
        self._zh_g2p = None
        self._en_g2p = None

    async def load(self) -> None:
        import json
        import onnxruntime as ort

        # CUDA DLL 预加载
        try:
            ort.preload_dlls()
        except (AttributeError, Exception):
            pass

        onnx_path = str(self.model_dir / "model_q4.onnx")
        tokenizer_path = str(self.model_dir / "tokenizer.json")
        voices_dir = self.model_dir / "voices"

        if not self.model_dir.exists():
            raise FileNotFoundError(f"模型目录不存在: {self.model_dir}")
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"ONNX 模型文件不存在: {onnx_path}")

        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        self._sess = ort.InferenceSession(onnx_path, providers=providers)
        self._input_names = [inp.name for inp in self._sess.get_inputs()]
        self._output_names = [out.name for out in self._sess.get_outputs()]

        logger.info("Kokoro TTS ONNX 输入: %s",
                    [(inp.name, inp.shape) for inp in self._sess.get_inputs()])

        # 加载分词器词汇表（与独立验证脚本一致，直接读取 JSON）
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            tokenizer_data = json.load(f)
        self._tokenizer_vocab = tokenizer_data["model"]["vocab"]
        logger.info("Kokoro TTS 词汇表加载完成 (vocab_size=%d)", len(self._tokenizer_vocab))

        self._load_voices(voices_dir)

        self._init_g2p()

        self._loaded = True
        logger.info("Kokoro TTS 引擎加载完成 (voices=%d)", len(self._voices))

    async def unload(self) -> None:
        self._sess = None
        self._tokenizer_vocab.clear()
        self._zh_g2p = None
        self._en_g2p = None
        self._voices.clear()
        self._loaded = False

    def list_voices(self) -> list[str]:
        self._ensure_loaded()
        return sorted(self._voices.keys())

    def synthesize(self, text: str, voice: str = "zf_001",
                   speed: float = 1.0, instruct: str | None = None) -> AudioResult:
        segments = list(self.synthesize_stream(text, voice, speed))
        if not segments:
            raise ValueError("没有可处理的句子（G2P 全部失败）")
        final_audio = np.concatenate([s.audio for s in segments])
        return AudioResult(audio=final_audio, sample_rate=self._sample_rate)

    def synthesize_stream(self, text: str, voice: str = "zf_001",
                          speed: float = 1.0, instruct: str | None = None):
        """流式文本转语音，逐句 yield AudioResult。"""
        self._ensure_loaded()

        if not text.strip():
            raise ValueError("输入文本不能为空")
        if voice not in self._voices:
            available = sorted(self._voices.keys())
            raise ValueError(f"未知音色 '{voice}'，可用: {available}")

        lang = detect_language(text)
        voice_table = self._voices[voice]  # (N, 256)

        paragraphs = re.split(r'\n+', text.strip())
        last_para_idx: int | None = None

        for para_idx, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue
            para_sents = self._split_sentences(paragraph, lang)
            for sent in para_sents:
                sent = sent.strip()
                if not sent:
                    continue
                phonemes = self._g2p(sent, lang)
                if not phonemes:
                    logger.warning("G2P 转换失败，跳过: '%s'", sent[:30])
                    continue
                tokens = self._tokenize_phonemes_with_pad(phonemes)

                phonemes_len = len(phonemes)
                ref_idx = max(0, phonemes_len - 1)
                if ref_idx >= voice_table.shape[0]:
                    ref_idx = voice_table.shape[0] - 1
                style_vec = voice_table[ref_idx]  # (256,)

                speed_val = self._compute_speed(phonemes_len) if speed == 1.0 else speed

                # 段落间插入静音
                if last_para_idx is not None and para_idx != last_para_idx:
                    yield AudioResult(
                        audio=np.zeros(5000, dtype=np.float32),
                        sample_rate=self._sample_rate,
                    )

                audio = self._run_onnx_sentence(tokens, style_vec, speed_val)
                yield AudioResult(audio=audio, sample_rate=self._sample_rate)
                last_para_idx = para_idx

    # ------------------------------------------------------------------
    # 句子分割
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str, lang: str) -> list[str]:
        sep = r'[。！？]+' if lang == 'zh' else r'[.!?]+'
        return [s for s in re.split(sep, text) if s.strip()]

    # ------------------------------------------------------------------
    # G2P（字素→音素）
    # ------------------------------------------------------------------

    def _init_g2p(self) -> None:
        try:
            from misaki import en, zh
            self._en_g2p = en.G2P()
            self._zh_g2p = zh.ZHG2P(version="1.1", en_callable=self._en_callable)
            logger.info("misaki G2P 流水线初始化完成")
        except ImportError:
            logger.warning("misaki 库不可用，G2P 将回退为直接文本分词")
        except Exception as exc:
            logger.warning("G2P 初始化失败 (%s)，将回退为直接文本分词", exc)

    def _en_callable(self, text: str) -> str:
        """自定义英文回调：专有名词硬编码音标，其余用 G2P 处理。"""
        if text in EN_SPECIAL_WORDS:
            return EN_SPECIAL_WORDS[text]
        result = self._en_g2p(text)
        if isinstance(result, tuple):
            return result[0]
        return result

    def _g2p(self, text: str, lang: str) -> str:
        """文本 → 音素字符串。若 G2P 不可用则返回原始文本。"""
        if lang == "zh" and self._zh_g2p:
            try:
                result = self._zh_g2p(text)
                return result[0] if isinstance(result, tuple) else result
            except Exception as exc:
                logger.warning("中文 G2P 失败 (%s)，使用原始文本", exc)
                return text
        elif lang == "en" and self._en_g2p:
            try:
                result = self._en_g2p(text)
                return result[0] if isinstance(result, tuple) else result
            except Exception as exc:
                logger.warning("英文 G2P 失败 (%s)，使用原始文本", exc)
                return text
        return text

    @staticmethod
    def _compute_speed(phoneme_length: int) -> float:
        """根据音素长度计算动态语速（复刻独立脚本公式）。"""
        speed = 0.8
        if phoneme_length <= 83:
            speed = 1.0
        elif phoneme_length < 183:
            speed = 1.0 - (phoneme_length - 83) / 500.0
        return speed * 1.1

    # ------------------------------------------------------------------
    # 分词 & 推理
    # ------------------------------------------------------------------

    def _tokenize_phonemes(self, phonemes: str) -> np.ndarray:
        """将 G2P 输出的音素字符串编码为 token IDs（无 padding，用于兼容旧逻辑）。"""
        tokens = []
        for char in phonemes:
            if char in self._tokenizer_vocab:
                tokens.append(self._tokenizer_vocab[char])
        if len(tokens) > 510:
            tokens = tokens[:510]
        return np.array(tokens, dtype=np.int64)

    def _tokenize_phonemes_with_pad(self, phonemes: str) -> list[int]:
        """将音素编码为 token 列表，添加 BOS/EOS padding [0, *tokens, 0]。"""
        tokens = []
        for char in phonemes:
            if char in self._tokenizer_vocab:
                tokens.append(self._tokenizer_vocab[char])
        if len(tokens) > 510:
            tokens = tokens[:510]
        return [0, *tokens, 0]

    def _run_onnx_sentence(self, tokens: list[int], style_vec: np.ndarray,
                           speed: float) -> np.ndarray:
        """单句 ONNX 推理。"""
        token_seq = np.array([tokens], dtype=np.int64)  # (1, N)
        style = style_vec.reshape(1, -1).astype(np.float32)  # (1, 256)
        speed_arr = np.array([speed], dtype=np.float32)
        mask = np.ones((1, len(tokens)), dtype=np.int64)

        feed = self._build_feed_dict(token_seq, style, speed_arr, mask)
        outputs = self._sess.run(self._output_names, feed)  # type: ignore[union-attr]
        audio = outputs[0]
        while audio.ndim > 1:
            audio = audio.squeeze(0)
        return audio.astype(np.float32)

    def _build_feed_dict(self, token_seq: np.ndarray, style: np.ndarray,
                         speed_arr: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
        """按名称启发式匹配 ONNX 输入。"""
        feed: dict[str, np.ndarray] = {}
        for name in self._input_names:
            nl = name.lower()
            if any(k in nl for k in ("token", "input_ids", "input")):
                feed[name] = token_seq
            elif any(k in nl for k in ("voice", "style", "embed", "speaker")):
                feed[name] = style
            elif "speed" in nl:
                feed[name] = speed_arr
            elif "mask" in nl:
                feed[name] = mask

        if not feed:
            logger.warning("未能按名称匹配 ONNX 输入，使用位置回退策略")
            ordered = list(self._input_names)
            if len(ordered) >= 1:
                feed[ordered[0]] = token_seq
            if len(ordered) >= 2:
                feed[ordered[1]] = style
            if len(ordered) >= 3:
                feed[ordered[2]] = speed_arr
            if len(ordered) >= 4:
                feed[ordered[3]] = mask
        return feed

    # ------------------------------------------------------------------
    def _load_voices(self, voices_dir: str) -> None:
        if not os.path.isdir(voices_dir):
            logger.warning("音色目录不存在: %s", voices_dir)
            return
        for fname in os.listdir(voices_dir):
            if fname.endswith(".bin"):
                name = fname[:-4]
                emb = np.fromfile(os.path.join(voices_dir, fname), dtype=np.float32)
                # 重塑为 (N, 256) 风格向量表
                if emb.size % 256 == 0:
                    emb = emb.reshape(-1, 256)
                self._voices[name] = emb
        if self._voices:
            emb_shape = next(iter(self._voices.values())).shape
            logger.info("加载了 %d 个音色嵌入 (shape=%s)", len(self._voices), emb_shape)

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("KokoroTtsEngine 尚未加载，请先调用 load()")
