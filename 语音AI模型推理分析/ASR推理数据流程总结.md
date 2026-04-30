# Qwen3-ASR-GGUF 推理数据流程总结

## 1. 系统环境

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 (6GB) |
| CUDA | 12.6 |
| cuDNN | 9.x (`nvidia-cudnn-cu12`) |
| ONNX Runtime | 1.25.0 (GPU 版) |
| 加速方式 | ONNX 编码器：CUDAExecutionProvider；LLM 解码器：llama.cpp GPU 层卸载 (-1 全部) |
| Python | 3.12+ |

## 2. 初始化流程

```
test_asr_inference.py
    │  自动设置 CUDA/cuDNN PATH
    ├─ 创建 ASREngineConfig (onnx_provider='CUDA', llm_use_gpu=True)
    └─ 创建 QwenASREngine
        ├─ 1. 加载 ONNX 编码器（Frontend + Backend），Provider: CUDAExecutionProvider
        │     └─ 预热：编码 2s 静音音频
        ├─ 2. 加载 llama.cpp GGUF LLM（n_gpu_layers=-1，全部卸载到 GPU）
        ├─ 3. 提取 GGUF 嵌入表（get_token_embeddings_gguf）
        ├─ 4. 创建 llama.cpp 上下文（n_ctx=2048, n_batch=4096, flash_attn=True）
        └─ 5. 缓存特殊 Token ID（<|im_start|>, <|im_end|>, <|audio_start|> 等）
```

**初始化耗时**: ~1.5s

## 3. 推理流程（单次转录）

```
音频文件 (.wav/.mp3 等)
    │
    ▼
┌──────────────────────────────────────────────────┐
│ Step 1: 音频加载 (audio.py)                       │
│   - WAV/FLAC/OGG/MP3 → soundfile                 │
│   - 其他格式 → ffmpeg pipe                         │
│   - 重采样到 16000Hz 单声道                        │
│   - 输出: float32 numpy array (N_samples,)        │
└────────────────────────┬─────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────┐
│ Step 2: 分片 (chunk_size=40s)                      │
│   - 按 seconds × 16000 切分                        │
│   - 不足 40s 的尾部补零                            │
│   - 输出: List[chunk] (每片 ≤ 640000 samples)      │
└────────────────────────┬─────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │      逐片循环 (i=0..N)        │
          └──────────────┬──────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────┐
│ Step 3: ONNX 编码器 (encoder.py)                   │
│                                                    │
│   3a. Mel 频谱提取 (FastWhisperMel, 纯 NumPy)       │
│     - 汉明窗 → 实数 FFT → 能量谱                    │
│     - Slaney 128-bin Mel 滤波器组                   │
│     - log10 → 归一化 [0, 1]                        │
│     - 输入: (N_samples,) → 输出: (128, T_frames)   │
│     - hop_length=160 → 每 10ms 一帧                 │
│                                                    │
│   3b. Frontend ONNX (CNN 下采样)                    │
│     - 模型: qwen3_asr_encoder_frontend.int4.onnx   │
│     - Pad 帧数到 100 的倍数                         │
│     - 每 100 帧一个 chunk 送入 ONNX                  │
│     - 每 chunk 输出 13 帧 hidden_state              │
│     - 拼接所有 chunk 输出                            │
│     - 输出: (1, T_fe, 1024)                        │
│                                                    │
│   3c. Backend ONNX (Transformer)                    │
│     - 模型: qwen3_asr_encoder_backend.int4.onnx    │
│     - 构造 attention_mask（全 0 = 全关注）           │
│     - 输入: hidden_states + attention_mask          │
│     - 输出: (1, T_fe, 896)                          │
│                                                    │
│   编码耗时 (CUDA): ~0.14s / 40s 音频                 │
│   输出: audio_embd (T_fe, 896) float32              │
└────────────────────────┬─────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────┐
│ Step 4: 构造 Prompt Embedding (asr.py)             │
│                                                    │
│   拼接顺序:                                         │
│   ┌────────────┬──────────────┬───────────────┐   │
│   │ Prefix     │ Audio Emb    │ Suffix        │   │
│   │ (文本token) │ (音频嵌入)    │ (指令+历史)    │   │
│   ├────────────┼──────────────┼───────────────┤   │
│   │ <|im_start|>│              │ <|audio_end|>  │   │
│   │ system\n    │ T_fe × 896   │ <|im_end|>     │   │
│   │ <prompt>    │ float32      │ <|im_start|>   │   │
│   │ <|im_end|>  │              │ assistant\n    │   │
│   │ <|im_start|>│              │ language zh    │   │
│   │ user\n      │              │ <|asr_text|>   │   │
│   │ <|audio_start|>│           │ <prefix_text>  │   │
│   └────────────┴──────────────┴───────────────┘   │
│                                                    │
│   分词表查找: embedding_table[token_ids]            │
│   总嵌入: concat(prefix_embd, audio_embd, suffix_embd) │
└────────────────────────┬─────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────┐
│ Step 5: llama.cpp GGUF 解码 (asr.py + llama.py)    │
│                                                    │
│   模型: qwen3_asr_llm.q4_k.gguf                    │
│   量化: Q4_K (4-bit)                               │
│   GPU 卸载: 全部层 (-1)                             │
│                                                    │
│   5a. Prefill (一次处理全部嵌入)                     │
│     - LlamaBatch 打包全部 token 嵌入                │
│     - pos 编码: 连续递增 0..N-1                     │
│     - ctx.decode(batch) → 填充 KV Cache             │
│     - 预填充速度 (CUDA): ~11282 tokens/s            │
│                                                    │
│   5b. 自回归生成 (max 512 tokens)                    │
│     - LlamaSampler (temperature=0.4, random seed)   │
│     - 单 token 解码循环                              │
│     - Rollback 窗口: 5 tokens (防抖)                │
│     - 增量 UTF-8 解码 → 实时流式输出                  │
│     - 停止条件: EOS / <|im_end|>                     │
│     - 熔断检测: 连续 15 个 token 重复 ≤ 3 种         │
│     - 生成速度 (CUDA): ~270 tokens/s                │
│                                                    │
│   5c. 熔断重试 (最多 4 次)                           │
│     - 温度每次 +0.3 → 重新生成                       │
└────────────────────────┬─────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────┐
│ Step 6: 记忆更新                                   │
│   - asr_memory.append((audio_embd, decoded_text)) │
│   - 下一片的 prefix_text = 记忆中的所有文本拼接      │
│   - 下一片的 combined_audio = 记忆音频嵌入 + 当前片  │
│   - memory_num=1: 只保留前 1 片的记忆               │
└──────────────────────────────────────────────────┘
```

## 4. 性能数据（CUDA 加速）

### 中文测试音频 (asr_zh.wav, 4.2s)

| 指标 | 数值 |
|---|---|
| 编码耗时 | 0.143s |
| LLM 预填充 | 0.078s (542 tokens → 6938 tokens/s) |
| LLM 生成 | 0.036s (7 tokens → 194 tokens/s) |
| 总处理耗时 | 0.27s |
| **RTF (实时率)** | **0.064** |

**转录结果**: 甚至出现交易几乎停滞的情况。

### 英文测试音频 (asr_en.wav, 15.1s)

| 指标 | 数值 |
|---|---|
| 编码耗时 | 0.125s |
| LLM 预填充 | 0.048s (542 tokens → 11282 tokens/s) |
| LLM 生成 | 0.167s (45 tokens → 270 tokens/s) |
| 总处理耗时 | 0.35s |
| **RTF (实时率)** | **0.023** |

**转录结果**: Hmm. Oh yeah, yeah. He wasn't even that big when I started listening to him. But in his solo music, didn't do overly well. But he did very well when he started writing for other people.

### CPU vs CUDA 对比

| 指标 | CPU (auto 回落) | CUDA | 提升 |
|---|---|---|---|
| 加载耗时 | 2.68s | 1.53s | 1.8× |
| 编码耗时 (zh) | 0.997s | 0.143s | 7.0× |
| LLM 预填充速度 | 5418 t/s | 6938 t/s | 1.3× |
| LLM 生成速度 | 157 t/s | 194 t/s | 1.2× |
| RTF (zh) | 0.276 | 0.064 | 4.3× |
| RTF (en) | 0.086 | 0.023 | 3.7× |

## 5. 数据维度变化

```
原始音频                    (16000 × seconds,)  float32
    │
    ▼ FastWhisperMel
Mel 频谱                    (128, T_frames)     float16/32
    │  每 10ms 一帧 (hop=160), 帧数 = duration × 100
    │
    ▼ Frontend ONNX (100帧/chunk → 13帧 输出)
Hidden States              (1, T_fe, 1024)     float16/32
    │  T_fe ≈ T_frames × 13/100
    │
    ▼ Backend ONNX (Transformer)
Audio Embedding            (T_fe, 896)         float32
    │
    ▼ 与文本嵌入拼接
Full Embedding             (N_total, 896)      float32
    │  N_total = len(prefix_token) + T_fe + len(suffix_token)
    │           = ~30 + T_fe + ~20 ≈ 542  (40s 音频)
    │
    ▼ llama.cpp GGUF 解码
输出 Token 序列            [...token_ids]       int32
    │
    ▼ detokenize
最终文本                  str
```

## 6. 模型文件清单

| 文件 | 用途 | 精度 |
|---|---|---|
| `qwen3_asr_encoder_frontend.int4.onnx` | Mel → 下采样特征 | INT4 |
| `qwen3_asr_encoder_backend.int4.onnx` | Transformer 编码 | INT4 |
| `qwen3_asr_llm.q4_k.gguf` | 自回归文本解码 | Q4_K (4-bit) |
| `qwen3_aligner_encoder_frontend.int4.onnx` | 对齐前端 (可选) | INT4 |
| `qwen3_aligner_encoder_backend.int4.onnx` | 对齐后端 (可选) | INT4 |
| `qwen3_aligner_llm.q4_k.gguf` | 对齐解码器 (可选) | Q4_K |
| `configuration.json` | 模型配置 | — |

## 7. 关键代码入口

| 模块 | 文件 | 职责 |
|---|---|---|
| 测试入口 | `test_asr_inference.py` | CUDA 路径配置 + 引擎加载 + 转录测试 |
| 引擎主类 | `qwen_asr_gguf/inference/asr.py` | QwenASREngine：初始化、Prompt 构造、分片调度 |
| ONNX 编码器 | `qwen_asr_gguf/inference/encoder.py` | FastWhisperMel + QwenAudioEncoder (Frontend + Backend) |
| llama.cpp 绑定 | `qwen_asr_gguf/inference/llama.py` | CTypes 绑定：LlamaModel, LlamaContext, LlamaBatch, LlamaSampler |
| 音频加载 | `qwen_asr_gguf/inference/audio.py` | soundfile/ffmpeg 加载 + NumPy 重采样 |
| 配置结构 | `qwen_asr_gguf/inference/schema.py` | ASREngineConfig, DecodeResult, TranscribeResult 等数据类 |
| 语言校验 | `qwen_asr_gguf/inference/utils.py` | 30 种语言归一化与校验 |

## 8. 运行命令

```bash
# 激活虚拟环境
source .venv/Scripts/activate

# 首次安装 cuDNN（仅一次）
pip install nvidia-cudnn-cu12

# 运行测试
python test_asr_inference.py
```
