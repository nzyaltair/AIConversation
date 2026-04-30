# Kokoro-82M-v1.1-zh ONNX INT4 TTS 推理总结

> 测试日期：2026-04-27 | 推理后端：CUDA (NVIDIA GPU)

---

## 一、模型信息

| 属性 | 值 |
|------|----|
| 基础模型 | hexgrad/Kokoro-82M-v1.1-zh |
| 参数量 | 82M |
| 模型类型 | style_text_to_speech_2 |
| 量化方式 | INT4 |
| 模型格式 | ONNX |
| 模型文件 | `model_q4.onnx` |
| 支持语言 | 中文 (zh)、英文 (en) |
| 采样率 | 24,000 Hz |
| Token 最大长度 | 512 |

---

## 二、运行环境

| 组件 | 版本/信息 |
|------|----------|
| ONNX Runtime | 1.25.0 (GPU) |
| 推理后端 | `CUDAExecutionProvider` |
| 可用 Providers | TensorRT, CUDA, CPU |
| NumPy | 2.4.4 |
| Python | 3.12+ |
| G2P 工具 | misaki[zh] (ZHG2P v1.1) |
| 分词器 | `tokenizer.json` |

---

## 三、推理流程图

```
┌─────────────┐
│  输入文本     │
└──────┬──────┘
       ▼
┌─────────────────────────────────────┐
│ 1. 文本预处理                        │
│    ├─ 按换行符分割段落 (paragraphs)    │
│    └─ 按标点符号分割句子 (。！？\n)     │
└──────┬──────────────────────────────┘
       ▼
┌─────────────────────────────────────┐
│ 2. G2P 音素转换 (misaki ZHG2P)       │
│    ├─ 中文文本 → 中文音素序列          │
│    ├─ 英文文本 → en_callable(专有名词) │
│    └─ 输出: 音素字符串 (phonemes)      │
└──────┬──────────────────────────────┘
       ▼
┌─────────────────────────────────────┐
│ 3. Token 化                          │
│    ├─ 查 tokenizer_vocab 映射         │
│    ├─ 过滤未知音素 (静默跳过)          │
│    ├─ token > 510 → 截断             │
│    └─ 输出: (phonemes, tokens, paragraph_idx) 元组列表
└──────┬──────────────────────────────┘
       ▼
┌─────────────────────────────────────┐
│ 4. 加载资源 (一次性)                  │
│    ├─ ONNX 模型 → InferenceSession   │
│    └─ 声音向量 → voices (N, 1, 256)  │
└──────┬──────────────────────────────┘
       ▼
┌─────────────────────────────────────────────┐
│ 5. 逐句推理循环 (每个句子)                    │
│    ┌─────────────────────────────────────┐  │
│    │ 5a. 选择声音风格向量                  │  │
│    │     idx = min(len(phonemes)-1, N-1) │  │
│    │     style = voices[idx] (1, 256)    │  │
│    ├─────────────────────────────────────┤  │
│    │ 5b. 计算动态语速                      │  │
│    │     · len ≤ 83  → speed = 1.1       │  │
│    │     · 83 < len < 183 → 递减          │  │
│    │     · len ≥ 183 → speed = 0.88      │  │
│    ├─────────────────────────────────────┤  │
│    │ 5c. 构造模型输入                      │  │
│    │     input_ids = [[0, *tokens, 0]]   │  │
│    │     style   = ref_s (1, 256)        │  │
│    │     speed   = [speed_val] float32   │  │
│    ├─────────────────────────────────────┤  │
│    │ 5d. ONNX 推理 (CUDA GPU)            │  │
│    │     audio = sess.run(inputs)        │  │
│    │     wav = audio.squeeze()           │  │
│    └──────────────┬──────────────────────┘  │
│                   ▼                          │
│     段落切换 → 插入 n_zeros 静音片段          │
│     audio_segments.append(wav)               │
└──────────────────────┬──────────────────────┘
                       ▼
┌─────────────────────────────────────┐
│ 6. 音频后处理                        │
│    ├─ np.concatenate(audio_segments) │
│    └─ soundfile.write(wav, 24000)   │
└─────────────────────────────────────┘
```

---

## 四、本次测试数据

### 4.1 输入文本

```
"你好，这是一个文本转语音测试。用于比较不同ONNX模型的性能和延迟。"
```

### 4.2 预处理结果

| 阶段 | 结果 |
|------|------|
| 段落数 | 1 |
| 句子数 | 2 |
| 使用的音色 | `zf_001.bin` (中文女声) |
| 可用声音向量总数 | 510 |

### 4.3 逐句推理明细

| 句子 | 音素数 | 风格索引 | 语速 (speed) | 推理时间 | 说明 |
|------|--------|---------|-------------|---------|------|
| 1 | 42 | 41 | 1.100 | 0.719 秒 | "你好，这是一个文本转语音测试" |
| 2 | 59 | 58 | 1.100 | 0.254 秒 | "用于比较不同ONNX模型的性能和延迟" |

### 4.4 汇总指标

| 指标 | 数值 |
|------|------|
| **模型加载时间** | 1.372 秒 |
| **总推理时间** | 0.973 秒 |
| **平均每句推理时间** | 0.487 秒 |
| **生成音频采样点数** | 190,800 |
| **生成音频时长** | 7.95 秒 |
| **输出文件** | `output_model_q4.wav` (381,644 字节) |
| **推理速度比** | 0.973 秒推理 → 7.95 秒音频 ≈ **8.2× 实时** |

---

## 五、关键技术细节

### 5.1 CUDA DLL 加载机制

脚本在 `import onnxruntime` 之前完成以下操作：

1. 扫描 `site-packages/nvidia/` 下所有 `cudnn`、`cublas`、`cuda_nvrtc`、`cuda_runtime`、`cufft`、`curand` 包的 `bin` 目录
2. 自动检测 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\` 下安装的 CUDA Toolkit（优先最新版本）
3. 将所有 DLL 目录添加到 `PATH` 环境变量
4. 调用 `os.add_dll_directory()` 注册 DLL 搜索路径
5. 调用 `ort.preload_dlls()` 预加载 CUDA/cuDNN 动态库

### 5.2 声音风格向量选择规则

- **索引基准**：音素长度 (`len(phonemes)`)，而非 token 数量
- **索引公式**：`ref_idx = max(0, len(phonemes) - 1)`，上界为 `voices.shape[0] - 1`
- 音素越多 → 索引越大 → 声音风格随句子长度不同而变化

### 5.3 动态语速计算公式

```python
speed = 0.8
if len_ps <= 83:
    speed = 1.0        # 短句: 基准速度
elif len_ps < 183:
    speed = 1 - (len_ps - 83) / 500  # 中句: 线性递减
speed *= 1.1           # 全局系数
```

- 短句 (≤83 音素): speed = 1.1
- 中等句子 (83~183 音素): speed = 1.1 → 0.88 线性递减
- 长句 (≥183 音素): speed ≈ 0.88

### 5.4 Token 序列格式

```python
token_seq = [[0, *tokens, 0]]
```

- 序列首尾各添加一个 padding token (0)
- 作为 `input_ids` 传入 ONNX 模型

### 5.5 段落间静音处理

```python
if last_paragraph_idx is not None and paragraph_idx != last_paragraph_idx:
    audio_segments.append(np.zeros(n_zeros, dtype=wav.dtype))
```

- 默认 `n_zeros = 5000`，即约 208ms 静音 (5000/24000)
- 仅在不同段落之间插入，同一段落内的句子直接拼接

### 5.6 英文专有名词映射

| 单词 | 音标 |
|------|------|
| Kokoro | kˈOkəɹO |
| Sol | sˈOl |
| Vale | vˈAɪl |
| Maple | mˈAːpəl |

通过 `en_callable` 回调函数硬编码映射，其他英文文本由 `misaki.en.G2P()` 自动处理。

---

## 六、GPU 警告信息说明

运行过程中 ONNX Runtime 输出的以下警告**不影响推理结果**：

| 警告 | 说明 |
|------|------|
| `Memcpy nodes are added to the graph` | CUDA EP 在 CPU/GPU 之间自动插入数据传输节点，正常行为 |
| `Some nodes were not assigned to the preferred EPs` | 部分形状相关算子被 ORT 自动分配至 CPU，属于性能优化策略 |
| `ScatterND with reduction=='none'` | 仅提醒非原子操作的 ScatterND 不保证重复索引的正确性，模型内部索引无重复 |

---

## 七、声音风格文件列表

```
voices/
├── zf_001.bin ~ zf_099.bin  ← 中文女声 (51 个)
├── zm_009.bin ~ zm_100.bin  ← 中文男声 (45 个)
├── af_maple.bin             ← 英文女声 Maple
├── af_sol.bin               ← 英文女声 Sol
└── bf_vale.bin              ← 英文男声 Vale
```

---

## 八、命令行用法

```bash
# 使用默认文本推理
python Kokoro-82M-v1.1-zh-ONNX-q4.py

# 指定自定义文本
python Kokoro-82M-v1.1-zh-ONNX-q4.py "你好世界，这是一段测试文本。"

# 多段文本 (段落间自动插入静音)
python Kokoro-82M-v1.1-zh-ONNX-q4.py "第一段文本。\n第二段文本。"
```
