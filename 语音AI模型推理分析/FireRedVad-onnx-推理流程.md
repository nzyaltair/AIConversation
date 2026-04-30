# FireRedVad ONNX 推理流程与数据总结

## 一、概述

FireRedVad 是基于 ONNX Runtime 的语音活动检测（Voice Activity Detection）推理脚本，完整实现了 **音频输入 → 特征提取 → 神经网络推理 → 后处理 → 语音时间戳输出** 的端到端流程。

| 项目 | 说明 |
|------|------|
| 模型框架 | PyTorch → ONNX 导出 |
| 推理引擎 | ONNX Runtime（支持 CUDA / CPU） |
| 本地 GPU | NVIDIA RTX 3060 (6 GB), CUDA 13.2 |
| ONNX Runtime | `onnxruntime-gpu` 1.25.0 |
| 依赖库 | `numpy`, `soundfile`, `kaldi-native-fbank`, `kaldiio`, `scipy` |

---

## 二、模型结构

### 2.1 模型文件

| 文件 | 说明 |
|------|------|
| [`model.onnx`](file:///d:/_Projects/_GraduationProject/ConvertToPython/models/FireRedVad-onnx/model.onnx) | ONNX 格式的 VAD 神经网络 |
| [`cmvn.ark`](file:///d:/_Projects/_GraduationProject/ConvertToPython/models/FireRedVad-onnx/cmvn.ark) | Kaldi 格式的 CMVN 归一化统计量 |
| [`configuration.json`](file:///d:/_Projects/_GraduationProject/ConvertToPython/models/FireRedVad-onnx/configuration.json) | 模型元信息（PyTorch 框架，ASR 任务） |

### 2.2 ONNX 模型 I/O 定义

#### 输入（共 9 个张量）

| 名称 | 形状 | 数据类型 | 说明 |
|------|------|----------|------|
| `feat` | `(1, T_chunk, 80)` | `float32` | 音频 FBank 特征（batch=1） |
| `cache_0` ~ `cache_7` | `(1, 128, 19)` | `float32` | 8 个 RNN 隐状态缓存 |

- **`cache` 的作用**：携带若干历史帧的上下文信息，使模型在分块推理时感知前后文的时序依赖。
- **`128`**（`P`）：RNN 隐状态维度。
- **`19`**（`C` = `lookback_padding`）：当前块末尾向前回溯的上下文帧数，用于缓存下一块的起始状态。
- **`T_chunk`**：当前块的帧数，为动态维度，最长受 `chunk_max_frame` 控制（默认 30000 帧 ≈ 300 秒）。

#### 输出（共 9 个张量）

| 名称 | 形状 | 数据类型 | 说明 |
|------|------|----------|------|
| `output` | `(1, T_chunk)` | `float32` | 每帧的语音概率（0~1） |
| `cache_0` ~ `cache_7` | `(1, 128, 19)` | `float32` | 更新后的 RNN 隐状态缓存 |

---

## 三、完整推理流程

```
┌───────────┐    ┌────────────┐    ┌──────────┐    ┌─────────────┐    ┌──────────┐    ┌──────────┐
│ 音频输入  │───→│ 重采样     │───→│ FBank    │───→│ CMVN       │───→│ ONNX    │───→│ 后处理  │───→ 时间戳
│ (wav)    │    │ (→16kHz)  │    │ 特征提取 │    │ 归一化     │    │ 推理    │    │ 平滑/合并│    输出
└───────────┘    └────────────┘    └──────────┘    └─────────────┘    └──────────┘    └──────────┘
```

### 步骤 1：音频输入与重采样

- **输入格式**：WAV 文件路径、或 `(采样率, 波形)` 元组、或裸 NumPy 数组。
- **重采样**：若采样率 ≠ 16000 Hz，使用 `scipy.signal.resample` 进行带抗混叠的重采样到 16kHz。
- **输出**：单声道 float64 波形数组，采样率统一为 16000 Hz。

```python
# 重采样计算
duration = wav_np.shape[0] / sample_rate
new_length = int(duration * 16000)
wav_np = signal.resample(wav_np, new_length)
```

### 步骤 2：FBank 特征提取

| 参数 | 值 | 说明 |
|------|-----|------|
| 采样率 | 16000 Hz | 固定 |
| 帧长 | 25 ms | `frame_length_ms` |
| 帧移 | 10 ms | `frame_shift_ms` |
| Mel 滤波器组数 | 80 | `num_mel_bins` |
| Dither | 0（推理时关闭） | 训练时可开启 |

- **工具库**：`kaldi-native-fbank`（`kaldi_native_fbank` Python 绑定）。
- **引擎**：`OnlineFbank`，流式接受波形样本，逐帧获取特征。
- **输出形状**：`(T, 80)`，T = 帧数 ≈ `音频秒数 × 100`（因为帧移 10ms）。

### 步骤 3：CMVN 归一化

- **来源**：Kaldi 格式的 `cmvn.ark` 文件，包含训练集的全局均值与方差统计量。
- **解析逻辑**：

  ```
  stats[0, :dim] / count           → means（均值）
  stats[1, :dim] / count - μ²      → variance（方差）
  1 / √(max(variance, 1e-20))      → inverse_std（标准差倒数）
  ```

- **应用**：`(X - means) × inverse_std`，逐维度进行 z-score 标准化。
- **输出形状**：`(T, 80)`，`float32`。

### 步骤 4：ONNX 分块推理

由于音频可能很长（最长支持 300 秒），推理采用**滑动分块 + 缓存传递**策略：

```
初始化：
  caches = [zeros(1,128,19) × 8]     # 8 个 cache 全零初始化

对每个分块 (start=0; start<T; start+=chunk_max_frame):
  chunk_feat = feats[start:end]      # 形状 (T_chunk, 80)
  feed = {feat: chunk_feat, cache_0..7: caches}
  → 调用 sess.run() 得到 (probs_chunk, new_cache_0..7)
  → caches = new_caches              # 更新缓存供下一块使用
```

#### 关键数据流转

| 阶段 | 输入形状 | 输出形状 |
|------|----------|----------|
| 特征输入 | `(T, 80)` | — |
| 单块推理 | `(1, T_chunk, 80)` + 8×`(1, 128, 19)` | `(1, T_chunk)` + 8×`(1, 128, 19)` |
| 拼接概率 | 多段 `(T_i,)` | `(T,)` 总帧数概率向量 |

### 步骤 5：后处理

后处理包含 **5 个子步骤**，将原始帧级概率转换为最终的时间戳：

#### 5.1 滑动窗口平滑

```python
kernel = [0.2, 0.2, 0.2, 0.2, 0.2]   # 窗口大小 = 5
smoothed = np.convolve(probs, kernel, mode='same')
```

#### 5.2 阈值二值化

```
speech_threshold = 0.4
decisions[i] = 1 if smoothed[i] > 0.4 else 0
```

#### 5.3 过滤 / 截断语音段

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `min_speech_frame` | 20 | 小于 20 帧（0.2 秒）的语音段被丢弃 |
| `max_speech_frame` | 2000 | 超过 2000 帧（20 秒）的语音段被截断 |

#### 5.4 合并短静音间隔

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `merge_silence_frame` | 20 | 相邻语音段之间的静音 ≤ 20 帧（0.2 秒）时，合并为一个段 |

#### 5.5 扩展语音边界

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `extend_speech_frame` | 20 | 每个语音段的起止各向外扩展 20 帧（0.2 秒） |

#### 5.6 帧索引 → 时间转换

```
开始时间 = start_frame × 0.01    （帧移 = 10ms）
结束时间 = end_frame × 0.01
```

---

## 四、配置参数汇总

| 参数 | 默认值 | 单位 | 说明 |
|------|--------|------|------|
| `use_gpu` | `True` | — | 是否使用 CUDA 加速 |
| `smooth_window_size` | 5 | 帧（50ms） | 概率平滑窗口 |
| `speech_threshold` | 0.4 | — | 语音概率判定阈值 |
| `min_speech_frame` | 20 | 帧（200ms） | 最短语音长度 |
| `max_speech_frame` | 2000 | 帧（20s） | 最长语音长度 |
| `min_silence_frame` | 20 | 帧（200ms） | （未使用） |
| `merge_silence_frame` | 20 | 帧（200ms） | 合并 ≤ 此值的静音间隔 |
| `extend_speech_frame` | 20 | 帧（200ms） | 语音段边界外扩量 |
| `chunk_max_frame` | 30000 | 帧（300s） | ONNX 单次推理最大帧数 |

---

## 五、测试结果

> 测试环境：Windows 11、Python 3.12、ONNX Runtime 1.25 GPU、RTX 3060

### asr_en.wav（英文，15.05 秒）

| 统计项 | 值 |
|--------|-----|
| 特征帧数 | 1503 |
| 概率均值 | 0.8026 |
| 超阈值帧数 | 1219 / 1503 |
| 检测到语音段 | 3 段 |

| 序号 | 开始 | 结束 | 时长 |
|------|------|------|------|
| 1 | 0.210s | 5.720s | 5.51s |
| 2 | 5.790s | 10.810s | 5.02s |
| 3 | 10.880s | 14.810s | 3.93s |

### asr_zh.wav（中文，4.20 秒）

| 统计项 | 值 |
|--------|-----|
| 特征帧数 | 418 |
| 概率均值 | 0.7701 |
| 超阈值帧数 | 324 / 418 |
| 检测到语音段 | 1 段 |

| 序号 | 开始 | 结束 | 时长 |
|------|------|------|------|
| 1 | 0.240s | 3.880s | 3.64s |

---

## 六、输入 / 输出 API

### 输入：`ONNXFireRedVad.detect(audio, do_postprocess=True)`

| 参数 | 类型 | 说明 |
|------|------|------|
| `audio` | `str` \| `(int, np.ndarray)` \| `np.ndarray` | 音频：文件路径 / (采样率, 波形) / 裸波形数组 |
| `do_postprocess` | `bool` | `True` 返回时间戳；`False` 返回原始概率 |

### 输出：`(result_dict, probs_array)`

```python
result = {
    "dur": 15.051,                # 音频总时长（秒）
    "timestamps": [               # 检测到的语音段时间戳
        [0.210, 5.720],           # [开始秒, 结束秒]
        [5.790, 10.810],
        [10.880, 14.810]
    ],
    "wav_path": "xxx.wav"         # 仅输入为文件路径时出现
}
probs = np.array([0.99, ...])     # 每帧语音概率 (T,)
```
