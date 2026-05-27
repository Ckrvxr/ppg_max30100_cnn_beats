# PPG 脉搏波标注 & 训练工具使用说明

工作目录下所有脚本均位于 `code/` 目录下，通过 `pixi run` 执行。

## 脚本总览

| 脚本 | 功能 | 依赖模型 |
|------|------|---------|
| `prelabel.py` | ONNX 自动预标记 | `model/*.onnx` |
| `label_tool.py` | 手动波峰标注（GUI） | 无 |
| `train.py` | 训练 CNN + 导出 `.pth` / `.h` | 需已标注数据 |
| `verify.py` | 模型推理结果可视化 | `model/*.pth` |

## 通用选项

4 个脚本统一使用 `argparse` 解析参数，均支持 `--help` 查看用法：

```
pixi run <task> --help
```

所有共享配置（模型路径、训练超参、阈值等）集中在 `ppg_config.json`。

---

## 1. 自动预标记 — `prelabel.py`

### 用法

```
pixi run prelabel <input> [选项]
```

### 参数

| 参数 | 说明 |
|------|------|
| `input` | 输入文件（`.txt` 或 `.csv`），必须含 `IR,RED` 列 |
| `--output, -o` | 输出 CSV 路径（默认覆盖输入文件） |
| `--config, -c` | 配置文件路径（默认 `ppg_config.json`） |
| `--threshold` | 检测阈值，覆盖配置文件中的值 |

### 示例

```bash
# 对原始 txt 数据预标记，结果输出到 labeled/
pixi run prelabel raw/raw_data.txt -o labeled/raw_labeled.csv

# 对已有 CSV 重新用更高阈值预标记
pixi run prelabel labeled/old.csv -o labeled/new.csv --threshold 0.7

# 使用自定义配置文件
pixi run prelabel raw/data.txt --config my_config.json
```

### 输出

生成包含 `IR, RED, beat_event` 三列的 CSV 文件。`beat_event=1` 表示检测到的脉搏事件。

---

## 2. 手动标注 — `label_tool.py`

### 用法

```
pixi run label [input] [选项]
```

### 参数

| 参数 | 说明 |
|------|------|
| `input` | 输入文件（默认 `raw/raw_data.txt`） |
| `--output, -o` | 输出 CSV（默认 `labeled/labeled_data.csv`） |
| `--view-window` | 每页显示点数（默认 1000） |

### 交互操作

| 操作 | 作用 |
|------|------|
| **左键单击** | 在波峰处标记脉搏事件 |
| **右键单击** | 擦除最近的标记 |
| **→** 右方向键 | 保存进度并翻到下一页 |
| **←** 左方向键 | 回退到上一页 |
| **ESC** | 保存并退出 |

### 示例

```bash
# 快速启动（使用默认路径）
pixi run label

# 指定文件和输出
pixi run label raw/2026-05-27_data.txt -o labeled/my_labeled.csv

# 每页只看 500 点
pixi run label raw/data.txt --view-window 500
```

### 说明

- 第一次打开原始数据时全部 `beat_event` 为 0，需手动标记
- 如果输出文件已存在且结构与数据匹配，会自动恢复之前的标注进度
- 左右翻页时会自动保存，ESC 退出时也会保存

---

## 3. 模型训练 — `train.py`

### 用法

```
pixi run train [选项]
```

### 参数

| 参数 | 说明 |
|------|------|
| `--config, -c` | 配置文件路径（默认 `ppg_config.json`） |
| `--epochs` | 覆盖训练轮数 |
| `--lr` | 覆盖学习率 |
| `--batch-size` | 覆盖 batch size |

### 示例

```bash
# 使用配置文件默认参数训练
pixi run train

# 快速迭代（20 轮 + 大学习率）
pixi run train --epochs 20 --lr 0.005

# 使用自定义配置
pixi run train --config my_config.json
```

### 流程说明

1. **合并数据** — 读取 `ppg_config.json` 中 `data.sources` 指定的所有源文件
   - `.csv` 文件直接读取标注数据
   - `.txt` 文件自动调用 ONNX 预标记并缓存到 `labeled/` 下
   - 全部合并写入 `data.dataset` 指定的路径
2. **训练** — 使用类别平衡采样训练 `McuPpgNet`，验证集占比 20%
3. **评估** — 输出敏感度、PPV、F1、AUPR 等指标
4. **导出** — 同时输出 `.pth`（PyTorch 权重）和 `.h`（C 语言头文件，含 BN 融合后的权重）

### 输出产物

| 文件 | 说明 |
|------|------|
| `labeled/merged.csv` | 合并后的训练数据集 |
| `model/ppg_mcu_model.pth` | PyTorch 模型权重 |
| `model/ppg_cnn_weights.h` | 嵌入式 C 语言权重头文件 |

---

## 4. 验证可视化 — `verify.py`

### 用法

```
pixi run verify [选项]
```

### 参数

| 参数 | 说明 |
|------|------|
| `--config, -c` | 配置文件路径（默认 `ppg_config.json`） |
| `--source` | 覆盖数据源列表（可指定多个） |

### 示例

```bash
# 使用配置文件中的数据源
pixi run verify

# 指定要验证的文件
pixi run verify --source labeled/2026-05-27_data.csv

# 同时验证多个文件
pixi run verify --source raw/test1.txt raw/test2.txt
```

### 可视化面板

共 6 行子图，从上到下：

| 行 | 内容 |
|----|------|
| 1 | IR 原始波形（EWM 高通滤波后） |
| 2 | 模型输出的原始概率 |
| 3 | 滑动平均（box filter, kernel=5） |
| 4 | 指数加权平均（EWM α=0.3） |
| 5 | 中值滤波（medfilt, kernel=5） |
| 6 | 积分平滑（box filter, kernel=20）+ 触发阈值线 |

阈值线来自 `ppg_config.json` 中 `prelabel.threshold` 的值。

### 交互操作

与 label_tool 一致：`→` 翻页、`←` 回退、`ESC` 退出。

---

## 推荐工作流程

```
原始数据 (.txt)
    │
    ▼
[prelabel]  ──→  自动预标记 CSV
    │
    ▼
[label_tool] ──→  人工修正标注（可选）
    │
    ▼
[train] ──→  训练模型 → .pth + .h
    │
    ▼
[verify] ──→  可视化验证推理效果
```

1. 先用 `prelabel.py` 批量自动预标记
2. 对有问题的数据用 `label_tool.py` 手动修正
3. 用 `train.py` 训练模型
4. 用 `verify.py` 验证效果，如需改进则回到步骤 2

---

## 配置文件参考 (`ppg_config.json`)

```json
{
    "data": {
        "sources": ["labeled/file1.csv", "labeled/file2.csv"],
        "dataset": "labeled/merged.csv"
    },
    "model": {
        "onnx": "model/ppg_mcu_model.onnx",
        "pth": "model/ppg_mcu_model.pth",
        "h": "model/ppg_cnn_weights.h"
    },
    "prelabel": {
        "threshold": 0.5,
        "refractory": 20
    },
    "train": {
        "window_size": 100,
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 0.001,
        "num_samples_epoch": 8000
    }
}
```
