## 项目简介

Fredformer是一个基于Transformer架构的时间序列预测模型，通过频率去偏机制提升长期预测性能。本项目基于该模型实现了完整的时间序列预测流程，包括数据预处理、模型训练、评估和推理。

## 主要特性

- **Fredformer模型实现**：完整的模型架构实现，包括频率分解、交叉注意力机制等核心组件
- **网格搜索**：支持超参数自动搜索，寻找最佳模型配置
- **数据处理**：支持多种数据预处理策略，包括水质数据等时间序列数据
- **模型评估**：提供完整的评估指标和可视化工具


## 项目结构

```
.
├── baselines/
│   └── Fredformer/          # Fredformer模型实现
│       ├── arch/             # 模型架构
│       ├── grid_search.py    # 网格搜索脚本
│       └── WLSQ.py           # 配置文件
├── basicts/                  # 时间序列框架
├── datasets/                  # 数据集目录
├── experiments/              # 实验脚本
│   ├── train.py              # 训练脚本
│   ├── evaluate.py           # 评估脚本
├── scripts/                   # 工具脚本
│   ├── data_preparation/     # 数据预处理
│   └── data_visualization/   # 数据可视化
└── checkpoints/               # 模型检查点

```

## 环境要求

- Python 3.6+
- PyTorch 1.8+
- 其他依赖见 `requirements.txt`

## 快速开始

### 1. 数据准备

将处理好的数据放在 `datasets/data_source/` 目录下，运行数据处理脚本：

```bash
python datasets/cov.py
```

### 2. 训练模型

```bash
python experiments/train.py -c baselines/Fredformer/ -g 0
```

### 3. 网格搜索

```bash
python baselines/Fredformer/grid_search.py -g 0
```

### 4. 评估模型

```bash
python experiments/evaluate.py -cfg baselines/Fredformer/WLSQ.py -ckpt <checkpoint_path> -g 0
```


## 模型配置

主要超参数在 `baselines/Fredformer/WLSQ.py` 中配置，包括：

- `patch_len`: 子频段长度
- `cf_depth`: 模型深度（编码器层数）
- `cf_heads`: 多头注意力数量
- `cf_dim`: 自注意力特征维度
- 其他模型参数

## 数据集

项目支持多种时间序列数据集，主要使用WLSQ等水质数据集。数据集应放置在 `datasets/` 目录下，包含 `data.dat` 和 `desc.json` 文件。



