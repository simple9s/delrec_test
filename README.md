# DELRec: Distilling Sequential Pattern to Enhance LLMs-based Sequential Recommendation

## 概述

DELRec 是一个将传统序列推荐（SR）模型的行为模式蒸馏进大语言模型（LLM）的推荐框架。本仓库在原始论文实现基础上，新增了以下功能：

- **LLM 后端**：支持 `meta-llama/Llama-3.2-3B-Instruct`（8-bit 量化，默认）
- **数据集**：支持 Amazon 2018 / 2023 评论数据集，自动下载 & 缓存
- **评估协议**：留一法（Leave-One-Out）+ 100 选一，指标为 **Hit@K / NDCG@K**（K = 1, 5, 10, 20）

---

## 论文

**DELRec: Distilling Sequential Pattern to Enhance LLMs-based Sequential Recommendation**

---

## 目录结构

```
DELRec/
├── main.py                                      # 训练 & 测试主入口
├── utils.py                                     # Hit@K / NDCG@K 指标工具
├── dataload.py                                  # 数据分区 & 留一法工具
├── test_DELRec.py                               # 测试入口（LLaMA3 / openprompt 路由）
│
├── data/
│   ├── download.py                              # Amazon 数据集自动下载 & 解压
│   └── amazon_loader.py                         # 三层智能加载（缓存→文件→下载）
│
├── llms/
│   └── llama3_wrapper.py                        # LLaMA 3-3B-Instruct 封装
│       ├── LLaMA3Recommender                    #   主模型类（含分类头 & soft-prompt）
│       ├── load_llama3_tokenizer()              #   分词器加载
│       ├── build_llama3_prompt()               #   Instruct 格式 prompt 构建
│       └── collate_llama3_batch()              #   批次整理函数
│
├── llms_based_sr/
│   ├── train.py                                 # 第二阶段训练（LLaMA3 / openprompt 路由）
│   ├── amazon_lsr_dataset.py                    # Amazon → PromptDataLoader（openprompt 用）
│   └── llms_based_sequential_recommendation.py # 原始 LSR 数据加载
│
├── MTL/
│   ├── MTL.py                                   # 多任务学习动态损失权重
│   └── HydaLearn.py                             # 异常检测自编码器
│
├── SR_models/
│   ├── SR_models.py                             # GRU / Caser / SASRec 实现
│   ├── GRU.py                                   # GRU 训练脚本
│   ├── Caser.py                                 # Caser 训练脚本
│   └── SASRec.py                                # SASRec 训练脚本
│
├── distill_pattern_from_conventional_SR_models/
│   ├── train.py                                 # 第一阶段训练（TA + RPS 多任务）
│   ├── temporal_analysis.py                     # TA 数据加载
│   └── recommendation_pattern_simulating.py     # RPS 数据加载
│
├── prompt_construction/
│   ├── LSR.txt                                  # LSR 阶段 prompt 模板
│   ├── RPS.txt                                  # RPS 阶段 prompt 模板
│   └── TA.txt                                   # TA 阶段 prompt 模板
│
└── requirements.txt                             # 依赖列表
```

---

## 环境配置

### 1. 克隆仓库

```bash
git clone https://github.com/haoge6660101/DELRec_hao.git
cd DELRec
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖版本：

| 包 | 版本 |
|---|---|
| torch | 2.1.0+cu121 |
| transformers | 4.35.0 |
| peft | 0.10.0 |
| bitsandbytes | ≥ 0.41.0 |
| openprompt | 1.0.1 |
| pytorch_lightning | 2.4.0 |

### 3. 下载 LLaMA 3-3B-Instruct

```bash
# 方法一：HuggingFace Hub（需申请访问权限）
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct \
    --local-dir ./Llama-3.2-3B-Instruct

# 方法二：直接在代码中使用 Hub ID（自动下载）
--llm_path meta-llama/Llama-3.2-3B-Instruct
```

---

## 数据集

### 支持的数据集

| 数据集 | 版本 | 下载源 |
|---|---|---|
| Movies and TV | 2018 / 2023 | McAuley Lab |
| Sports and Outdoors | 2018 / 2023 | McAuley Lab |
| Beauty | 2018 / 2023 | McAuley Lab |
| Home and Kitchen | 2018 / 2023 | McAuley Lab |
| Steam（游戏） | — | HuggingFace |

### 自动下载（推荐）

数据集会在第一次运行时**自动下载**，无需手动操作：

```bash
# 单独下载（可选）
python -m DELRec.data.download --year 2018 --category Movies_and_TV
python -m DELRec.data.download --year 2023 --category Sports_and_Outdoors
```

### 智能缓存机制

数据加载采用三层判断，避免重复操作：

```
第 1 层：./cache/*.pkl 存在？  → 直接返回，跳过一切处理
第 2 层：原始 json/jsonl 存在？ → 跳过下载，直接解析构建
第 3 层：文件不存在？           → 自动触发网络下载 & 解压
```

---

## 评估协议

### 留一法（Leave-One-Out）

每位用户的交互序列按时间排序后：

| 划分 | 正样本位置 | 用途 |
|---|---|---|
| test | 最后 1 个 item | 测试评估 |
| val | 倒数第 2 个 item | 验证集 |
| train | 倒数第 3 个 item（序列长度 ≥ 4）| 训练 |

### 100 选一

每条评估记录包含 **1 个正样本 + 99 个随机负样本**，共 100 个候选。

### 评估指标

$$\text{Hit@K} = \frac{1}{|U|} \sum_{u} \mathbf{1}[\text{rank}_u \leq K]$$

$$\text{NDCG@K} = \frac{1}{|U|} \sum_{u} \frac{\mathbf{1}[\text{rank}_u \leq K]}{\log_2(\text{rank}_u + 1)}$$

默认 K ∈ {1, 5, 10, 20}，以 **Hit@10** 为模型保存主指标。

---

## 快速开始

### 训练（LLaMA 3 + Amazon 2018，自动下载数据）

```bash
python main.py \
    --llm llama3 \
    --llm_path meta-llama/Llama-3.2-3B-Instruct \
    --mode train \
    --amazon_version 2018 \
    --amazon_category Luxury_Beauty \
    --num_candidates 100 \
    --eval_ks 1 5 10 20 \
    --second_batch_size 8 \
    --second_total_epoch 2 \
    --first_total_epoch  2 \
    --second_model_path ./model_beauty
```

### 测试

```bash
python -m DELRec.main \
    --llm llama3 \
    --llm_path ./Llama-3.2-3B-Instruct \
    --mode test \
    --amazon_version 2018 \
    --amazon_category Movies_and_TV \
    --second_model_path ./model_llama3
```

### 使用原始 T5 流程（两阶段训练）

```bash
python -m DELRec.main \
    --llm t5 \
    --llm_path ./flan-t5-xl \
    --mode train \
    --SR_model SASRec
```

### 切换 Amazon 2023 数据集

```bash
python -m DELRec.main \
    --llm llama3 \
    --llm_path ./Llama-3.2-3B-Instruct \
    --mode train \
    --amazon_version 2023 \
    --amazon_category Sports_and_Outdoors
```

---

## 量化配置

LLaMA 3-3B-Instruct 默认使用 **8-bit 量化**（bitsandbytes），平衡精度与显存消耗。

| 模式 | 参数 | 显存占用（3B 模型）| 适用场景 |
|---|---|---|---|
| 8-bit（默认） | `--llama3_load_8bit True` | ~6 GB | 单张 A100 / 3090 |
| 4-bit | `--llama3_load_4bit True` | ~3 GB | 显存极度不足 |
| 全精度 bf16 | 两者均为 False | ~12 GB | 精度优先 |

> **注意**：`--llama3_load_4bit` 和 `--llama3_load_8bit` 同时为 True 时，自动以 4-bit 为准。

---

## 主要参数说明

### 基础参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--llm` | `llama3` | LLM 类型，支持 llama3 / t5 / roberta 等 |
| `--llm_path` | `./Llama-3.2-3B-Instruct` | 模型本地路径或 Hub ID |
| `--SR_model` | `SASRec` | 传统 SR 模型，SASRec / Caser / GRU |
| `--mode` | `train` | 运行模式，train / test |
| `--seed` | `1234` | 随机种子 |
| `--device` | `cuda` | 运行设备 |

### Amazon 数据集参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--use_amazon` | `True` | 是否使用 Amazon 数据集 |
| `--amazon_version` | `2018` | 数据集版本，2018 / 2023 |
| `--amazon_category` | `Movies_and_TV` | 类别名称 |
| `--amazon_review_path` | `""` | 手动指定评论文件路径（空 = 自动处理）|
| `--amazon_meta_path` | `""` | 手动指定元数据路径（空 = 自动处理）|
| `--amazon_cache_dir` | `./cache` | 缓存目录 |
| `--amazon_min_inter` | `5` | 用户最少交互次数过滤 |
| `--amazon_force_rebuild` | `False` | 强制忽略缓存重新构建 |

### 评估参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--eval_ks` | `1 5 10 20` | 评估 K 列表 |
| `--num_candidates` | `100` | 候选集大小（1 正 + 99 负）|

### 第二阶段训练参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--second_batch_size` | `8` | 批大小（LLaMA3 建议 ≤ 8）|
| `--second_total_epoch` | `20` | 训练轮数 |
| `--second_lr` | `2e-4` | 学习率 |
| `--second_max_seq_length` | `1024` | 最大 token 长度 |
| `--second_model_path` | `./model_llama3` | 模型保存路径 |
| `--second_if_peft` | `True` | 是否使用 PEFT（LoRA）|
| `--second_lora_r` | `16` | LoRA 秩（LLaMA3-3B 建议 8~16）|
| `--second_lora_alpha` | `32` | LoRA scaling（通常为 2 × lora_r）|
| `--second_lora_dropout` | `0.05` | LoRA dropout |
| `--soft_prompt_len` | `100` | Soft-prompt token 数量 |

---

## 框架流程

```
Amazon 数据集
    │
    ▼
[data/download.py]  ←── 三层智能判断（缓存 / 文件 / 下载）
    │
    ▼
[data/amazon_loader.py]
    ├── 解析 JSON-lines / Parquet
    ├── 留一法切分（train / val / test）
    └── 100 选一候选构建
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│                       完整两阶段训练                           │
│                                                              │
│  第一阶段：蒸馏 SR 模型行为模式 → soft-prompt                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  T5 流程（openprompt）   LLaMA3 流程（原生 PyTorch）  │    │
│  │  TA task                TA task                     │    │
│  │  RPS task               RPS task                    │    │
│  │  动态损失权重（MTL）      动态损失权重（MTL）            │    │
│  │  ↓ soft-prompt 权重保存  ↓ soft-prompt 权重保存       │    │
│  └─────────────────────────────────────────────────────┘    │
│                             │                                │
│                             ▼ 加载 soft-prompt（冻结）        │
│                                                              │
│  第二阶段：fine-tune LLM → LSR 推荐任务                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  T5 流程（openprompt）   LLaMA3 流程                  │    │
│  │  AdaLoRA 微调            AdaLoRA 微调 q_proj/v_proj   │    │
│  │  100 选一分类             100 选一分类头                │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
[utils.py]  Hit@K / NDCG@K (K=1,5,10,20)
```

---

## 传统 SR 模型预训练

在运行 DELRec 之前，需要先用传统 SR 模型生成用户交互预测文件：

```bash
# SASRec
python -m DELRec.SR_models.SASRec

# GRU
python -m DELRec.SR_models.GRU

# Caser
python -m DELRec.SR_models.Caser
```

生成的文件（`user_interactions_with_text_title_and_predicted_items_by_*.txt`）供第一阶段训练使用。

---

## 常见问题

**Q：提示 CUDA out of memory**

优先开启 8-bit 量化（默认已开启），若仍不足可尝试 4-bit：
```bash
--llama3_load_4bit True --second_batch_size 4
```

**Q：数据集下载失败**

可手动下载后通过 `--amazon_review_path` 和 `--amazon_meta_path` 指定路径，程序会跳过下载直接解析。

**Q：如何强制重新处理数据**

```bash
--amazon_force_rebuild True
```

**Q：LLaMA3 与原始 T5 流程的区别**

两个流程都完整执行两个阶段，核心差异在实现方式：

| | T5 流程 | LLaMA3 流程 |
|---|---|---|
| 第一阶段框架 | openprompt | 原生 PyTorch（`llama3_stage1.py`）|
| 第一阶段任务 | TA + RPS 双任务 | TA + RPS 双任务（相同）|
| soft-prompt 训练 | openprompt MixedTemplate | 自定义 `nn.Embedding`，直接拼接到 input embeddings |
| 第二阶段微调 | openprompt + AdaLoRA | 原生 transformers + AdaLoRA |
| 量化 | 无 | 默认 8-bit bitsandbytes |

第一阶段结束后，soft-prompt 权重保存至 `--first_learned_soft_prompt_path`，
第二阶段自动加载并冻结，只对 LLM 做 AdaLoRA 微调。

---

## 引用

```bibtex
@article{delrec2024,
  title={DELRec: Distilling Sequential Pattern to Enhance LLMs-based Sequential Recommendation},
  author={...},
  year={2024}
}
```

---

## License

Apache License 2.0
