"""
DELRec/llms_based_sr/amazon_lsr_dataset.py
==========================================
将 Amazon 2018/2023 数据集接入 DELRec 的 LSR（LLMs-based Sequential Recommendation）流程。
- 留一法（Leave-One-Out）切分
- 100 选一候选构建
- 转为 PromptDataLoader
"""

import pickle
import os
from openprompt import PromptDataLoader
from openprompt.plms import load_plm

from data.amazon_loader import load_amazon_dataset
from dataload import amazon_data_partition, amazon_to_input_examples
from utils import create_prompt


def load_amazon_LSR_dataset(args):
    """
    加载 Amazon 数据集并返回 PromptDataLoader（train / test / val）。
    参数均从 args 中读取（与 main.py 参数命名对应）。

    新增 args 字段（在 main.py 中已注册）:
        --use_amazon
        --amazon_version      '2018' | '2023'
        --amazon_review_path
        --amazon_meta_path
        --amazon_cache_dir
        --amazon_min_inter
        --amazon_force_rebuild
        --num_candidates      100（1 正 + 99 负）
        --eval_ks             [1, 5, 10, 20]
    """

    # ── 1. 加载 & 缓存 Amazon 原始数据 ─────────────────────────────────────────
    splits, item2title = load_amazon_dataset(
        dataset_version=args.amazon_version,
        category=getattr(args, 'amazon_category', 'Movies_and_TV'),
        review_path=getattr(args, 'amazon_review_path', '') or None,
        meta_path=getattr(args, 'amazon_meta_path', '') or None,
        cache_dir=getattr(args, 'amazon_cache_dir', './cache'),
        num_candidates=getattr(args, 'num_candidates', 100),
        min_interactions=getattr(args, 'amazon_min_inter', 5),
        seed=getattr(args, 'seed', 42),
        force_rebuild=getattr(args, 'amazon_force_rebuild', False),
    )

    # ── 2. 转为 InputExample ───────────────────────────────────────────────────
    model_name = getattr(args, 'SR_model', 'SASRec')
    train_examples = amazon_to_input_examples(splits['train'], model_name)
    val_examples   = amazon_to_input_examples(splits['val'],   model_name)
    test_examples  = amazon_to_input_examples(splits['test'],  model_name)

    # ── 3. 缓存 InputExample（可选，加速重复实验）──────────────────────────────
    cache_dir = getattr(args, 'amazon_cache_dir', './cache')
    os.makedirs(cache_dir, exist_ok=True)

    # ── 4. 构建 PromptDataLoader ───────────────────────────────────────────────
    plm, tokenizer, model_config, WrapperClass = load_plm(args.llm, args.llm_path)
    mytemplate = create_prompt('LSR', plm, tokenizer, prompt_id=args.LSR_prompt_id)

    def _make_loader(examples, shuffle):
        return PromptDataLoader(
            dataset=examples,
            template=mytemplate,
            tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass,
            max_seq_length=args.first_max_seq_length,
            decoder_max_length=args.first_decoder_max_length,
            batch_size=args.first_batch_size,
            shuffle=shuffle,
            teacher_forcing=args.first_teacher_forcing,
            predict_eos_token=args.first_predict_eos_token,
            truncate_method=args.first_truncate_method,
        )

    train_dataloader = _make_loader(train_examples, shuffle=True)
    val_dataloader   = _make_loader(val_examples,   shuffle=False)
    test_dataloader  = _make_loader(test_examples,  shuffle=False)

    print(f"[Amazon LSR DataLoader] "
          f"train_batches={len(train_dataloader)}  "
          f"val_batches={len(val_dataloader)}  "
          f"test_batches={len(test_dataloader)}")

    return train_dataloader, test_dataloader, val_dataloader