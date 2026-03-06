"""
DELRec/llms_based_sr/train.py
==============================
第二阶段训练（LSR）：
- 支持 LLaMA 3-3B-Instruct（--llm llama3）
- 支持 Amazon 数据集（--use_amazon）
- 评估：Hit@K / NDCG@K，K ∈ {1, 5, 10, 20}，留一法 + 100 选一
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from utils import (
    init_metrics, update_metrics, finalize_metrics,
    metrics_to_str, EVAL_KS,
)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset wrapper（纯字典列表 → DataLoader）
# ─────────────────────────────────────────────────────────────────────────────

class AmazonRecommendDataset(Dataset):
    def __init__(self, data_list: list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 内部辅助：构建 DataLoader
# ─────────────────────────────────────────────────────────────────────────────

def _build_dataloader(data_list: list, batch_size: int,
                      shuffle: bool, collate_fn) -> DataLoader:
    ds = AmazonRecommendDataset(data_list)
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=shuffle, collate_fn=collate_fn)


# ─────────────────────────────────────────────────────────────────────────────
# LLaMA 3 训练流程
# ─────────────────────────────────────────────────────────────────────────────

def training_of_second_stage_llama3(args, learned_soft_prompt_path: str = None):
    """
    使用 LLaMA 3-3B-Instruct 进行第二阶段 LSR 训练。
    learned_soft_prompt_path：第一阶段保存的 stage1.pt 所在目录。
    """
    from llms.llama3_wrapper import (
        LLaMA3Recommender, load_llama3_tokenizer, collate_llama3_batch
    )

    ks = getattr(args, 'eval_ks', EVAL_KS)
    num_candidates = getattr(args, 'num_candidates', 100)

    # ── 数据加载（与第一阶段共用同一份 splits）────────────────────────────────
    from data.amazon_loader import load_amazon_dataset

    splits, item2title = load_amazon_dataset(
        dataset_version=args.amazon_version,
        category=args.amazon_category,
        review_path=getattr(args, 'amazon_review_path', None) or None,
        meta_path=getattr(args, 'amazon_meta_path', None) or None,
        cache_dir=getattr(args, 'amazon_cache_dir', './cache'),
        num_candidates=num_candidates,
        min_interactions=getattr(args, 'amazon_min_inter', 5),
        seed=args.seed,
        force_rebuild=getattr(args, 'amazon_force_rebuild', False),
        auto_download=True,
    )

    tokenizer = load_llama3_tokenizer(args.llm_path)

    def collate(batch):
        return collate_llama3_batch(batch, tokenizer,
                                    max_length=getattr(args, 'second_max_seq_length', 1024))

    train_loader = _build_dataloader(splits['train'], args.second_batch_size,
                                     shuffle=True,  collate_fn=collate)
    val_loader   = _build_dataloader(splits['val'],   args.second_batch_size,
                                     shuffle=False, collate_fn=collate)

    # ── 模型 ──────────────────────────────────────────────────────────────────
    model = LLaMA3Recommender(
        model_path=args.llm_path,
        num_classes=num_candidates,
        load_in_4bit=getattr(args, 'llama3_load_4bit', False),
        load_in_8bit=getattr(args, 'llama3_load_8bit', True),
        soft_prompt_len=getattr(args, 'soft_prompt_len', 100),
        freeze_llm=True,
    )

    # ── 加载第一阶段学到的 soft-prompt ───────────────────────────────────────
    stage1_pt = os.path.join(learned_soft_prompt_path, 'stage1.pt') \
                if learned_soft_prompt_path else None
    if stage1_pt and os.path.exists(stage1_pt):
        state = torch.load(stage1_pt, map_location='cpu')
        if state.get('soft_embeddings') and model.soft_embeddings is not None:
            model.soft_embeddings.load_state_dict(state['soft_embeddings'])
            print(f"[Stage2] ✓ 已加载第一阶段 soft-prompt 权重: {stage1_pt}")
    else:
        print("[Stage2] ⚠ 未找到第一阶段权重，soft-prompt 使用随机初始化")

    # 解冻 LLM 并应用 LoRA
    for param in model.llm.parameters():
        param.requires_grad = True
    if getattr(args, 'second_if_peft', True):
        model.apply_lora(
            r=args.second_lora_r,
            lora_alpha=args.second_lora_alpha,
            lora_dropout=args.second_lora_dropout,
            target_modules=getattr(args, 'llama3_target_modules',
                                   ['q_proj', 'v_proj']),
        )

    # 冻结 soft-prompt（只微调 LLM adapter + 分类头）
    if model.soft_embeddings is not None:
        for param in model.soft_embeddings.parameters():
            param.requires_grad = False

    use_cuda = args.device == 'cuda' and torch.cuda.is_available()
    device   = torch.device('cuda' if use_cuda else 'cpu')
    # 注：LLaMA3Recommender 内部使用 device_map="auto"，分类头需单独移动
    model.classifier = model.classifier.to(device)
    if model.soft_embeddings is not None:
        model.soft_embeddings = model.soft_embeddings.to(device)

    # ── 优化器 ────────────────────────────────────────────────────────────────
    try:
        from bitsandbytes.optim import PagedAdamW8bit
        optimizer = PagedAdamW8bit(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.second_lr, weight_decay=args.second_weight_decay
        )
    except ImportError:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.second_lr, weight_decay=args.second_weight_decay
        )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.second_num_warmup_steps,
        num_training_steps=args.second_num_training_steps,
    )

    loss_func  = nn.CrossEntropyLoss()
    best_hit10 = 0.0
    grad_accum = args.second_gradient_accumulation_steps
    eval_every = args.second_eval_every_steps
    glb_step = actual_step = 0

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    for epoch in tqdm(range(args.second_total_epoch), desc='Epoch'):
        model.train()
        model.classifier.train()
        tot_loss     = 0.0
        epoch_metrics = init_metrics(ks)

        for step, batch in enumerate(train_loader):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)   # (B, num_candidates)
            loss   = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()

            # 累积训练指标
            probs = F.softmax(logits.detach(), dim=-1)
            ranks = (torch.argsort(probs, descending=True)
                     == labels.unsqueeze(-1)).nonzero(as_tuple=False)[:, -1] + 1
            update_metrics(epoch_metrics, ranks, ks)

            actual_step += 1
            if actual_step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                glb_step += 1

            if step % 100 == 1:
                avg = tot_loss / (step + 1)
                print(f"  Epoch {epoch}  step {step}  loss={avg:.4f}")

            # ── 验证 ──────────────────────────────────────────────────────────
            if (actual_step % grad_accum == 0 and glb_step > 0
                    and glb_step % eval_every == 0):
                val_metrics = _evaluate_llama3(model, val_loader, device, ks)
                train_now   = finalize_metrics(epoch_metrics, ks)
                print(f"  [Val]   {metrics_to_str(val_metrics, ks)}")
                print(f"  [Train] {metrics_to_str(train_now,  ks)}")

                primary_k = 10 if 10 in ks else ks[-1]
                if val_metrics[f'hit@{primary_k}'] > best_hit10:
                    best_hit10 = val_metrics[f'hit@{primary_k}']
                    model.save(args.second_model_path)
                    print(f"  ✓ 保存模型  (best hit@{primary_k}={best_hit10:.4f})")

                model.train()
                model.classifier.train()

        ep_final = finalize_metrics(epoch_metrics, ks)
        print(f"\n[Epoch {epoch}] loss={tot_loss/max(len(train_loader),1):.4f}  "
              f"{metrics_to_str(ep_final, ks)}\n")

    return args.second_model_path


def _evaluate_llama3(model, dataloader, device, ks):
    """在 val/test 集上计算 Hit@K / NDCG@K。"""
    model.eval()
    agg = init_metrics(ks)
    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)
            logits = model(input_ids, attention_mask)
            probs  = F.softmax(logits, dim=-1)
            ranks  = (torch.argsort(probs, descending=True)
                      == labels.unsqueeze(-1)).nonzero(as_tuple=False)[:, -1] + 1
            update_metrics(agg, ranks, ks)
    return finalize_metrics(agg, ks)


# ─────────────────────────────────────────────────────────────────────────────
# 原始 openprompt 流程（非 LLaMA3，向后兼容）
# ─────────────────────────────────────────────────────────────────────────────

def training_of_second_stage(args, learned_soft_prompt_path: str = None):
    """
    路由函数：根据 --llm 选择 LLaMA3 流程或原始 openprompt 流程。

    LLaMA3 流程：
        main.py 先调用 training_of_first_stage_llama3() 得到 soft-prompt 路径，
        再传入本函数加载后做 AdaLoRA 微调。
    """
    if getattr(args, 'llm', 't5') == 'llama3':
        return training_of_second_stage_llama3(args, learned_soft_prompt_path)

    # ── 原始 openprompt 流程 ──────────────────────────────────────────────────
    from bitsandbytes.optim import PagedLion8bit
    from openprompt import PromptForClassification
    from openprompt.plms import load_plm
    from peft import AdaLoraConfig, get_peft_model
    try:
        from MTL.MTL import dynamic_loss_weighting
    except ImportError:
        def dynamic_loss_weighting(a, b, *args, **kwargs): return 0.5*a + 0.5*b
    from utils import creat_Verbalizer, calculate_metrics, evaluate
    from distill_pattern_from_conventional_SR_models.temporal_analysis import load_TA_dataset
    from distill_pattern_from_conventional_SR_models.recommendation_pattern_simulating import load_RPS_dataset

    ks = getattr(args, 'eval_ks', EVAL_KS)

    if getattr(args, 'use_amazon', False):
        from llms_based_sr.amazon_lsr_dataset import load_amazon_LSR_dataset
        LSR_train, LSR_test, LSR_val = load_amazon_LSR_dataset(args)
    else:
        from llms_based_sr.llms_based_sequential_recommendation import (
            load_LSR_dataset, load_LSR_prompt)
        LSR_train, LSR_test, LSR_val = load_LSR_dataset(args)

    from llms_based_sr.llms_based_sequential_recommendation import load_LSR_prompt
    LSR_template = load_LSR_prompt(args)
    plm, tokenizer, model_config, WrapperClass = load_plm(args.llm, args.llm_path)

    if args.second_if_peft:
        prompt_model = PromptForClassification(
            plm=plm, template=LSR_template,
            verbalizer=creat_Verbalizer(tokenizer), freeze_plm=True)
        prompt_model.load_state_dict(
            torch.load(learned_soft_prompt_path, map_location=args.device))
        for name, param in prompt_model.named_parameters():
            if 'soft' in name:
                param.requires_grad = False
        adalora_config = AdaLoraConfig(
            peft_type=args.second_peft_type, init_r=args.second_init_r,
            lora_alpha=args.second_lora_alpha, lora_dropout=args.second_lora_dropout,
            target_modules=args.second_target_modules)
        prompt_model = get_peft_model(prompt_model, peft_config=adalora_config)
    else:
        prompt_model = PromptForClassification(
            plm=plm, template=LSR_template,
            verbalizer=creat_Verbalizer(tokenizer), freeze_plm=False)
        prompt_model.load_state_dict(
            torch.load(learned_soft_prompt_path, map_location=args.device))
        for name, param in prompt_model.named_parameters():
            if 'soft' in name:
                param.requires_grad = False

    if args.parallelize:
        prompt_model.parallelize()
    use_cuda = args.device == 'cuda'
    if use_cuda:
        prompt_model = prompt_model.to(torch.device('cuda'))

    loss_func = nn.CrossEntropyLoss()
    no_decay  = ['bias', 'LayerNorm.weight', 'raw_embedding']
    opt_params = [{'params': [p for n, p in prompt_model.template.named_parameters()
                               if not any(nd in n for nd in no_decay) and p.requires_grad]}]
    optimizer  = PagedLion8bit(opt_params, lr=args.second_lr,
                               weight_decay=args.second_weight_decay)
    scheduler  = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.second_num_warmup_steps,
        num_training_steps=args.second_num_training_steps)

    best_hit10 = 0.0
    grad_accum = args.second_gradient_accumulation_steps
    eval_every = args.second_eval_every_steps
    glb_step = actual_step = 0
    prompt_model.train()

    for epoch in tqdm(range(args.second_total_epoch)):
        tot_loss      = 0.0
        epoch_metrics = init_metrics(ks)

        for step, inputs in enumerate(LSR_train):
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['label']
            _, _, _, step_m = calculate_metrics(logits, labels, ks=ks)
            for k in ks:
                epoch_metrics[f'hit@{k}']  += step_m[f'hit@{k}']
                epoch_metrics[f'ndcg@{k}'] += step_m[f'ndcg@{k}']
            epoch_metrics['count'] += step_m['count']

            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            actual_step += 1

            if step % 100 == 1:
                print(f"Epoch {epoch}  loss={tot_loss/(step+1):.4f}")

            if actual_step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 2.0)
                glb_step += 1

            if (actual_step % grad_accum == 0 and glb_step > 0
                    and glb_step % eval_every == 0):
                _, _, _, val_full = evaluate(prompt_model, LSR_val,
                                             use_cuda=use_cuda, ks=ks)
                primary_k = 10 if 10 in ks else ks[-1]
                if val_full[f'hit@{primary_k}'] > best_hit10:
                    best_hit10 = val_full[f'hit@{primary_k}']
                    torch.save(prompt_model.state_dict(), args.second_model_path)
                    print(f"  ✓ 已保存  val: {metrics_to_str(val_full, ks)}")
                prompt_model.train()

        ep = finalize_metrics(epoch_metrics, ks)
        print(f"\n[Epoch {epoch}] {metrics_to_str(ep, ks)}\n")

    return args.second_model_path