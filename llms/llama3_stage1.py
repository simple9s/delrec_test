"""
DELRec/llms/llama3_stage1.py
=============================
LLaMA 3 兼容的第一阶段训练。

原始 DELRec 第一阶段（openprompt / T5）做两件事：
  - TA  (Temporal Analysis)：给定用户历史前半段，预测「最近一个交互 item」
        → 让 soft-prompt 学会感知时序模式
  - RPS (Recommendation Pattern Simulating)：给定用户历史 + SR 模型预测结果，
        预测 SR 模型会推荐哪个 item
        → 让 soft-prompt 学会模拟 SR 模型的推荐行为

两个任务共享同一套 soft_embeddings（即论文中的 soft-prompt），
各自拥有独立的分类头（task head），用动态损失权重（GradNorm / MTL）联合训练。
LLM 权重全程冻结，只有 soft_embeddings + task heads 参与梯度更新。

第一阶段结束后，soft_embeddings 权重被保存，供第二阶段加载继续使用。
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from llms.llama3_wrapper import (
    LLaMA3Recommender,
    load_llama3_tokenizer,
    SYSTEM_PROMPT,
)
from utils import init_metrics, update_metrics, finalize_metrics, metrics_to_str, EVAL_KS


def dynamic_loss_weighting(loss_a: torch.Tensor, loss_b: torch.Tensor,
                            shared_param: torch.Tensor) -> torch.Tensor:
    """
    简化版 GradNorm 动态损失权重：
    根据两个任务对共享参数的梯度范数比例，自动平衡损失权重。
    若梯度计算失败则退化为等权重。
    """
    try:
        g_a = torch.autograd.grad(loss_a, shared_param, retain_graph=True,
                                  create_graph=False)[0]
        g_b = torch.autograd.grad(loss_b, shared_param, retain_graph=True,
                                  create_graph=False)[0]
        norm_a = g_a.norm().item() + 1e-8
        norm_b = g_b.norm().item() + 1e-8
        w_a = norm_b / (norm_a + norm_b)
        w_b = norm_a / (norm_a + norm_b)
        return w_a * loss_a + w_b * loss_b
    except Exception:
        return 0.5 * loss_a + 0.5 * loss_b


# ─────────────────────────────────────────────────────────────────────────────
# Prompt 构建：TA 任务
# ─────────────────────────────────────────────────────────────────────────────

def build_ta_prompt(
    icl_seq: str,          # 用户前期历史（item title 逗号分隔）
    ta_seq: str,           # 用户后期历史（item title 逗号分隔）
    candidates: str,       # 候选集（item title 逗号分隔）
    model_name: str = "SASRec",
) -> str:
    """
    TA 任务 prompt：
    给定「前期历史 + 后期历史」，预测后期历史中最后一个 item（[item X]）。
    对应原始 TA.txt 模板的核心语义。
    """
    user_content = (
        f"Simulate the recommendation pattern of the {model_name} model "
        f"to perform a temporal analysis.\n\n"
        f"Early interaction history: {icl_seq}\n\n"
        f"Recent interaction history: {ta_seq}\n\n"
        f"Candidate set: {candidates}\n\n"
        f"Based on the temporal pattern, the most recent item [item X] is:"
    )
    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prompt 构建：RPS 任务
# ─────────────────────────────────────────────────────────────────────────────

def build_rps_prompt(
    user_seq: str,         # 用户完整历史
    sr_pre: str,           # SR 模型已预测的 item（逗号分隔）
    candidates: str,       # 候选集
    model_name: str = "SASRec",
) -> str:
    """
    RPS 任务 prompt：
    给定用户历史 + SR 模型已推荐的 item，预测 SR 模型还会推荐哪个 item。
    对应原始 RPS.txt 模板的核心语义。
    """
    user_content = (
        f"Simulate the recommendation pattern of the {model_name} model.\n\n"
        f"User interaction history: {user_seq}\n\n"
        f"Items already predicted by {model_name}: {sr_pre}\n\n"
        f"Candidate set: {candidates}\n\n"
        f"The next item predicted by the {model_name} model is:"
    )
    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 从 Amazon LOO splits 构建 TA / RPS 数据
# ─────────────────────────────────────────────────────────────────────────────

def build_stage1_data(splits: dict, model_name: str = "SASRec") -> dict:
    """
    从 leave_one_out_split 的结果构建 TA / RPS 训练数据。

    TA 数据构建策略
    ---------------
    每条 LOO 记录中，用户历史 seq 按时间排序。
    取前半段作为 icl_seq，后半段作为 ta_seq，
    ta_seq 最后一个 item 作为正样本，其余候选作为负样本。

    RPS 数据构建策略
    ----------------
    使用同一条记录，将 seq 中间若干 item 视为 SR 模型已推荐结果（sr_pre），
    正样本为 SR 模型下一个应推荐的 item（即 pos_item）。

    Returns
    -------
    dict with keys 'train' / 'val' / 'test'，
    每条记录包含 task / prompt / label_idx / candidates_len。
    """
    stage1_data: dict = {'train': [], 'val': [], 'test': []}

    for split, records in splits.items():
        for d in records:
            seq       = d['user_seq']          # list[str]，已按时间排序
            cands     = d['candidates']        # list[str]，100 个候选
            label_idx = d['label_idx']
            seq_len   = len(seq)

            # ── TA 记录 ────────────────────────────────────────────────────
            if seq_len >= 4:
                mid       = max(2, seq_len // 2)
                icl_seq   = ', '.join(seq[:mid])
                ta_seq    = ', '.join(seq[mid:])
                cands_str = ', '.join(cands)
                stage1_data[split].append({
                    'task'      : 'TA',
                    'prompt'    : build_ta_prompt(icl_seq, ta_seq, cands_str, model_name),
                    'label_idx' : label_idx,
                    'cands_len' : len(cands),
                })

            # ── RPS 记录 ──────────────────────────────────────────────────
            # 模拟 SR 模型：取序列后 1/3 作为已预测结果
            if seq_len >= 3:
                sr_start  = max(1, seq_len * 2 // 3)
                sr_pre    = ', '.join(seq[sr_start:]) if sr_start < seq_len else seq[-1]
                user_seq  = ', '.join(seq[:sr_start])
                cands_str = ', '.join(cands)
                stage1_data[split].append({
                    'task'      : 'RPS',
                    'prompt'    : build_rps_prompt(user_seq, sr_pre, cands_str, model_name),
                    'label_idx' : label_idx,
                    'cands_len' : len(cands),
                })

    for split, data in stage1_data.items():
        n_ta  = sum(1 for d in data if d['task'] == 'TA')
        n_rps = sum(1 for d in data if d['task'] == 'RPS')
        print(f"[Stage1 Data] {split}: TA={n_ta}  RPS={n_rps}  total={len(data)}")

    return stage1_data


# ─────────────────────────────────────────────────────────────────────────────
# Dataset & Collate
# ─────────────────────────────────────────────────────────────────────────────

class Stage1Dataset(Dataset):
    def __init__(self, data: list):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_stage1_batch(batch: list, tokenizer, max_length: int = 1024) -> dict:
    """按 task 分组，分别返回 TA / RPS 两组张量。"""
    ta_texts,   ta_labels   = [], []
    rps_texts,  rps_labels  = [], []

    for item in batch:
        if item['task'] == 'TA':
            ta_texts.append(item['prompt'])
            ta_labels.append(item['label_idx'])
        else:
            rps_texts.append(item['prompt'])
            rps_labels.append(item['label_idx'])

    def _encode(texts, labels):
        if not texts:
            return None
        enc = tokenizer(texts, padding=True, truncation=True,
                        max_length=max_length, return_tensors='pt')
        return {**enc, 'labels': torch.tensor(labels, dtype=torch.long)}

    return {
        'TA' : _encode(ta_texts,  ta_labels),
        'RPS': _encode(rps_texts, rps_labels),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 第一阶段模型：共享 soft-prompt + 两个独立任务头
# ─────────────────────────────────────────────────────────────────────────────

class LLaMA3Stage1Model(nn.Module):
    """
    第一阶段模型：
      - LLM 权重全程冻结
      - soft_embeddings：唯一被训练的 LLM 相关参数（从 LLaMA3Recommender 共享）
      - ta_head  : TA 任务分类头
      - rps_head : RPS 任务分类头

    两个任务头输出 shape 均为 (B, num_candidates)。
    """

    def __init__(self, backbone: LLaMA3Recommender, num_candidates: int = 100):
        super().__init__()
        H = backbone.hidden_size

        # 共用 backbone（soft_embeddings 在 backbone 内）
        self.backbone = backbone

        # 两个独立任务头（结构与 backbone.classifier 相同）
        self.ta_head = nn.Sequential(
            nn.Linear(H, H // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(H // 2, num_candidates),
        )
        self.rps_head = nn.Sequential(
            nn.Linear(H, H // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(H // 2, num_candidates),
        )

    def _get_hidden(self, input_ids: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
        """提取最后一个有效 token 的隐向量，复用 backbone 的 soft-prompt 拼接逻辑。"""
        B = input_ids.shape[0]
        backbone = self.backbone

        if backbone.soft_embeddings is not None:
            token_embeds  = backbone.llm.get_input_embeddings()(input_ids)
            soft_idx      = torch.arange(backbone.soft_prompt_len, device=input_ids.device)
            soft_emb      = backbone.soft_embeddings(soft_idx).to(token_embeds.device)
            soft_emb      = soft_emb.unsqueeze(0).expand(B, -1, -1)
            inputs_embeds = torch.cat([soft_emb, token_embeds], dim=1)
            soft_mask     = torch.ones(B, backbone.soft_prompt_len,
                                       dtype=attention_mask.dtype,
                                       device=attention_mask.device)
            attention_mask = torch.cat([soft_mask, attention_mask], dim=1)
            outputs = backbone.llm(inputs_embeds=inputs_embeds,
                                   attention_mask=attention_mask,
                                   output_hidden_states=True)
        else:
            outputs = backbone.llm(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   output_hidden_states=True)

        last_hidden = outputs.hidden_states[-1]            # (B, S, H)
        seq_lens    = attention_mask.sum(dim=1) - 1        # (B,)
        return last_hidden[torch.arange(B), seq_lens].float()  # (B, H)

    def forward_ta(self, input_ids, attention_mask) -> torch.Tensor:
        h = self._get_hidden(input_ids, attention_mask)
        return self.ta_head(h)

    def forward_rps(self, input_ids, attention_mask) -> torch.Tensor:
        h = self._get_hidden(input_ids, attention_mask)
        return self.rps_head(h)

    def get_soft_prompt_params(self):
        """返回 soft_embeddings 参数（供优化器使用）。"""
        if self.backbone.soft_embeddings is not None:
            return list(self.backbone.soft_embeddings.parameters())
        return []

    def get_head_params(self):
        """返回两个任务头的参数。"""
        return list(self.ta_head.parameters()) + list(self.rps_head.parameters())

    def save_soft_prompt(self, path: str):
        """只保存 soft_embeddings（供第二阶段加载）。"""
        os.makedirs(path, exist_ok=True)
        state = {
            'soft_embeddings': self.backbone.soft_embeddings.state_dict()
                               if self.backbone.soft_embeddings else None,
            'ta_head' : self.ta_head.state_dict(),
            'rps_head': self.rps_head.state_dict(),
        }
        save_file = os.path.join(path, 'stage1.pt')
        torch.save(state, save_file)
        print(f"[Stage1] soft-prompt 已保存: {save_file}")
        return save_file

    @classmethod
    def load_soft_prompt(cls, model: 'LLaMA3Stage1Model', path: str):
        """从保存路径恢复 soft_embeddings。"""
        save_file = os.path.join(path, 'stage1.pt')
        state = torch.load(save_file, map_location='cpu')
        if state['soft_embeddings'] and model.backbone.soft_embeddings:
            model.backbone.soft_embeddings.load_state_dict(state['soft_embeddings'])
        model.ta_head.load_state_dict(state['ta_head'])
        model.rps_head.load_state_dict(state['rps_head'])
        print(f"[Stage1] soft-prompt 已加载: {save_file}")


# ─────────────────────────────────────────────────────────────────────────────
# 第一阶段训练主函数
# ─────────────────────────────────────────────────────────────────────────────

def training_of_first_stage_llama3(args, splits: dict) -> str:
    """
    LLaMA3 第一阶段：TA + RPS 双任务训练 soft-prompt。

    Parameters
    ----------
    args   : argparse.Namespace，从 main.py 传入
    splits : leave_one_out_split 的输出（train/val/test 字典）

    Returns
    -------
    soft_prompt_path : str  第一阶段权重保存路径
    """
    ks         = getattr(args, 'eval_ks', EVAL_KS)
    model_name = getattr(args, 'SR_model', 'SASRec')

    # ── 构建第一阶段数据 ──────────────────────────────────────────────────────
    stage1_data = build_stage1_data(splits, model_name)

    tokenizer = load_llama3_tokenizer(args.llm_path)
    max_len   = getattr(args, 'first_max_seq_length', 1024)

    def collate(batch):
        return collate_stage1_batch(batch, tokenizer, max_len)

    train_loader = DataLoader(
        Stage1Dataset(stage1_data['train']),
        batch_size=getattr(args, 'first_batch_size', 16),
        shuffle=True, collate_fn=collate,
    )
    val_loader = DataLoader(
        Stage1Dataset(stage1_data['val']),
        batch_size=getattr(args, 'first_batch_size', 16),
        shuffle=False, collate_fn=collate,
    )

    # ── 构建模型 ──────────────────────────────────────────────────────────────
    backbone = LLaMA3Recommender(
        model_path=args.llm_path,
        num_classes=getattr(args, 'num_candidates', 100),
        load_in_4bit=getattr(args, 'llama3_load_4bit', False),
        load_in_8bit=getattr(args, 'llama3_load_8bit', True),
        soft_prompt_len=getattr(args, 'soft_prompt_len', 100),
        freeze_llm=True,   # 第一阶段全程冻结 LLM
    )

    num_candidates = getattr(args, 'num_candidates', 100)
    model = LLaMA3Stage1Model(backbone, num_candidates)

    use_cuda = args.device == 'cuda' and torch.cuda.is_available()
    device   = torch.device('cuda' if use_cuda else 'cpu')
    # 分类头移到 device，LLM 由 device_map="auto" 管理
    model.ta_head  = model.ta_head.to(device)
    model.rps_head = model.rps_head.to(device)
    if backbone.soft_embeddings is not None:
        backbone.soft_embeddings = backbone.soft_embeddings.to(device)

    # ── 优化器：只更新 soft_embeddings + 两个任务头 ───────────────────────────
    trainable_params = model.get_soft_prompt_params() + model.get_head_params()
    print(f"[Stage1] 可训练参数量: "
          f"{sum(p.numel() for p in trainable_params):,}")

    try:
        from bitsandbytes.optim import PagedAdamW8bit
        optimizer = PagedAdamW8bit(trainable_params,
                                   lr=args.first_lr,
                                   weight_decay=args.first_weight_decay)
    except ImportError:
        optimizer = torch.optim.AdamW(trainable_params,
                                      lr=args.first_lr,
                                      weight_decay=args.first_weight_decay)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.first_num_warmup_steps,
        num_training_steps=args.first_num_training_steps,
    )

    loss_func   = nn.CrossEntropyLoss()
    best_hit    = 0.0
    grad_accum  = args.first_gradient_accumulation_steps
    eval_every  = args.first_eval_every_steps
    glb_step = actual_step = 0

    soft_prompt_save_path = getattr(args, 'first_learned_soft_prompt_path',
                                    './stage1_soft_prompt')

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    for epoch in tqdm(range(args.first_total_epoch), desc='[Stage1] Epoch'):
        model.ta_head.train()
        model.rps_head.train()
        if backbone.soft_embeddings is not None:
            backbone.soft_embeddings.train()

        tot_loss      = 0.0
        epoch_metrics = init_metrics(ks)
        n_steps       = 0

        for step, batch in enumerate(train_loader):
            ta_batch  = batch.get('TA')
            rps_batch = batch.get('RPS')

            # 至少需要一个任务有数据
            if ta_batch is None and rps_batch is None:
                continue

            loss_ta  = None
            loss_rps = None

            # ── TA 前向 ──────────────────────────────────────────────────────
            if ta_batch is not None:
                ta_ids    = ta_batch['input_ids'].to(device)
                ta_mask   = ta_batch['attention_mask'].to(device)
                ta_labels = ta_batch['labels'].to(device)
                ta_logits = model.forward_ta(ta_ids, ta_mask)
                loss_ta   = loss_func(ta_logits, ta_labels)
                # 累积指标
                ranks_ta = (torch.argsort(F.softmax(ta_logits.detach(), dim=-1),
                                          descending=True)
                            == ta_labels.unsqueeze(-1)).nonzero(as_tuple=False)[:, -1] + 1
                update_metrics(epoch_metrics, ranks_ta, ks)

            # ── RPS 前向 ─────────────────────────────────────────────────────
            if rps_batch is not None:
                rps_ids    = rps_batch['input_ids'].to(device)
                rps_mask   = rps_batch['attention_mask'].to(device)
                rps_labels = rps_batch['labels'].to(device)
                rps_logits = model.forward_rps(rps_ids, rps_mask)
                loss_rps   = loss_func(rps_logits, rps_labels)
                ranks_rps  = (torch.argsort(F.softmax(rps_logits.detach(), dim=-1),
                                            descending=True)
                              == rps_labels.unsqueeze(-1)).nonzero(as_tuple=False)[:, -1] + 1
                update_metrics(epoch_metrics, ranks_rps, ks)

            # ── 动态损失权重（两任务都有时）────────────────────────────────────
            if loss_ta is not None and loss_rps is not None:
                # 用 soft_embeddings 参数作为共享参数
                shared_params = model.get_soft_prompt_params()
                if shared_params:
                    loss = dynamic_loss_weighting(loss_ta, loss_rps,
                                                  shared_params[0])
                else:
                    loss = 0.5 * loss_ta + 0.5 * loss_rps
            elif loss_ta is not None:
                loss = loss_ta
            else:
                loss = loss_rps

            loss.backward()
            tot_loss  += loss.item()
            n_steps   += 1
            actual_step += 1

            if actual_step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                glb_step += 1

            if step % 100 == 1:
                print(f"  [Stage1 Epoch {epoch}] step={step}  "
                      f"loss={tot_loss/n_steps:.4f}")

            # ── 验证 & 保存 ────────────────────────────────────────────────────
            if (actual_step % grad_accum == 0 and glb_step > 0
                    and glb_step % eval_every == 0):
                val_m = _evaluate_stage1(model, val_loader, device, ks)
                primary_k = 10 if 10 in ks else ks[-1]
                if val_m[f'hit@{primary_k}'] > best_hit:
                    best_hit = val_m[f'hit@{primary_k}']
                    model.save_soft_prompt(soft_prompt_save_path)
                    print(f"  ✓ [Stage1] 保存  val: {metrics_to_str(val_m, ks)}")

                model.ta_head.train()
                model.rps_head.train()
                if backbone.soft_embeddings is not None:
                    backbone.soft_embeddings.train()

        ep = finalize_metrics(epoch_metrics, ks)
        print(f"\n[Stage1 Epoch {epoch}] "
              f"loss={tot_loss/max(n_steps,1):.4f}  "
              f"{metrics_to_str(ep, ks)}\n")

    # 训练结束确保保存一次（防止验证步骤未触发）
    if not os.path.exists(os.path.join(soft_prompt_save_path, 'stage1.pt')):
        model.save_soft_prompt(soft_prompt_save_path)

    return soft_prompt_save_path


# ─────────────────────────────────────────────────────────────────────────────
# 第一阶段验证
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_stage1(model: LLaMA3Stage1Model,
                     dataloader: DataLoader,
                     device: torch.device,
                     ks: list) -> dict:
    """在 val 集上联合评估 TA + RPS 两个任务。"""
    model.ta_head.eval()
    model.rps_head.eval()
    agg = init_metrics(ks)

    with torch.no_grad():
        for batch in dataloader:
            for task_key, forward_fn in [
                ('TA',  model.forward_ta),
                ('RPS', model.forward_rps),
            ]:
                tb = batch.get(task_key)
                if tb is None:
                    continue
                ids    = tb['input_ids'].to(device)
                mask   = tb['attention_mask'].to(device)
                labels = tb['labels'].to(device)
                logits = forward_fn(ids, mask)
                probs  = F.softmax(logits, dim=-1)
                ranks  = (torch.argsort(probs, descending=True)
                          == labels.unsqueeze(-1)).nonzero(as_tuple=False)[:, -1] + 1
                update_metrics(agg, ranks, ks)

    return finalize_metrics(agg, ks)