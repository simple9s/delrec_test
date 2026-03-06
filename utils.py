import re
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np

# ── 评估 K 值 ────────────────────────────────────────────────────────────────
EVAL_KS = [1, 5, 10, 20]

# openprompt 相关函数仅在 T5 流程中使用，懒加载避免版本冲突
def creat_Verbalizer(tokenizer):
    from openprompt.prompts import ManualVerbalizer
    with open('../title_set.txt', 'r') as f:
        lines = f.readlines()
    cla = [line.strip() for line in lines]
    saspre_label = {item: [item, item[:-7]] for item in cla}
    return ManualVerbalizer(tokenizer=tokenizer, classes=cla, label_words=saspre_label)


def create_prompt(scriptsbase, plm, tokenizer, prompt_id):
    from openprompt.prompts import MixedTemplate
    mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer).from_file(
        f'data/prompts/{scriptsbase}.txt', choice=prompt_id)
    return mytemplate


# ── 指标工具 ──────────────────────────────────────────────────────────────────

def init_metrics(ks=None):
    """初始化指标字典，key 为 hit@k / ndcg@k。"""
    ks = ks or EVAL_KS
    d = {}
    for k in ks:
        d[f'hit@{k}'] = 0.0
        d[f'ndcg@{k}'] = 0.0
    d['count'] = 0
    return d


def update_metrics(metrics: dict, ranks_tensor, ks=None):
    """
    给定每个样本的排名（1-based），累加 Hit@K / NDCG@K。

    Parameters
    ----------
    metrics  : 由 init_metrics() 创建的字典，原地更新
    ranks_tensor : 1-D LongTensor，每个元素为对应样本 ground-truth 的排名（1-based）
    ks       : 评估的 K 列表，默认 EVAL_KS
    """
    ks = ks or EVAL_KS
    for rank in ranks_tensor:
        r = int(rank.cpu().item())
        metrics['count'] += 1
        for k in ks:
            if r <= k:
                metrics[f'hit@{k}'] += 1.0
                metrics[f'ndcg@{k}'] += 1.0 / np.log2(r + 1)


def finalize_metrics(metrics: dict, ks=None):
    """将累加值除以样本数，返回平均指标字典（不含 count）。"""
    ks = ks or EVAL_KS
    n = metrics['count'] if metrics['count'] > 0 else 1
    result = {}
    for k in ks:
        result[f'hit@{k}'] = metrics[f'hit@{k}'] / n
        result[f'ndcg@{k}'] = metrics[f'ndcg@{k}'] / n
    return result


def metrics_to_str(metrics: dict, ks=None):
    """将指标字典格式化为可打印字符串。"""
    ks = ks or EVAL_KS
    parts = []
    for k in ks:
        parts.append(f"Hit@{k}={metrics.get(f'hit@{k}', 0):.4f}")
        parts.append(f"NDCG@{k}={metrics.get(f'ndcg@{k}', 0):.4f}")
    return "  ".join(parts)


# ── 兼容旧接口（供原 train.py / test_DELRec.py 直接使用） ─────────────────────

def calculate_metrics(logits, labels, nd5=0.0, ht5=0.0, ht1=0.0, ks=None):
    """
    兼容旧调用方式，同时返回扩展指标字典。

    Returns
    -------
    nd5, ht5, ht1 : float  （保持旧接口）
    metrics       : dict   Hit@K / NDCG@K for K in EVAL_KS
    """
    ks = ks or EVAL_KS
    probabilities = F.softmax(logits, dim=-1)
    sorted_indices = torch.argsort(probabilities, descending=True)
    # (N,) — 每个样本 ground-truth 的排名（1-based）
    ranks = (sorted_indices == labels.unsqueeze(-1)).nonzero(as_tuple=False)[:, -1] + 1

    metrics = init_metrics(ks)
    update_metrics(metrics, ranks, ks)

    # 维持旧的标量返回，方便原 train 代码不改动
    for r in ranks:
        rv = int(r.cpu().item())
        if rv <= 5:
            nd5 += 1.0 / np.log2(rv + 1)
            ht5 += 1.0
        if rv <= 1:
            ht1 += 1.0

    return nd5, ht5, ht1, metrics


def evaluate(prompt_model, dataloader, nd5=0.0, ht5=0.0, ht1=0.0,
             use_cuda=True, ks=None):
    """
    兼容旧接口，同时返回扩展指标字典。

    Returns
    -------
    nd5, ht5, ht1 : float
    full_metrics  : dict  Hit@K / NDCG@K (averaged)
    """
    ks = ks or EVAL_KS
    prompt_model.eval()
    agg = init_metrics(ks)

    with torch.no_grad():
        for step, inputs in enumerate(dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['label']
            probabilities = F.softmax(logits, dim=-1)
            sorted_indices = torch.argsort(probabilities, descending=True)
            ranks = (sorted_indices == labels.unsqueeze(-1)).nonzero(as_tuple=False)[:, -1] + 1

            update_metrics(agg, ranks, ks)

            for r in ranks:
                rv = int(r.cpu().item())
                if rv <= 5:
                    nd5 += 1.0 / np.log2(rv + 1)
                    ht5 += 1.0
                if rv <= 1:
                    ht1 += 1.0

    full_metrics = finalize_metrics(agg, ks)
    return nd5, ht5, ht1, full_metrics