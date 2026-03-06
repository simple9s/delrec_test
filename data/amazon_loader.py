"""
DELRec/data/amazon_loader.py
============================
Amazon 2018 / 2023 数据集加载器，集成自动下载脚本。
- 自动调用 download.py 完成下载 & 解压
- 留一法（Leave-One-Out）切分
- 100 选一候选构建
- 本地 pickle 缓存，避免重复处理
"""

import json
import random
import gzip
import os
import pickle
from collections import defaultdict
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from data.download import ensure_amazon_dataset


# ─────────────────────────────────────────────────────────────────────────────
# 内部工具
# ─────────────────────────────────────────────────────────────────────────────

def _open_file(path: str):
    return gzip.open(path, 'rt', encoding='utf-8') if path.endswith('.gz') \
        else open(path, 'r', encoding='utf-8')


def _read_jsonl(path: str) -> List[dict]:
    records = []
    with _open_file(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Amazon 2018 解析
# ─────────────────────────────────────────────────────────────────────────────

def load_amazon_2018(
    review_path: str,
    meta_path: Optional[str] = None,
    min_interactions: int = 5,
) -> Tuple[dict, dict, dict, dict]:
    print(f"[Amazon-2018] 读取评论: {review_path}")
    records = _read_jsonl(review_path)

    raw_user_set = sorted({r['reviewerID'] for r in records})
    raw_item_set = sorted({r['asin'] for r in records})
    user2id = {u: i + 1 for i, u in enumerate(raw_user_set)}
    item2id = {a: i + 1 for i, a in enumerate(raw_item_set)}
    id2user = {v: k for k, v in user2id.items()}
    id2item = {v: k for k, v in item2id.items()}

    raw_inter: Dict[int, List] = defaultdict(list)
    for r in records:
        uid = user2id[r['reviewerID']]
        iid = item2id[r['asin']]
        ts  = r.get('unixReviewTime', 0)
        raw_inter[uid].append((iid, ts))

    interactions: Dict[int, List[int]] = {}
    for uid, seq in raw_inter.items():
        sorted_seq = [iid for iid, _ in sorted(seq, key=lambda x: x[1])]
        if len(sorted_seq) >= min_interactions:
            interactions[uid] = sorted_seq

    item2title = {iid: id2item[iid] for iid in id2item}
    if meta_path and os.path.exists(meta_path):
        print(f"[Amazon-2018] 读取元数据: {meta_path}")
        for m in _read_jsonl(meta_path):
            asin  = m.get('asin', '')
            title = (m.get('title', '') or
                     (m.get('description', [''])[0] if m.get('description') else '') or
                     asin)
            if asin in item2id:
                item2title[item2id[asin]] = str(title).strip()

    print(f"[Amazon-2018] 用户数={len(interactions)}  商品数={len(item2title)}")
    return interactions, item2title, id2item, id2user


# ─────────────────────────────────────────────────────────────────────────────
# Amazon 2023 解析
# ─────────────────────────────────────────────────────────────────────────────

def load_amazon_2023(
    review_path: str,
    meta_path: Optional[str] = None,
    min_interactions: int = 5,
) -> Tuple[dict, dict, dict, dict]:
    print(f"[Amazon-2023] 读取评论: {review_path}")

    if review_path.endswith('.parquet'):
        df = pd.read_parquet(review_path)
        col_map = {}
        if 'user_id'     in df.columns: col_map['user_id']     = 'reviewerID'
        if 'parent_asin' in df.columns: col_map['parent_asin'] = 'asin'
        if 'timestamp'   in df.columns: col_map['timestamp']   = 'unixReviewTime'
        df = df.rename(columns=col_map)
        records = df.to_dict('records')
    else:
        records = _read_jsonl(review_path)
        for r in records:
            if 'user_id'     in r and 'reviewerID'     not in r: r['reviewerID']     = r['user_id']
            if 'parent_asin' in r and 'asin'           not in r: r['asin']           = r['parent_asin']
            if 'timestamp'   in r and 'unixReviewTime' not in r: r['unixReviewTime'] = r['timestamp']

    raw_user_set = sorted({r['reviewerID'] for r in records})
    raw_item_set = sorted({r['asin'] for r in records})
    user2id = {u: i + 1 for i, u in enumerate(raw_user_set)}
    item2id = {a: i + 1 for i, a in enumerate(raw_item_set)}
    id2user = {v: k for k, v in user2id.items()}
    id2item = {v: k for k, v in item2id.items()}

    raw_inter: Dict[int, List] = defaultdict(list)
    for r in records:
        uid = user2id[r['reviewerID']]
        iid = item2id[r['asin']]
        ts  = r.get('unixReviewTime', 0)
        raw_inter[uid].append((iid, ts))

    interactions: Dict[int, List[int]] = {}
    for uid, seq in raw_inter.items():
        sorted_seq = [iid for iid, _ in sorted(seq, key=lambda x: x[1])]
        if len(sorted_seq) >= min_interactions:
            interactions[uid] = sorted_seq

    item2title = {iid: id2item[iid] for iid in id2item}
    if meta_path and os.path.exists(meta_path):
        print(f"[Amazon-2023] 读取元数据: {meta_path}")
        if meta_path.endswith('.parquet'):
            mdf = pd.read_parquet(meta_path)
            for _, row in mdf.iterrows():
                asin  = str(row.get('parent_asin', row.get('asin', '')))
                title = str(row.get('title', '') or asin).strip()
                if asin in item2id:
                    item2title[item2id[asin]] = title
        else:
            for m in _read_jsonl(meta_path):
                asin  = m.get('parent_asin', m.get('asin', ''))
                title = (m.get('title', '') or asin).strip()
                if asin in item2id:
                    item2title[item2id[asin]] = str(title)

    print(f"[Amazon-2023] 用户数={len(interactions)}  商品数={len(item2title)}")
    return interactions, item2title, id2item, id2user


# ─────────────────────────────────────────────────────────────────────────────
# 留一法切分 + 100 选一
# ─────────────────────────────────────────────────────────────────────────────

def _sample_negatives(
    pos_item: int, hist_items: List[int],
    all_item_set: set, n: int, rng: random.Random
) -> List[int]:
    exclude = set(hist_items) | {pos_item}
    pool    = list(all_item_set - exclude)
    return rng.choices(pool, k=n) if len(pool) < n else rng.sample(pool, n)


def leave_one_out_split(
    interactions: Dict[int, List[int]],
    item2title: Dict[int, str],
    num_candidates: int = 100,
    seed: int = 42,
) -> dict:
    """
    留一法切分，每条记录格式：
        user_id, user_seq(list[str]), pos_item(str),
        candidates(list[str]), label_idx(int)
    """
    rng = random.Random(seed)
    all_item_set = set(item2title.keys())
    splits: Dict[str, List[dict]] = {'train': [], 'val': [], 'test': []}

    def _make(uid: int, hist: List[int], pos: int) -> dict:
        negs  = _sample_negatives(pos, hist, all_item_set, num_candidates - 1, rng)
        cands = [pos] + negs
        rng.shuffle(cands)
        return {
            'user_id'   : uid,
            'user_seq'  : [item2title.get(i, str(i)) for i in hist],
            'pos_item'  : item2title.get(pos, str(pos)),
            'candidates': [item2title.get(i, str(i)) for i in cands],
            'label_idx' : cands.index(pos),
        }

    for uid, seq in tqdm(interactions.items(), desc='[LOO] 构建切分'):
        if len(seq) < 3:
            continue
        splits['test'].append(_make(uid, seq[:-1], seq[-1]))
        splits['val'].append( _make(uid, seq[:-2], seq[-2]))
        if len(seq) >= 4:
            splits['train'].append(_make(uid, seq[:-3], seq[-3]))

    print(f"[LOO] train={len(splits['train'])}  "
          f"val={len(splits['val'])}  test={len(splits['test'])}")
    return splits


# ─────────────────────────────────────────────────────────────────────────────
# 统一入口（自动下载 + 缓存）
# ─────────────────────────────────────────────────────────────────────────────

def load_amazon_dataset(
    dataset_version: str,
    category: str,
    review_path: Optional[str] = None,
    meta_path: Optional[str] = None,
    cache_dir: str = './cache',
    num_candidates: int = 100,
    min_interactions: int = 5,
    seed: int = 42,
    force_rebuild: bool = False,
    auto_download: bool = True,
) -> Tuple[dict, Dict[int, str]]:
    """
    统一加载入口：自动下载 → 解析 → 留一法切分 → 缓存。

    Parameters
    ----------
    dataset_version : '2018' 或 '2023'
    category        : 数据集类别名称，例如 'Movies_and_TV'
    review_path     : 手动指定评论文件路径（None = 自动下载）
    meta_path       : 手动指定元数据路径（None = 自动下载）
    auto_download   : 路径未指定时自动调用 download.py
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(
        cache_dir,
        f"amazon{dataset_version}_{category}_"
        f"cands{num_candidates}_min{min_interactions}_seed{seed}.pkl"
    )

    # ── 1. 有处理好的缓存，直接返回 ─────────────────────────────────────────
    if not force_rebuild and os.path.exists(cache_file):
        print(f"[Cache] 命中缓存，直接加载: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    # ── 2. 确定原始文件路径（手动指定 > 自动下载默认路径）──────────────────
    if dataset_version == '2018':
        default_review = f"data/amazon_2018/raw/{category}.json"
        default_meta   = f"data/amazon_2018/raw/meta_{category}.json"
    else:
        default_review = f"data/amazon_2023/raw/{category}.jsonl"
        default_meta   = f"data/amazon_2023/raw/meta_{category}.jsonl"

    review_path = review_path or default_review
    meta_path   = meta_path   or default_meta

    # ── 3. 原始文件不存在才触发下载 ──────────────────────────────────────────
    review_missing = not os.path.exists(review_path)
    meta_missing   = not os.path.exists(meta_path)

    if (review_missing or meta_missing) and auto_download:
        if review_missing:
            print(f"[Download] 评论文件不存在，开始下载: {review_path}")
        if meta_missing:
            print(f"[Download] 元数据文件不存在，开始下载: {meta_path}")
        dl_review, dl_meta = ensure_amazon_dataset(int(dataset_version), category)
        if review_missing: review_path = dl_review
        if meta_missing:   meta_path   = dl_meta
    elif review_missing or meta_missing:
        missing = []
        if review_missing: missing.append(review_path)
        if meta_missing:   missing.append(meta_path)
        raise FileNotFoundError(
            f"以下文件不存在，请手动下载或设置 auto_download=True：\n" +
            "\n".join(f"  {p}" for p in missing)
        )
    else:
        print(f"[√] 原始文件已存在，跳过下载")

    # 解析
    if dataset_version == '2018':
        interactions, item2title, _, _ = load_amazon_2018(
            review_path, meta_path, min_interactions)
    elif dataset_version == '2023':
        interactions, item2title, _, _ = load_amazon_2023(
            review_path, meta_path, min_interactions)
    else:
        raise ValueError(f"不支持的版本: {dataset_version}")

    splits = leave_one_out_split(interactions, item2title, num_candidates, seed)
    result = (splits, item2title)

    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    print(f"[Cache] 已缓存到: {cache_file}")
    return result
