"""
DELRec/dataload.py
==================
数据加载工具，新增：
- amazon_data_partition : 对接 Amazon 数据集，留一法 + 100 选一
- 保留原有函数，保持向后兼容
"""

import random
from collections import defaultdict
import pickle
import torch
import numpy as np
from openprompt.data_utils.utils import InputExample
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# 原有函数（保持不变）
# ─────────────────────────────────────────────────────────────────────────────

def read_file_portions_based_ratio(fname, ratios=None):
    if ratios is None:
        ratios = [0.8, 0.1, 0.1]

    with open(f'{fname}.txt', 'r') as f:
        lines = f.readlines()

    user_data = defaultdict(list)
    for line in lines:
        u = line.rstrip().split(' ')[0]
        user_data[u].append(line)

    user_ids = list(user_data.keys())
    random.shuffle(user_ids)

    split1 = int(ratios[0] * len(user_ids))
    split2 = split1 + int(ratios[1] * len(user_ids))

    f1_lines = [line for uid in user_ids[:split1]       for line in user_data[uid]]
    f2_lines = [line for uid in user_ids[split1:split2] for line in user_data[uid]]
    f3_lines = [line for uid in user_ids[split2:]       for line in user_data[uid]]

    return f1_lines, f2_lines, f3_lines


def create_sas_dataset(org_dataset, dataset_name):
    user_seq, user_can, user_label, user_saspre = org_dataset
    dataset = []
    for user_id in user_seq.keys():
        dataset.append({
            'user_seq'  : user_seq[user_id],
            'user_can'  : user_can[user_id],
            'user_id'   : user_id,
            'user_label': user_label[user_id],
            'user_saspre': user_saspre[user_id]
        })
    return dataset


def create_RPS_dataset(dataset, dataset_name):
    user_seq, user_can, user_label, model_name, SR_pre = dataset
    dataset = []
    for user_id in user_seq.keys():
        dataset.append({
            'user_seq'  : user_seq[user_id],
            'user_can'  : user_can[user_id],
            'user_id'   : user_id,
            'user_label': user_label[user_id],
            'model_name': model_name[user_id],
            'SR_pre'    : SR_pre[user_id]
        })
    return dataset


def create_LSR_dataset(dataset, dataset_name):
    user_seq, user_can, user_label, model_name = dataset
    dataset = []
    for user_id in user_seq.keys():
        dataset.append({
            'user_seq'  : user_seq[user_id],
            'user_can'  : user_can[user_id],
            'user_id'   : user_id,
            'user_label': user_label[user_id],
            'model_name': model_name[user_id],
        })
    return dataset


def create_TA_dataset(dataset, dataset_name):
    ICL, movie_m, movie_m_1, user_TA, movie_next, user_can, model_name, user_label = dataset
    proc_dataset = []
    for user_id in user_can.keys():
        proc_dataset.append({
            'ICL'       : ICL[user_id],
            'm'         : movie_m[user_id],
            'm_1'       : movie_m_1[user_id],
            'user_TA'   : user_TA[user_id],
            'next'      : movie_next[user_id],
            'user_can'  : user_can[user_id],
            'model_name': model_name[user_id],
            'user_id'   : user_id,
            'user_label': user_label[user_id]
        })
    return proc_dataset


def read_file_portions_based_rows(fname, train_id, test_id, val_id):
    with open(f'{fname}.txt', 'r') as f:
        lines = f.readlines()
    user_data = defaultdict(list)
    for line in lines:
        u = line.rstrip().split(' ')[0]
        user_data[u].append(line)

    user_ids = list(user_data.keys())
    ud1, ud2, ud3 = [], [], []
    for i in train_id: ud1.append(user_ids[i - 1])
    for i in test_id:  ud2.append(user_ids[i - 1])
    for i in val_id:   ud3.append(user_ids[i - 1])

    f1_lines = [line for uid in ud1 for line in user_data[uid]]
    f2_lines = [line for uid in ud2 for line in user_data[uid]]
    f3_lines = [line for uid in ud3 for line in user_data[uid]]

    return f1_lines, f2_lines, f3_lines


def topk(input, k):
    sorted_values_list, sorted_indices_list = [], []
    for i in range(input.shape[0]):
        sv, si = torch.sort(input[i], descending=True)
        sorted_values_list.append(sv[:k])
        sorted_indices_list.append(si[:k])
    return torch.stack(sorted_values_list), torch.stack(sorted_indices_list)


def topk2(input, k):
    sorted_indices = sorted(range(len(input)), key=lambda x: input[x], reverse=True)
    sorted_values  = sorted(input, reverse=True)
    return sorted_values[:k], sorted_indices[:k]


# ─────────────────────────────────────────────────────────────────────────────
# Amazon 留一法数据分区（新增）
# ─────────────────────────────────────────────────────────────────────────────

def amazon_data_partition(interactions: dict,
                          item2title: dict,
                          num_candidates: int = 100,
                          seed: int = 42):
    """
    对 Amazon 数据集进行留一法切分，构建三组数据（train / val / test）。

    Parameters
    ----------
    interactions  : dict[user_id(int), list[item_id(int)]]  已按时间排序
    item2title    : dict[item_id(int), str]
    num_candidates: 候选集大小（正 + 负），默认 100
    seed          : 随机种子

    Returns
    -------
    train_data, val_data, test_data : list of dict
        每条记录：
            user_id, user_seq(list[str]), pos_item(str),
            candidates(list[str]), label_idx(int)
    """
    rng = random.Random(seed)
    all_items    = list(item2title.keys())
    all_item_set = set(all_items)

    train_data, val_data, test_data = [], [], []

    for uid, seq in tqdm(interactions.items(), desc='Amazon 留一法切分'):
        if len(seq) < 3:
            continue

        def _make_record(hist, pos):
            exclude  = set(hist) | {pos}
            neg_pool = list(all_item_set - exclude)
            n_neg    = num_candidates - 1
            if len(neg_pool) < n_neg:
                negs = rng.choices(neg_pool, k=n_neg)
            else:
                negs = rng.sample(neg_pool, n_neg)
            cands = [pos] + negs
            rng.shuffle(cands)
            label = cands.index(pos)
            return {
                'user_id'   : uid,
                'user_seq'  : [item2title.get(i, str(i)) for i in hist],
                'pos_item'  : item2title.get(pos, str(pos)),
                'candidates': [item2title.get(i, str(i)) for i in cands],
                'label_idx' : label,
            }

        # test：最后一个
        test_data.append(_make_record(seq[:-1], seq[-1]))
        # val：倒数第二个
        val_data.append(_make_record(seq[:-2], seq[-2]))
        # train：倒数第三个（序列够长才加入）
        if len(seq) >= 4:
            train_data.append(_make_record(seq[:-3], seq[-3]))

    print(f"[amazon_data_partition] "
          f"train={len(train_data)}  val={len(val_data)}  test={len(test_data)}")
    return train_data, val_data, test_data


def amazon_to_input_examples(data_list: list, model_name: str = 'SASRec'):
    """
    将 amazon_data_partition 的输出转为 openprompt InputExample 列表。

    Parameters
    ----------
    data_list  : list of dict（来自 amazon_data_partition）
    model_name : SR 模型名称（写入 meta）

    Returns
    -------
    examples : list[InputExample]
    """
    examples = []
    for d in data_list:
        user_seq = ', '.join(d['user_seq'])
        user_can = ', '.join(d['candidates'])
        ex = InputExample(
            label=d['label_idx'],
            guid=d['user_id'],
            meta={
                'user_seq'  : user_seq,
                'user_can'  : user_can,
                'model_name': model_name,
            }
        )
        examples.append(ex)
    return examples
