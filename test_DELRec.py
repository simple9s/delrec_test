"""
DELRec/test_DELRec.py
=====================
测试入口：支持 LLaMA 3-3B-Instruct 和原有 openprompt 流程。
输出 Hit@K / NDCG@K，K ∈ {1, 5, 10, 20}，留一法 + 100 选一。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm import tqdm

from DELRec.utils import (
    init_metrics, update_metrics, finalize_metrics, metrics_to_str, EVAL_KS,
)


# ─────────────────────────────────────────────────────────────────────────────
# 统一测试入口（路由）
# ─────────────────────────────────────────────────────────────────────────────

def test(args):
    if getattr(args, 'llm', 't5') == 'llama3':
        return _test_llama3(args)
    return _test_openprompt(args)


# ─────────────────────────────────────────────────────────────────────────────
# LLaMA 3 测试
# ─────────────────────────────────────────────────────────────────────────────

def _test_llama3(args):
    from DELRec.llms.llama3_wrapper import (
        LLaMA3Recommender, load_llama3_tokenizer, collate_llama3_batch
    )
    from DELRec.data.amazon_loader import load_amazon_dataset
    from torch.utils.data import DataLoader, Dataset

    ks             = getattr(args, 'eval_ks', EVAL_KS)
    num_candidates = getattr(args, 'num_candidates', 100)

    # 数据
    splits, item2title = load_amazon_dataset(
        dataset_version=args.amazon_version,
        category=args.amazon_category,
        review_path=getattr(args, 'amazon_review_path', None) or None,
        meta_path=getattr(args, 'amazon_meta_path', None) or None,
        cache_dir=getattr(args, 'amazon_cache_dir', './cache'),
        num_candidates=num_candidates,
        min_interactions=getattr(args, 'amazon_min_inter', 5),
        seed=args.seed,
        force_rebuild=False,
        auto_download=True,
    )

    tokenizer = load_llama3_tokenizer(args.llm_path)

    class _DS(Dataset):
        def __init__(self, d): self.d = d
        def __len__(self): return len(self.d)
        def __getitem__(self, i): return self.d[i]

    def collate(batch):
        return collate_llama3_batch(batch, tokenizer,
                                    max_length=getattr(args, 'second_max_seq_length', 1024))

    test_loader = DataLoader(
        _DS(splits['test']),
        batch_size=getattr(args, 'second_batch_size', 16),
        shuffle=False, collate_fn=collate
    )

    # 模型
    model = LLaMA3Recommender(
        model_path=args.llm_path,
        num_classes=num_candidates,
        load_in_4bit=getattr(args, 'llama3_load_4bit', False),
        load_in_8bit=getattr(args, 'llama3_load_8bit', False),
        soft_prompt_len=getattr(args, 'soft_prompt_len', 100),
        freeze_llm=True,
    )
    model.load(args.second_model_path)

    use_cuda = args.device == 'cuda' and torch.cuda.is_available()
    device   = torch.device('cuda' if use_cuda else 'cpu')
    model.classifier = model.classifier.to(device)
    model.eval()

    return _run_eval(model, test_loader, device, ks, mode='llama3')


# ─────────────────────────────────────────────────────────────────────────────
# 原始 openprompt 测试（向后兼容）
# ─────────────────────────────────────────────────────────────────────────────

def _test_openprompt(args):
    import torch
    from openprompt import PromptForClassification
    from openprompt.plms import load_plm
    from openprompt.prompts import ManualVerbalizer
    from peft import AdaLoraConfig, get_peft_model
    from DELRec.llms_based_sr.llms_based_sequential_recommendation import (
        load_LSR_prompt, load_LSR_dataset
    )

    ks = getattr(args, 'eval_ks', EVAL_KS)

    if getattr(args, 'use_amazon', False):
        from DELRec.llms_based_sr.amazon_lsr_dataset import load_amazon_LSR_dataset
        _, test_loader, _ = load_amazon_LSR_dataset(args)
    else:
        _, test_loader, _ = load_LSR_dataset(args)

    plm, tokenizer, _, WrapperClass = load_plm(args.llm, args.llm_path)
    template = load_LSR_prompt(args)

    def _verbalizer():
        with open('../title_set.txt', 'r') as f:
            cla = [l.strip() for l in f]
        lw = {item: [item, item[:-7]] for item in cla}
        return ManualVerbalizer(tokenizer=tokenizer, classes=cla, label_words=lw)

    if args.second_if_peft:
        prompt_model = PromptForClassification(
            plm=plm, template=template,
            verbalizer=_verbalizer(), freeze_plm=True)
        adalora_config = AdaLoraConfig(
            peft_type=args.second_peft_type, init_r=args.second_init_r,
            lora_alpha=args.second_lora_alpha, lora_dropout=args.second_lora_dropout,
            target_modules=args.second_target_modules)
        prompt_model = get_peft_model(prompt_model, adalora_config)
        prompt_model.load_state_dict(
            torch.load(args.second_model_path, map_location=args.device))
    else:
        prompt_model = PromptForClassification(
            plm=plm, template=template,
            verbalizer=_verbalizer(), freeze_plm=True)
        prompt_model.load_state_dict(
            torch.load(args.second_model_path, map_location=args.device))

    for name, param in prompt_model.named_parameters():
        if 'soft' in name:
            param.requires_grad = False

    if args.parallelize:
        prompt_model.parallelize()

    use_cuda = args.device == 'cuda'
    device   = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        prompt_model = prompt_model.to(device)

    return _run_eval(prompt_model, test_loader, device, ks, mode='openprompt')


# ─────────────────────────────────────────────────────────────────────────────
# 通用评估循环
# ─────────────────────────────────────────────────────────────────────────────

def _run_eval(model, dataloader, device, ks, mode: str = 'llama3'):
    model.eval()
    global_metrics = init_metrics(ks)

    print("\n══════════ 开始测试 ══════════")
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), desc='Test',
                                total=len(dataloader)):
            if mode == 'llama3':
                input_ids      = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels         = batch['labels'].to(device)
                logits         = model(input_ids, attention_mask)
            else:
                # openprompt 模式
                if str(device) != 'cpu':
                    batch = batch.cuda()
                logits = model(batch)
                labels = batch['label']

            probs = F.softmax(logits, dim=-1)
            ranks = (torch.argsort(probs, descending=True)
                     == labels.unsqueeze(-1)).nonzero(as_tuple=False)[:, -1] + 1
            update_metrics(global_metrics, ranks, ks)

            if (step + 1) % 50 == 0:
                tmp = finalize_metrics(global_metrics, ks)
                print(f"  [Step {step+1:5d}] {metrics_to_str(tmp, ks)}")

    # ── 最终结果 ───────────────────────────────────────────────────────────────
    final = finalize_metrics(global_metrics, ks)
    print("\n══════════ 测试结果 ══════════")
    print(f"  样本总数: {global_metrics['count']}")
    for k in ks:
        print(f"  Hit@{k:<3d} = {final[f'hit@{k}']:.4f}    "
              f"NDCG@{k:<3d} = {final[f'ndcg@{k}']:.4f}")
    print("══════════════════════════════\n")
    return final
