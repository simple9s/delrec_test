"""
DELRec/main.py
==============
主入口。新增：
- LLaMA 3-3B-Instruct 支持（--llm llama3）
- Amazon 2018 / 2023 数据集，自动下载（--use_amazon / --amazon_version / --amazon_category）
- 评估指标 Hit@K / NDCG@K，K ∈ {1,5,10,20}，留一法 + 100 选一
"""

import sys
import os
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger

from DELRec.distill_pattern_from_conventional_SR_models.train import training_of_first_stage
from DELRec.llms_based_sr.train import training_of_second_stage
from test_DELRec import test


def main(args):
    pl.seed_everything(args.seed)
    logger = TensorBoardLogger(save_dir='./log/', name=args.log_dir)
    args.logger = logger

    save_dir = os.path.dirname(args.second_model_path) or '.'
    os.makedirs(save_dir, exist_ok=True)

    if args.mode == 'train':
        if args.llm == 'llama3':
            # ── LLaMA3 完整两阶段 ─────────────────────────────────────────────
            # 第一阶段：冻结 LLM，只训练 soft-prompt（TA + RPS 双任务）
            from DELRec.llms.llama3_stage1 import training_of_first_stage_llama3
            from DELRec.data.amazon_loader import load_amazon_dataset
            splits, _ = load_amazon_dataset(
                dataset_version=args.amazon_version,
                category=args.amazon_category,
                review_path=getattr(args, 'amazon_review_path', None) or None,
                meta_path=getattr(args, 'amazon_meta_path', None) or None,
                cache_dir=getattr(args, 'amazon_cache_dir', './cache'),
                num_candidates=getattr(args, 'num_candidates', 100),
                min_interactions=getattr(args, 'amazon_min_inter', 5),
                seed=args.seed,
                force_rebuild=getattr(args, 'amazon_force_rebuild', False),
                auto_download=True,
            )
            soft_prompt_path = training_of_first_stage_llama3(args, splits)
            # 第二阶段：加载 soft-prompt，解冻 LLM，AdaLoRA 微调
            training_of_second_stage(args, soft_prompt_path)
        else:
            # ── 原始 T5 / openprompt 两阶段 ──────────────────────────────────
            learned_soft_prompt = training_of_first_stage(args)
            training_of_second_stage(args, learned_soft_prompt)
    else:
        test(args)

    sys.exit()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = ArgumentParser(description='DELRec: Distilling Sequential Pattern for LLMs SR')

    # ── 基础 ──────────────────────────────────────────────────────────────────
    parser.add_argument('--device',       default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--parallelize',  default=True,  type=bool)
    parser.add_argument('--llm_path',     default='./Llama-3.2-3B-Instruct', type=str,
                        help='LLM 本地路径或 HuggingFace Hub ID')
    parser.add_argument('--llm',          default='llama3',
                        choices=['t5', 'roberta', 'bert', 'albert',
                                 'gpt', 'gpt2', 'opt', 'llama', 'llama3'],
                        help='llama3 = meta-llama/Llama-3.2-3B-Instruct')
    parser.add_argument('--SR_model',     default='SASRec', choices=['SASRec', 'Caser', 'GRU'])
    parser.add_argument('--mode',         default='train', choices=['train', 'test'])
    parser.add_argument('--seed',         default=1234, type=int)
    parser.add_argument('--log_dir',      default='record_logs', type=str)

    # ── LLaMA 3 专属 ─────────────────────────────────────────────────────────
    parser.add_argument('--llama3_load_4bit',       default=False, type=bool,
                        help='4-bit 量化（极度显存不足时才用，精度损失较大）')
    parser.add_argument('--llama3_load_8bit',       default=True,  type=bool,
                        help='8-bit 量化（默认开启，平衡精度与显存）')
    parser.add_argument('--llama3_target_modules',  default=['q_proj', 'v_proj'],
                        nargs='+', help='AdaLoRA 目标模块')
    parser.add_argument('--soft_prompt_len',        default=100, type=int,
                        help='Soft-prompt token 数量')

    # ── 评估设置 ──────────────────────────────────────────────────────────────
    parser.add_argument('--eval_ks',        default=[1, 5, 10, 20], type=int, nargs='+',
                        help='Hit@K / NDCG@K 的 K 列表')
    parser.add_argument('--num_candidates', default=100, type=int,
                        help='留一法候选集大小（1 正 + N-1 负）')

    # ── Amazon 数据集 ─────────────────────────────────────────────────────────
    parser.add_argument('--use_amazon',           default=True,  type=bool,
                        help='使用 Amazon 数据集')
    parser.add_argument('--amazon_version',       default='2018', choices=['2018', '2023'],
                        help='Amazon 数据集版本')
    parser.add_argument('--amazon_category',      default='Movies_and_TV', type=str,
                        help='Amazon 类别名称，例如 Movies_and_TV / Sports_and_Outdoors')
    parser.add_argument('--amazon_review_path',   default='', type=str,
                        help='手动指定评论文件路径（空 = 自动下载）')
    parser.add_argument('--amazon_meta_path',     default='', type=str,
                        help='手动指定元数据路径（空 = 自动下载）')
    parser.add_argument('--amazon_cache_dir',     default='./cache', type=str)
    parser.add_argument('--amazon_min_inter',     default=5, type=int,
                        help='用户最少交互次数')
    parser.add_argument('--amazon_force_rebuild', default=False, type=bool,
                        help='强制重新构建缓存')

    # ── ICL ───────────────────────────────────────────────────────────────────
    parser.add_argument('--ICL_length',     default=4, choices=[4, 6],  type=int)
    parser.add_argument('--ICL_back',       default=3, choices=[3, 5],  type=int)
    parser.add_argument('--candidate_size', default=100, type=int)
    parser.add_argument('--load_soft_prompt_log', default=False, type=bool)

    # ── 第一阶段（T5 / openprompt 流程，LLaMA3 跳过）─────────────────────────
    parser.add_argument('--first_shuffle',              default=False, type=bool)
    parser.add_argument('--first_teacher_forcing',      default=False, type=bool)
    parser.add_argument('--first_predict_eos_token',    default=False, type=bool)
    parser.add_argument('--first_batch_size',           default=20,   type=int)
    parser.add_argument('--first_decoder_max_length',   default=20,   type=int)
    parser.add_argument('--first_truncate_method',      default='tail', choices=['head', 'tail'])
    parser.add_argument('--first_max_seq_length',       default=1065, type=int)
    parser.add_argument('--first_learned_soft_prompt_path', default='./learned_soft_prompt.ckpt')
    parser.add_argument('--first_total_epoch',          default=1000, type=int)
    parser.add_argument('--first_lr',                   default=5e-3, type=float)
    parser.add_argument('--first_weight_decay',         default=1e-5, type=float)
    parser.add_argument('--first_num_warmup_steps',     default=400,  type=int)
    parser.add_argument('--first_num_training_steps',   default=800,  type=int)
    parser.add_argument('--first_gradient_accumulation_steps', default=4, type=int)
    parser.add_argument('--first_eval_every_steps',     default=2,    type=int)

    # ── TA 参数 ───────────────────────────────────────────────────────────────
    parser.add_argument('--TA_prompt_id',      default=0, choices=[0,1,2], type=int)
    parser.add_argument('--TA_load',           default=False, type=bool)
    parser.add_argument('--TA_load_log',       default=False, type=bool)
    parser.add_argument('--TA_all_item_titles_path', default='./title_set.csv')
    parser.add_argument('--TA_log_train_dataset_path',      default='TA_dataset1.pkl')
    parser.add_argument('--TA_log_test_dataset_path',       default='TA_dataset2.pkl')
    parser.add_argument('--TA_log_validation_dataset_path', default='TA_dataset3.pkl')
    parser.add_argument('--TA_seq_with_recommended_size_h', default=-10, type=int)
    parser.add_argument('--TA_truncation_seq', default=14, type=int)

    _sr_paths = {
        'SASRec': './user_interactions_with_text_title_and_predicted_items_by_SASRec',
        'Caser' : './user_interactions_with_text_title_and_predicted_items_by_Caser',
        'GRU'   : './user_interactions_with_text_title_and_predicted_items_by_GRU',
    }
    _sr_default = parser.parse_known_args()[0].SR_model
    _path_default = _sr_paths.get(_sr_default, _sr_paths['SASRec'])
    parser.add_argument('--TA_user_interactions_with_text_title_predicted_by_SR_path',
                        default=_path_default)

    # ── RPS 参数 ──────────────────────────────────────────────────────────────
    parser.add_argument('--RPS_prompt_id',     default=0, choices=[0,1,2], type=int)
    parser.add_argument('--RPS_load',          default=False, type=bool)
    parser.add_argument('--RPS_load_log',      default=False, type=bool)
    parser.add_argument('--RPS_SR_pre_forw',   default=-9, type=int)
    parser.add_argument('--RPS_all_item_titles_path', default='./title_set.csv')
    parser.add_argument('--RPS_log_train_dataset_path',      default='RPS_dataset1.pkl')
    parser.add_argument('--RPS_log_test_dataset_path',       default='RPS_dataset2.pkl')
    parser.add_argument('--RPS_log_validation_dataset_path', default='RPS_dataset3.pkl')
    parser.add_argument('--RPS_seq_with_recommended_size_h', default=-10, type=int)
    parser.add_argument('--RPS_truncation_seq', default=14, type=int)
    parser.add_argument('--RPS_user_interactions_with_text_title_predicted_by_SR_path',
                        default=_path_default)

    # ── 第二阶段 ──────────────────────────────────────────────────────────────
    parser.add_argument('--second_shuffle',              default=False, type=bool)
    parser.add_argument('--second_teacher_forcing',      default=False, type=bool)
    parser.add_argument('--second_predict_eos_token',    default=False, type=bool)
    parser.add_argument('--second_batch_size',           default=8,    type=int,
                        help='LLaMA3 建议 8 或更小（显存限制）')
    parser.add_argument('--second_decoder_max_length',   default=20,   type=int)
    parser.add_argument('--second_truncate_method',      default='tail')
    parser.add_argument('--second_max_seq_length',       default=1024, type=int,
                        help='LLaMA3 输入最大长度（token 数）')
    parser.add_argument('--second_model_path',           default='./model_llama3')
    parser.add_argument('--second_total_epoch',          default=20,   type=int)
    parser.add_argument('--second_lr',                   default=2e-4, type=float)
    parser.add_argument('--second_weight_decay',         default=1e-6, type=float)
    parser.add_argument('--second_num_warmup_steps',     default=100,  type=int)
    parser.add_argument('--second_num_training_steps',   default=500,  type=int)
    parser.add_argument('--second_gradient_accumulation_steps', default=4, type=int)
    parser.add_argument('--second_eval_every_steps',     default=50,   type=int)
    parser.add_argument('--second_if_peft',              default=True, type=bool)
    parser.add_argument('--second_peft_type',            default='LORA')
    parser.add_argument('--second_lora_r',               default=16,   type=int,
                        help='LoRA 秩（LLaMA3-3B 建议 8~16）')
    parser.add_argument('--second_lora_alpha',           default=32,   type=int,
                        help='LoRA scaling，通常设为 2 * lora_r')
    parser.add_argument('--second_lora_dropout',         default=0.05, type=float)
    parser.add_argument('--second_target_modules',       default=['q', 'v'], nargs='+')

    # ── LSR 参数 ──────────────────────────────────────────────────────────────
    parser.add_argument('--LSR_prompt_id',     default=0, choices=[0,1,2], type=int)
    parser.add_argument('--LSR_load',          default=False, type=bool)
    parser.add_argument('--LSR_load_log',      default=False, type=bool)
    parser.add_argument('--LSR_all_item_titles_path', default='./title_set.csv')
    parser.add_argument('--LSR_log_train_dataset_path',      default='LSR_dataset1.pkl')
    parser.add_argument('--LSR_log_test_dataset_path',       default='LSR_dataset2.pkl')
    parser.add_argument('--LSR_log_validation_dataset_path', default='LSR_dataset3.pkl')
    parser.add_argument('--LSR_truncation_seq', default=5, type=int)
    parser.add_argument('--LSR_user_interactions_with_text_title_ground_truth_path',
                        default='./user_interactions_with_text_title_and_ground_truth')

    args = parser.parse_args()
    main(args)
