"""
DELRec/llms/llama3_wrapper.py
==============================
LLaMA 3 (meta-llama/Llama-3.2-3B-Instruct) 自定义封装。

背景
----
openprompt 的 load_plm 对 LLaMA 3 支持不完善（LLaMA 3 使用 tiktoken 分词器，
与 openprompt 内部假设不符）。本模块绕开 openprompt，直接用 transformers +
PEFT 实现：
    1. 加载 LLaMA-3 模型 & 分词器
    2. 在最后隐藏层上加线性分类头（用于 100 选一候选排序）
    3. 支持 AdaLoRA / LoRA fine-tuning（第二阶段）
    4. 支持 soft-prompt（第一阶段，可冻结 LLM 权重，只训练 soft token embeddings）

推理方式
--------
给定 prompt（包含 user history + candidates），把候选列表拼入 prompt，
取 [EOS] / [LAST] 位置的隐向量经分类头得到每个候选的 logit，
与原始 DELRec 保持相同接口：输出 shape = (batch, num_candidates)。
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    AdaLoraConfig,
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
)

# ─────────────────────────────────────────────────────────────────────────────
# 默认模型标识（HuggingFace Hub 或本地路径）
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

# Instruct 系统提示
SYSTEM_PROMPT = (
    "You are a helpful sequential recommendation assistant. "
    "Given a user's interaction history and a candidate list, "
    "predict which item the user will interact with next."
)


# ─────────────────────────────────────────────────────────────────────────────
# 分词器工具
# ─────────────────────────────────────────────────────────────────────────────

def load_llama3_tokenizer(model_path: str = DEFAULT_MODEL_ID):
    """加载 LLaMA 3 分词器，设置 pad_token 为 eos_token。"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True,
    )
    # LLaMA 3 没有独立 pad_token，用 eos_token 替代
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"   # decoder-only 模型左填充
    return tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 核心模型：LLaMA3Recommender
# ─────────────────────────────────────────────────────────────────────────────

class LLaMA3Recommender(nn.Module):
    """
    LLaMA 3 + 分类头，用于 100 选一候选排序。

    参数
    ----
    model_path   : 本地目录或 HuggingFace Hub 模型 ID
    num_classes  : 候选集大小（默认 100）
    load_in_4bit : 是否使用 bitsandbytes 4-bit 量化（显存不足时开启）
    load_in_8bit : 是否使用 8-bit 量化
    soft_prompt_len : soft-prompt token 数（0 = 不使用）
    freeze_llm   : 是否冻结 LLaMA 权重（第一阶段只训练 soft-prompt）
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_ID,
        num_classes: int = 100,
        load_in_4bit: bool = False,
        load_in_8bit: bool = True,    # 默认 8-bit，平衡精度与显存
        soft_prompt_len: int = 100,
        freeze_llm: bool = True,
    ):
        super().__init__()

        # 4-bit 与 8-bit 互斥；4-bit 优先（用户显式传入时）
        if load_in_4bit and load_in_8bit:
            load_in_8bit = False

        # ── 量化配置 ──────────────────────────────────────────────────────────
        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif load_in_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_4bit_compute_dtype=torch.float16,  # 显式传入，绕过新版 transformers 的 torch 引用 bug
            )

        quant_label = "4-bit" if load_in_4bit else ("8-bit" if load_in_8bit else "bf16 full")

        # ── 加载 LLaMA 3 ─────────────────────────────────────────────────────
        print(f"[LLaMA3] 加载模型: {model_path}  量化: {quant_label}")
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16 if not (load_in_4bit or load_in_8bit) else None,
            device_map="auto",
            trust_remote_code=True,
        )
        # 开启梯度检查点，以时间换显存（显存减少约 40%）
        if hasattr(self.llm, 'gradient_checkpointing_enable'):
            self.llm.gradient_checkpointing_enable()
            self.llm.config.use_cache = False  # 梯度检查点与 kv-cache 不兼容
        # 开启梯度检查点，以计算时间换显存（训练时约节省 30~40% 显存）
        if hasattr(self.llm, 'gradient_checkpointing_enable'):
            self.llm.gradient_checkpointing_enable()
            self.llm.config.use_cache = False  # 梯度检查点与 kv-cache 不兼容
        self.hidden_size = self.llm.config.hidden_size

        # ── Soft-prompt embeddings ────────────────────────────────────────────
        self.soft_prompt_len = soft_prompt_len
        if soft_prompt_len > 0:
            self.soft_embeddings = nn.Embedding(soft_prompt_len, self.hidden_size)
            nn.init.normal_(self.soft_embeddings.weight, std=0.02)
        else:
            self.soft_embeddings = None

        # ── 分类头 ────────────────────────────────────────────────────────────
        # 输入：[LAST TOKEN] 的隐向量；输出：num_classes 个 logit
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, num_classes),
        )

        # ── 冻结 LLM（第一阶段只训练 soft-prompt & classifier）────────────────
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False

    def _get_last_hidden(self, inputs_embeds, attention_mask):
        """用 hook 只捕获最后一层隐向量，避免 output_hidden_states=True 存全部28层。"""
        last_hidden = {}

        def hook(module, input, output):
            # output[0] shape: (B, seq_len, H)
            last_hidden['h'] = output[0]

        # 注册到最后一个 decoder layer
        handle = self.llm.model.layers[-1].register_forward_hook(hook)
        try:
            self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
        finally:
            handle.remove()

        return last_hidden['h']   # (B, seq_len, H)

    # ── 前向传播 ──────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,       # (B, L)
        attention_mask: torch.Tensor,  # (B, L)
    ) -> torch.Tensor:
        B, L = input_ids.shape

        if self.soft_embeddings is not None:
            token_embeds   = self.llm.get_input_embeddings()(input_ids)
            soft_tok_idx   = torch.arange(self.soft_prompt_len, device=input_ids.device)
            soft_emb       = self.soft_embeddings(soft_tok_idx).to(token_embeds.device)
            soft_emb       = soft_emb.unsqueeze(0).expand(B, -1, -1)
            inputs_embeds  = torch.cat([soft_emb, token_embeds], dim=1)
            soft_mask      = torch.ones(B, self.soft_prompt_len,
                                        dtype=attention_mask.dtype,
                                        device=attention_mask.device)
            attention_mask = torch.cat([soft_mask, attention_mask], dim=1)
        else:
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        # 统一转为 LLM dtype
        llm_dtype = next(self.llm.parameters()).dtype
        inputs_embeds = inputs_embeds.to(llm_dtype)

        last_hidden = self._get_last_hidden(inputs_embeds, attention_mask)  # (B, S, H)
        seq_lens    = attention_mask.sum(dim=1) - 1                          # (B,)
        last_vecs   = last_hidden[torch.arange(B), seq_lens]                 # (B, H)

        logits = self.classifier(last_vecs.float())
        return logits

    # ── PEFT 相关 ──────────────────────────────────────────────────────────────

    def apply_lora(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
    ):
        """
        在 LLM 上应用标准 LoRA（第二阶段微调）。
        LLaMA3-3B 推荐：r=16, lora_alpha=32, target_modules=[q_proj, v_proj]
        """
        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        self.llm = get_peft_model(self.llm, lora_config)
        self.llm.print_trainable_parameters()
        return self

    def apply_adalora(
        self,
        init_r: int = 64,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
    ):
        """AdaLoRA（备用，动态调整秩）。"""
        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]

        adalora_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            init_r=init_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
        )
        self.llm = get_peft_model(self.llm, adalora_config)
        self.llm.print_trainable_parameters()
        return self

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        # 保存分类头 & soft-prompt
        torch.save({
            'classifier'    : self.classifier.state_dict(),
            'soft_embeddings': self.soft_embeddings.state_dict()
                               if self.soft_embeddings else None,
        }, os.path.join(path, "head.pt"))
        # 保存 LLM（含 PEFT adapter）
        if hasattr(self.llm, 'save_pretrained'):
            self.llm.save_pretrained(os.path.join(path, "llm"))
        print(f"[LLaMA3] 模型已保存到: {path}")

    def load(self, path: str):
        head = torch.load(os.path.join(path, "head.pt"), map_location='cpu')
        self.classifier.load_state_dict(head['classifier'])
        if head['soft_embeddings'] and self.soft_embeddings:
            self.soft_embeddings.load_state_dict(head['soft_embeddings'])
        llm_path = os.path.join(path, "llm")
        if os.path.exists(llm_path):
            self.llm = PeftModel.from_pretrained(self.llm, llm_path)
        print(f"[LLaMA3] 模型已从 {path} 加载")
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Prompt 构建工具
# ─────────────────────────────────────────────────────────────────────────────

def build_llama3_prompt(
    user_seq: str,
    candidates: str,
    model_name: str = "SASRec",
    soft_prompt_placeholder: str = "<|soft_prompt|>",
) -> str:
    """
    构建 LLaMA 3 Instruct 格式的 prompt。

    Parameters
    ----------
    user_seq    : 用户历史序列（逗号分隔的 item title 字符串）
    candidates  : 候选集（逗号分隔的 item title 字符串）
    model_name  : SR 模型名称（写入 prompt 上下文）

    Returns
    -------
    prompt : str（完整 instruct 格式文本）
    """
    user_content = (
        f"Refer to the recommendation pattern of the {model_name} model "
        f"{soft_prompt_placeholder} "
        f"to predict the next item the user will interact with.\n\n"
        f"User interaction history: {user_seq}\n\n"
        f"Candidate set: {candidates}\n\n"
        f"The next item the user will interact with is:"
    )
    # LLaMA 3 Instruct 格式
    prompt = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return prompt


def collate_llama3_batch(
    batch: List[dict],
    tokenizer,
    max_length: int = 1024,
) -> dict:
    """
    将一批样本转换为模型输入张量。

    Parameters
    ----------
    batch : list of dict，每条含 user_seq / candidates / model_name / label_idx

    Returns
    -------
    dict with keys: input_ids, attention_mask, labels
    """
    texts  = []
    labels = []
    for item in batch:
        # user_seq / candidates 可能是 list[str]，统一转为逗号分隔字符串
        user_seq   = item['user_seq']
        candidates = item['candidates']
        if isinstance(user_seq,   list): user_seq   = ', '.join(user_seq)
        if isinstance(candidates, list): candidates = ', '.join(candidates)
        prompt = build_llama3_prompt(
            user_seq=user_seq,
            candidates=candidates,
            model_name=item.get('model_name', 'SASRec'),
        )
        texts.append(prompt)
        labels.append(item['label_idx'])

    encoding = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {
        **encoding,
        'labels': torch.tensor(labels, dtype=torch.long),
    }