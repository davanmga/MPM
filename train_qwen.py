'''
    adjust by 田宇瀚 2025.12.02
'''
# ======== 强制禁用 FlashAttention（必须放在最开头，import 之前） ========
import os
os.environ["DISABLE_FLASH_ATTN"] = "1"
os.environ["FLASH_ATTENTION_DISABLED"] = "1"
os.environ["ENABLE_FLASH_ATTN"] = "0"
os.environ["USE_FLASH_ATTENTION"] = "0"
os.environ["PYTORCH_CUDA_FUSER_DISABLE_FALLBACK"] = "1"

# ======== 原始版权保留 ========
# Adopted from https://github.com/lm-sys/FastChat ...
# (以下省略版权说明)

import logging
import pathlib
import torch
import transformers
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


# ====== Qwen 模型家族 ======
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoProcessor, Trainer

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    # Vision Tower
    for p in model.visual.parameters():
        p.requires_grad = bool(model_args.tune_mm_vision)

    # MLP bridge
    for p in model.visual.merger.parameters():
        p.requires_grad = bool(model_args.tune_mm_mlp)

    # Language model
    for p in model.language_model.parameters():
        p.requires_grad = bool(model_args.tune_mm_llm)
    model.lm_head.requires_grad = bool(model_args.tune_mm_llm)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    # ===== 强制禁用 FlashAttn 优先级更高 =====
    attn_implementation = "sdpa"  # 覆盖命令行

    # ===== 根据模型类型加载 =====
    model_path_lower = model_args.model_name_or_path.lower()
    if "qwen3" in model_path_lower and "a" in Path(model_args.model_name_or_path.rstrip("/")).name.lower():
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation="eager",
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen3" in model_path_lower:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen2.5" in model_path_lower:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2vl"

    print(f'Initialized model: {model_args.model_name_or_path} ({model.__class__.__name__})')
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)

    # 双向禁用缓存，避免内存爆炸
    model.config.use_cache = False

    # gradient checkpointing 支持
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # ===== 是否 LoRA =====
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        set_model(model_args, model)
        if local_rank in (0, None):
            model.visual.print_trainable_parameters()
            model.model.print_trainable_parameters()

    data_module = make_supervised_data_module(processor, data_args=data_args)

    # ================= DEBUG：检查数据加载是否完整 =================
    print("==== DEBUG: 加载数据集大小检查 ====")
    import json

    # 1. 原始 JSON 条数
    try:
        raw_data = json.load(open(data_args.data_path, "r", encoding="utf-8"))
        print("原始 train.json 条数 =", len(raw_data))
    except Exception as e:
        print("读取原始 JSON 失败：", e)

    # 2. DataProcessor 处理后的数据条数
    try:
        tmp_dm = make_supervised_data_module(processor, data_args=data_args)
        print("DataProcessor 返回条数 =", len(tmp_dm["train_dataset"]))
    except Exception as e:
        print("DataProcessor 解析失败：", e)

    print("==== DEBUG END ====")
    # ============================================================

    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    # ====== checkpoint ======
    trainer.train()

    trainer.save_state()
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    # 忽略 attn_implementation 参数（我们强制为 sdpa）
    sys.argv = sys.argv  # 不需要处理额外 args
    train()