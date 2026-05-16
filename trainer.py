'''
    adjust by 田宇瀚 2025.12.02
'''
from typing import Dict, List, Optional, Sequence, Tuple, Callable

import torch
# from flash_attn.flash_attn_interface import flash_attn_varlen_func
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers import Trainer
from transformers.cache_utils import Cache
from transformers.utils.deprecation import deprecate_kwarg
from transformers.processing_utils import Unpack
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
    Qwen2VLModel,
    apply_multimodal_rotary_pos_emb,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModel,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionModel,
    Qwen3VLModel,
    apply_rotary_pos_emb,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeVisionModel,
    Qwen3VLMoeModel,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)


'''
def return_mask(
    config,
    input_embeds,
    attention_mask,
    cache_position,
    past_key_values,
    position_ids,
    **kwargs
):
    return attention_mask
'''
'''
def replace_qwen2_vl_attention_class():
    """
    Replace only the mask-creation helpers for Qwen VL families to use our return_mask,
    and force native attention usage by making get_device_map return an empty dict.

    This function is idempotent: calling it multiple times is safe.
    """
    import transformers

    try:
        # qwen2_vl
        transformers.models.qwen2_vl.modeling_qwen2_vl.create_causal_mask = return_mask
        transformers.models.qwen2_vl.modeling_qwen2_vl.create_sliding_window_causal_mask = return_mask
    except Exception:
        logger.warning("qwen2_vl masks not found — skipping qwen2_vl mask patch")

    try:
        # qwen2_5_vl
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.create_causal_mask = return_mask
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.create_sliding_window_causal_mask = return_mask
    except Exception:
        logger.warning("qwen2_5_vl masks not found — skipping qwen2_5_vl mask patch")

    try:
        # qwen3vl
        transformers.models.qwen3_vl.modeling_qwen3_vl.create_causal_mask = return_mask
    except Exception:
        logger.warning("qwen3_vl mask not found — skipping qwen3_vl mask patch")

    try:
        # qwen3vl moe
        transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.create_causal_mask = return_mask
    except Exception:
        logger.warning("qwen3_vl_moe mask not found — skipping qwen3_vl_moe mask patch")

    # 强制使用原生注意力（避免 transformers 在内部启用 flash-attn / model-parallel）
    # 返回 {} 比返回 None 更安全（避免外部代码期望 dict）
    try:
        transformers.utils.model_parallel_utils.get_device_map = lambda *args, **kwargs: {"",[]}
    except Exception:
        logger.warning("Could not override get_device_map — this may be okay depending on transformers version.")
'''
def print_trainable_parameters_visual(self) -> None:
    """
    Prints the trainable status of all vision components including attention blocks and merger module.
    Outputs the indices of trainable/non-trainable blocks and the merger module status.
    """
    trainable_blocks = []
    non_trainable_blocks = []

    # Check trainable status of vision attention blocks
    for block_idx, block in enumerate(self.blocks):
        is_trainable = all(param.requires_grad for param in block.parameters())
        if is_trainable:
            trainable_blocks.append(block_idx)
        else:
            non_trainable_blocks.append(block_idx)

    # Check trainable status of merger module
    is_merger_trainable = any(param.requires_grad for param in self.merger.parameters())

    # Print results
    print("Vision Module - Attention Blocks:")
    print(
        f"Trainable Block Indices: {trainable_blocks if trainable_blocks else 'None'}"
    )
    print(
        f"Non-Trainable Block Indices: {non_trainable_blocks if non_trainable_blocks else 'None'}"
    )
    print(f"Merger Module Trainable: {is_merger_trainable}")


def print_trainable_parameters(self) -> None:
    """
    Prints the trainable status of all LLM components including embeddings, layers, and normalization.
    Outputs the indices of trainable/non-trainable layers and other module statuses.
    """
    # Check embed_tokens
    is_embed_trainable = any(
        param.requires_grad for param in self.language_model.embed_tokens.parameters()
    )
    print(f"LLM Module - Embed Tokens Trainable: {is_embed_trainable}")

    # Check each decoder layer
    trainable_layers = []
    non_trainable_layers = []

    for layer_idx, layer in enumerate(self.language_model.layers):
        is_trainable = any(param.requires_grad for param in layer.parameters())
        if is_trainable:
            trainable_layers.append(layer_idx)
        else:
            non_trainable_layers.append(layer_idx)

    # Print layer status
    print(
        f"LLM Module - Trainable Layer Indices: {trainable_layers if trainable_layers else 'None'}"
    )
    print(
        f"LLM Module - Non-Trainable Layer Indices: {non_trainable_layers if non_trainable_layers else 'None'}"
    )


def create_optimizer(self):

    opt_model = self.model

    if self.optimizer is None:
        decay_parameters = self.get_decay_parameter_names(opt_model)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        if self.args.mm_projector_lr is not None and self.args.mm_projector_lr != 0:
            projector_parameters = [
                name for name, _ in opt_model.named_parameters() if "merger" in name
            ]
            if self.args.vision_tower_lr is not None and self.args.vision_tower_lr != 0:
                vision_tower_parameters = [
                    name for name, _ in opt_model.named_parameters() if "visual" in name
                ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.vision_tower_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.vision_tower_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args
        )
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    return self.optimizer

# ========== Force disable FlashAttention for Qwen models ==========
'''
import transformers

def disable_flash_attn(model):
    """
    Ensures the model uses eager attention even if transformers or config tries to enable flash-attn.
    """
    if hasattr(model, "config"):
        try:
            # 对 Qwen3 / Qwen-VL 系列都有效
            model.config._attn_implementation = "eager"
            if hasattr(model.config, "attn_implementation"):
                model.config.attn_implementation = "eager"
        except Exception as e:
            print("[Warning] Failed to set model.config._attn_implementation:", e)

    # 如果模型内部有使用 FlashAttentionKwargs，也强制关闭
    try:
        if hasattr(model, "forward"):
            model.forward = torch._dynamo.disable(model.forward)
    except:
        pass

    print(">>> FlashAttention 已强制关闭（设置为 eager）")
    return model


# Patch Trainer to auto-apply this fix
_old_trainer_init = Trainer.__init__

def _new_trainer_init(self, *args, **kwargs):
    _old_trainer_init(self, *args, **kwargs)

    # 确保模型已经被加载
    if getattr(self, "model", None) is not None:
        disable_flash_attn(self.model)

Trainer.__init__ = _new_trainer_init

print(">>> Trainer patched: FlashAttention disabled by default.")
'''
# ================================================================

# Apply monkey patches
Trainer.create_optimizer = create_optimizer

Qwen2VisionTransformerPretrainedModel.print_trainable_parameters = (
    print_trainable_parameters_visual
)
Qwen2VLModel.print_trainable_parameters = print_trainable_parameters
Qwen2_5_VisionTransformerPretrainedModel.print_trainable_parameters = (
    print_trainable_parameters_visual
)
Qwen2_5_VLModel.print_trainable_parameters = print_trainable_parameters

Qwen3VLVisionModel.print_trainable_parameters = (
    print_trainable_parameters_visual
)
Qwen3VLModel.print_trainable_parameters = print_trainable_parameters
Qwen3VLMoeVisionModel.print_trainable_parameters = print_trainable_parameters_visual
Qwen3VLMoeModel.print_trainable_parameters = print_trainable_parameters