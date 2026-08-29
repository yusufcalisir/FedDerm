"""
fedderm/models/vit_lora.py
--------------------------
Vision Transformer (ViT-B/16) with Low-Rank Adaptation (LoRA) adapters.

Freezes 99.65% of the backbone parameters and fine-tunes only rank-r
adapters on query/key/value attention projections plus the linear
classification head, reducing trainable parameters to ~300k.
"""

from __future__ import annotations

from typing import Sequence
import timm
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model


def build_vit_lora_model(
    model_name: str = "vit_base_patch16_224",
    num_classes: int = 7,
    rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    target_modules: Sequence[str] | None = None,
    modules_to_save: Sequence[str] | None = None,
    pretrained: bool = True,
) -> nn.Module:
    """Instantiate a ViT backbone from timm and wrap it with PEFT LoRA adapters.

    Args:
        model_name:      timm model architecture name (default: vit_base_patch16_224).
        num_classes:     Number of output target classes (DermaMNIST has 7).
        rank:            LoRA rank r (default: 8).
        lora_alpha:      LoRA scaling parameter alpha (default: 2 * rank = 16).
        lora_dropout:    LoRA dropout rate (default: 0.0 for DP compatibility).
        target_modules:  Module names to attach LoRA adapters to (default: ['qkv']).
        modules_to_save: Module names to train fully without LoRA (default: ['head']).
        pretrained:      Whether to load ImageNet pretrained weights.

    Returns:
        peft.PeftModel wrapping the timm Vision Transformer.
    """
    if target_modules is None:
        target_modules = ["qkv"]
    if modules_to_save is None:
        modules_to_save = ["head"]

    base_model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=list(target_modules),
        lora_dropout=lora_dropout,
        bias="none",
        modules_to_save=list(modules_to_save),
    )

    model = get_peft_model(base_model, lora_config)
    return model
