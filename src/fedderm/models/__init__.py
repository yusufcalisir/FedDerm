"""Model architecture definitions."""

from fedderm.models.minicnn import MiniCNN, build_model
from fedderm.models.vit_lora import build_vit_lora_model

__all__ = ["MiniCNN", "build_model", "build_vit_lora_model"]
