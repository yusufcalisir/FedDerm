"""
scripts/benchmark_vit_lora.py
-----------------------------
Benchmark timing and Opacus compatibility for ViT-B/16 + LoRA on CPU.
Measures forward, backward, non-DP and DP-SGD pass timing.
"""

from __future__ import annotations

import time
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from peft import LoraConfig, get_peft_model
import timm

def test_model(model_name: str = "vit_base_patch16_224", rank: int = 8, batch_size: int = 64):
    print("=" * 70)
    print(f"BENCHMARKING: {model_name} with LoRA (r={rank}), Batch Size={batch_size}")
    print("=" * 70)

    # 1. Instantiate backbone
    print("Loading model backbone from timm...")
    t0 = time.time()
    base_model = timm.create_model(model_name, pretrained=True, num_classes=7)
    print(f"Model loaded in {time.time() - t0:.2f} s")

    # 2. Configure LoRA
    # In timm ViT, attention projection is in blocks[i].attn.qkv
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=["qkv"],
        lora_dropout=0.0,
        bias="none",
        modules_to_save=["head"],
    )
    lora_model = get_peft_model(base_model, lora_config)

    # Count parameters
    total_params = sum(p.numel() for p in lora_model.parameters())
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"Total parameters:     {total_params:,}")
    print(f"Frozen parameters:    {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"MiniCNN parameters:   468,967")
    ratio = trainable_params / 468967
    noise_ratio = (trainable_params / 468967) ** 0.5
    print(f"Trainable param ratio vs MiniCNN: {ratio:.4f}x ({1/ratio:.1f}x reduction)")
    print(f"DP Noise vector norm ratio (sqrt(d)): {noise_ratio:.4f}x ({1/noise_ratio:.1f}x noise reduction)")

    # 3. Check Opacus compatibility
    print("\nChecking Opacus compatibility with ModuleValidator...")
    errors = ModuleValidator.validate(lora_model, strict=False)
    if errors:
        print(f"Opacus compatibility warnings/errors: {errors}")
    else:
        print("Opacus ModuleValidator: PASSED (100% compatible)")

    # 4. Benchmark Single-Batch Forward + Backward (Non-DP)
    device = torch.device("cpu")
    lora_model.to(device)
    lora_model.train()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, lora_model.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    dummy_images = torch.randn(batch_size, 3, 224, 224, device=device)
    dummy_labels = torch.randint(0, 7, (batch_size,), device=device)

    # Warmup
    print("\nWarming up Non-DP pass...")
    for _ in range(2):
        optimizer.zero_grad()
        out = lora_model(dummy_images)
        loss = criterion(out, dummy_labels)
        loss.backward()
        optimizer.step()

    # Timing Non-DP
    n_iters = 5
    t0 = time.time()
    for _ in range(n_iters):
        optimizer.zero_grad()
        out = lora_model(dummy_images)
        loss = criterion(out, dummy_labels)
        loss.backward()
        optimizer.step()
    non_dp_batch_time = (time.time() - t0) / n_iters
    print(f"Non-DP 1 batch ({batch_size} samples) forward+backward: {non_dp_batch_time:.3f} s")

    # 5. Benchmark Opacus DP-SGD Single-Batch Forward + Backward
    print("\nSetting up Opacus PrivacyEngine on LoRA model...")
    privacy_engine = PrivacyEngine(secure_mode=False)
    dp_model, dp_optimizer, dp_loader = privacy_engine.make_private(
        module=lora_model,
        optimizer=optimizer,
        data_loader=torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(dummy_images, dummy_labels),
            batch_size=batch_size,
        ),
        noise_multiplier=1.0,
        max_grad_norm=1.0,
    )

    # Warmup DP
    print("Warming up DP-SGD pass...")
    for x, y in dp_loader:
        dp_optimizer.zero_grad()
        out = dp_model(x)
        loss = criterion(out, y)
        loss.backward()
        dp_optimizer.step()
        break

    # Timing DP
    t0 = time.time()
    for _ in range(n_iters):
        for x, y in dp_loader:
            dp_optimizer.zero_grad()
            out = dp_model(x)
            loss = criterion(out, y)
            loss.backward()
            dp_optimizer.step()
    dp_batch_time = (time.time() - t0) / n_iters
    print(f"DP-SGD 1 batch ({batch_size} samples) forward+backward: {dp_batch_time:.3f} s")

    # 6. Federated Simulation Extrapolation
    # Dataset: 7007 samples partitioned across 10 clients (~700 samples/client)
    # Batch size 64 -> ~11 batches per client epoch
    # 3 local epochs -> ~33 batches per client per round
    # 5 clients sampled per round -> 165 client batches per round
    # 20 rounds -> 3,300 client batches per simulation run
    batches_per_client_epoch = int(np.ceil(700 / batch_size))
    batches_per_round = 5 * (3 * batches_per_client_epoch)
    total_batches = 20 * batches_per_round

    print("\n" + "=" * 70)
    print("EXTRAPOLATED WALL-CLOCK RUNTIME ESTIMATES (CPU)")
    print("=" * 70)
    print(f"Federated Setup: 20 rounds, 5 sampled/round, 3 local epochs/client, batch_size={batch_size}")
    print(f"Total local training steps per run: {total_batches:,} steps")

    # Non-DP extrapolation (assuming parallel Ray actors across 4-8 physical cores)
    # Effective speedup from 4 worker processes is ~2.5x to 3.0x on 8 cores
    speedup = 2.5
    non_dp_total_sec = (total_batches * non_dp_batch_time) / speedup
    dp_total_sec = (total_batches * dp_batch_time) / speedup

    print(f"\n1. Non-DP Sanity Check Run (1 run):")
    print(f"   Sequential estimate: {total_batches * non_dp_batch_time / 60:.1f} min ({total_batches * non_dp_batch_time / 3600:.2f} hours)")
    print(f"   Parallel (4 Ray actors) estimate: {non_dp_total_sec / 60:.1f} min ({non_dp_total_sec / 3600:.2f} hours)")

    print(f"\n2. Full Multi-Seed DP Sweep (4 noise levels x 3 seeds = 12 runs):")
    print(f"   Sequential estimate: {12 * total_batches * dp_batch_time / 3600:.1f} hours")
    print(f"   Parallel (4 Ray actors) estimate: {12 * dp_total_sec / 3600:.1f} hours")

    print(f"\n3. Reduced Multi-Seed DP Sweep (4 noise levels x 2 seeds = 8 runs):")
    print(f"   Parallel (4 Ray actors) estimate: {8 * dp_total_sec / 3600:.1f} hours")
    print("=" * 70)

if __name__ == "__main__":
    import numpy as np
    test_model("vit_base_patch16_224", rank=8, batch_size=64)
