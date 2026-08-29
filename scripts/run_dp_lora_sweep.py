"""
scripts/run_dp_lora_sweep.py
----------------------------
Automated DP-LoRA (ViT-B/16 + LoRA r=8) multi-seed privacy-utility sweep CLI driver.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fedderm.experiments.dp_lora_sweep import (
    DEFAULT_NOISE_MULTIPLIERS,
    DEFAULT_SEEDS,
    run_sweep,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DP-LoRA multi-seed sweep driver")
    parser.add_argument("--sanity-only", action="store_true", help="Run only the Non-DP sanity check")
    parser.add_argument("--sweep-only", action="store_true", help="Run only the multi-seed DP sweep")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS, help="Random seeds to evaluate")
    parser.add_argument(
        "--noise-multipliers",
        nargs="+",
        type=float,
        default=DEFAULT_NOISE_MULTIPLIERS,
        help="Noise multipliers (sigma) to sweep",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/dp_lora_fedprox",
        help="Output base directory for results",
    )
    args = parser.parse_args()

    run_sanity = not args.sweep_only
    run_dp = not args.sanity_only

    run_sweep(
        noise_multipliers=args.noise_multipliers,
        seeds=args.seeds,
        run_sanity=run_sanity,
        run_dp=run_dp,
        results_base=args.output_dir,
    )
