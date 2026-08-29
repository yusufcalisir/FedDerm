"""
smoke_test.py -- Phase 1 environment verification
==================================================
Imports every core dependency and reports GPU / CPU availability.
Run with:  python scripts/smoke_test.py
Expected:  all lines print OK with version numbers.
"""

from __future__ import annotations

import sys

# Force UTF-8 stdout on Windows (avoids cp1252/cp1254 codec errors)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

OK  = "[ OK ]"
ERR = "[FAIL]"
SEP = "=" * 44


def _check(label: str, fn: "callable[[], str]") -> None:
    try:
        result = fn()
        print(f"  {OK}  {label}: {result}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {ERR}  {label}: FAILED -- {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    print(f"\n{SEP}")
    print("  FedDerm - Phase 1 Environment Smoke Test")
    print(f"{SEP}\n")

    # ── Core imports ──────────────────────────────────────────────────────────
    _check("numpy", lambda: __import__("numpy").__version__)
    _check("torch", lambda: __import__("torch").__version__)
    _check("torchvision", lambda: __import__("torchvision").__version__)
    _check("flwr (Flower)", lambda: __import__("flwr").__version__)
    _check("opacus", lambda: __import__("opacus").__version__)
    _check("medmnist", lambda: __import__("medmnist").__version__)
    _check("sklearn", lambda: __import__("sklearn").__version__)
    _check("matplotlib", lambda: __import__("matplotlib").__version__)
    _check("seaborn", lambda: __import__("seaborn").__version__)
    _check("tqdm", lambda: __import__("tqdm").__version__)
    _check("PIL (Pillow)", lambda: __import__("PIL").__version__)
    _check("pandas", lambda: __import__("pandas").__version__)
    _check("hydra-core", lambda: __import__("hydra").__version__)
    _check("omegaconf", lambda: __import__("omegaconf").__version__)

    # ── fedderm package self-import ───────────────────────────────────────────
    _check("fedderm (this package)", lambda: __import__("fedderm").__version__)

    # ── PyTorch compute device ────────────────────────────────────────────────
    print()
    import torch  # noqa: PLC0415

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  {OK}  GPU: {gpu_name} ({vram:.1f} GB VRAM)")
    else:
        print(f"  [INFO] No CUDA GPU detected - running on CPU only.")
        print("         (Training will be slow; use Colab/Kaggle for full experiments.)")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print(f"  {OK}  Apple MPS (Metal) backend available.")

    print()
    print(SEP)
    print("  All checks passed. Environment is ready.")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
