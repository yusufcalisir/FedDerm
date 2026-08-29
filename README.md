# FedDerm

**Federated Learning for Privacy-Preserving Skin Lesion Classification**

> A research project investigating privacy-utility trade-offs in cross-institutional
> federated learning under realistic small-sample, non-IID conditions.

See [`OVERVIEW.md`](OVERVIEW.md) for the full research motivation and planned
experimental arc.

---

## Quick Start

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install the package (editable mode + all deps)

```bash
pip install -e ".[dev]"
```

> For Phase 5 (DP-LoRA / ViT), also run:
> ```bash
> pip install -e ".[peft]"
> ```

### 3. Verify the environment

```bash
python scripts/smoke_test.py
```

### 4. Run the test suite

```bash
pytest tests/ -v
```

---

## Project Structure

```
src/fedderm/      ← installable Python package
configs/          ← Hydra experiment YAML configs
scripts/          ← training entry points
tests/            ← pytest test suite
notebooks/        ← EDA / analysis
data/             ← downloaded datasets (gitignored)
results/          ← experiment outputs (gitignored)
checkpoints/      ← saved weights (gitignored)
```

---

## Research Phases

| Phase | Status | Description |
|---|---|---|
| 1 | Done | Project setup, environment, documentation |
| 2 | Done | Centralized baseline (DermaMNIST, MiniCNN) |
| 3 | Next | Federated non-IID splits (FedAvg vs robust aggregation) |
| 4 | Pending | Differential privacy (DP-SGD, Opacus) |
| 5 | Pending | DP-LoRA on frozen ViT-B/16 backbone |
| 6 | Pending | Secure Aggregation + central DP |
| 7 | Pending | Final benchmark report |

---

## Citation

*To be added after paper submission.*

## License

MIT
