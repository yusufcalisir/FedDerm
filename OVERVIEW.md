# FedDerm: Project Overview

## Research Motivation

Modern dermatology AI achieves near-dermatologist accuracy when trained on
large, centralized image datasets. In practice, however, medical data is
siloed: hospitals cannot share raw patient images due to privacy regulations
(HIPAA, GDPR) and institutional policies. **Federated learning (FL)** addresses
this by training models collaboratively — each institution keeps its data
locally and only shares model updates.

Real federated healthcare deployments face two compounding challenges that are
seldom studied together:

1. **Heterogeneous, small-sample data (non-IID).**  
   Different hospitals serve different demographics. A rural clinic might have
   5–10 images per rare lesion class; an urban center hundreds. Class
   distributions vary dramatically across sites ("non-IID" or heterogeneous).
   This hurts standard FedAvg convergence badly.

2. **Privacy guarantees require differential privacy (DP).**  
   Simply not sharing raw data is not enough — gradient updates can still leak
   patient information. Differential Privacy (DP-SGD via Opacus) provably
   bounds leakage but introduces noise that crushes accuracy when per-hospital
   datasets are tiny.

FedDerm investigates these two challenges on **DermaMNIST** (7-class skin
lesion classification derived from HAM10000), a realistic benchmark where
sample sizes naturally resemble rare-disease settings. The project culminates
in a parameter-efficient solution (DP-LoRA on a frozen pretrained backbone)
that recovers meaningful accuracy under strict privacy budgets.

---

## Research Arc

### Phase 1 — Project Setup *(this phase)*
Reproducible environment, package structure, version control, and this
overview document. No training logic yet.

### Phase 2 — Centralized Baseline
Train a standard ResNet-18 on the full DermaMNIST dataset (centralized,
no federation, no privacy). Establishes the performance ceiling for later
comparisons. Reports: accuracy, balanced accuracy, AUC-ROC, confusion matrix.

### Phase 3 — Federated Learning with Non-IID Splits
Simulate N hospital clients by partitioning DermaMNIST using Dirichlet
sampling (α ∈ {0.05, 0.1, 0.5, ∞}) to produce varying degrees of
heterogeneity. Smaller α → more extreme imbalance (some clients hold only
one or two classes). Compare:
- **FedAvg** — McMahan et al. (2017), the standard baseline
- **FedProx / SCAFFOLD / FedDyn** — heterogeneity-robust alternatives  
  *(specific method chosen after Phase 2 literature review)*

### Phase 4 — Differential Privacy (DP-SGD)
Apply local DP via Opacus DP-SGD at each client. Track the
accuracy–privacy trade-off across a grid of (ε, δ) budgets. Demonstrate
the expected utility collapse: under small per-hospital n, even modest
ε values destroy model quality. This motivates Phase 5.

### Phase 5 — DP-LoRA: Parameter-Efficient Privacy
Replace fine-tuning of the full model with **LoRA adapters** on a frozen
pretrained ViT-B/16 backbone (ImageNet-pretrained). Only the low-rank
adapter weights are trainable, so the DP noise is concentrated over a much
smaller parameter space → dramatically better privacy-accuracy trade-off at
the same ε. Compare DP-LoRA vs full DP-SGD at identical ε.

### Phase 6 — Secure Aggregation & Central DP
Explore whether **Secure Aggregation** (hiding individual updates from the
server) plus **central DP** (server adds noise to the aggregate, clients
train without DP) outperforms local DP at the per-hospital scale we face.
Feasibility depends on Flower's SecAgg support.

### Phase 7 — Final Benchmark Report
Produce a comprehensive comparison table across all methods (accuracy,
privacy budget, communication cost, wall-clock time). Write up as a short
research paper targeting MICCAI workshops (DART or FAIMI) or similar venues.

---

## Dataset

**DermaMNIST** (from the MedMNIST v2 benchmark, `medmnist` package):
- Derived from HAM10000 (human against machine — 10 000 training images)
- 7-class skin lesion classification: mel, nv, bcc, akiec, bkl, df, vasc
- Pre-resized to 28×28 (or 64×64 / 224×224 via transforms)
- Official train/val/test split provided
- Class imbalance: dominant nv class vs rare df/vasc (<3% each) —
  naturally mirrors a rare-disease small-sample scenario

---

## Repository Structure

```
FedDerm/
├── src/fedderm/          # Installable Python package
│   ├── data/             # Dataset loading, Dirichlet partitioning
│   ├── models/           # CNN baselines, ViT backbone, LoRA adapters
│   ├── federated/        # FL strategies (FedAvg, FedProx, SCAFFOLD, …)
│   ├── privacy/          # Opacus wrappers, DP accountant utilities
│   └── utils/            # Metrics, logging, seeding, visualisation
├── configs/              # Hydra experiment configs (YAML)
│   └── experiments/      # One file per experiment variant
├── scripts/              # Entry-point training scripts
│   ├── smoke_test.py     # Phase 1: environment check
│   ├── train_centralized.py
│   ├── train_federated.py
│   └── train_dp.py
├── tests/                # pytest unit tests (one file per phase)
├── notebooks/            # EDA and result analysis notebooks
├── data/                 # Downloaded datasets (gitignored)
├── results/              # Experiment outputs (gitignored)
├── checkpoints/          # Saved model weights (gitignored)
├── logs/                 # Training logs (gitignored)
├── pyproject.toml        # Project metadata and dependencies
└── OVERVIEW.md           # This document
```

---

## Reproducibility Commitments

- All experiments seeded globally (PyTorch, NumPy, Python `random`)
- Exact dependency versions pinned in `pyproject.toml`
- Hydra config captures every hyperparameter; configs committed to git
- Results written to timestamped subdirectories under `results/`
- Checkpoints saved with their associated config hash

---

## Key References

- McMahan et al. (2017) — *Communication-Efficient Learning of Deep Networks
  from Decentralized Data* (FedAvg)
- Li et al. (2020) — *FedProx: Federated Optimization in Heterogeneous Networks*
- Karimireddy et al. (2020) — *SCAFFOLD: Stochastic Controlled Averaging for FL*
- Abadi et al. (2016) — *Deep Learning with Differential Privacy* (DP-SGD)
- Hu et al. (2022) — *LoRA: Low-Rank Adaptation of Large Language Models*
- Yang et al. (2023) — *MedMNIST v2: A Large-Scale Lightweight Benchmark*
- Diao et al. (2023) — *FedDyn: Federated Dynamic Optimization*

---

*Last updated: Phase 1 (project setup)*
