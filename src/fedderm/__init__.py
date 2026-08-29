"""
FedDerm: Federated Learning for Privacy-Preserving Skin Lesion Classification.

Package structure
-----------------
fedderm/
├── data/        Dataset loading, partitioning (IID / non-IID Dirichlet splits)
├── models/      Model architectures (CNN baseline, ViT backbone, LoRA adapters)
├── federated/   FL simulation: strategies (FedAvg, FedProx, SCAFFOLD, FedDyn)
├── privacy/     Differential-privacy wrappers (Opacus DP-SGD, DP-LoRA)
└── utils/       Metrics, logging, reproducibility helpers, visualisation
"""

__version__ = "0.1.0"
__author__ = "FedDerm Research"
