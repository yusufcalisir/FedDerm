# FedDerm Experimental Results

## Centralized Baseline (DermaMNIST)

### Experiment Setup

- **Dataset**: DermaMNIST (MedMNIST v2), 7-class dermatoscopic skin lesion classification derived from HAM10000.
  - Image resolution: 28x28 (3 channels RGB).
  - Sample counts: 7,007 train, 1,003 val, 2,005 test (official splits).
- **Architecture**: `MiniCNN` (4 convolutional blocks with BatchNorm, ReLU, AdaptiveAvgPool2d, Dropout(0.4), and 2-layer classifier head; 468,967 trainable parameters).
- **Optimization**: Adam optimizer with initial learning rate 1e-3, weight decay 1e-4, and Cosine Annealing scheduler across 30 epochs. Batch size 64.
- **Loss function**: CrossEntropyLoss with inverse-frequency class weights computed from the training split to counteract severe class imbalance.
- **Hardware & Runtime**: Local CPU execution (16 threads utilized), total training time: 11.9 minutes (713.7 seconds).

### Evaluation Metrics (Official Test Split)

| Metric | Score |
|---|---|
| Test Accuracy | 64.64% |
| Balanced Accuracy | 53.23% |
| Macro F1 | 44.03% |
| Weighted F1 | 68.23% |
| Best Validation Accuracy | 62.91% |

### Per-Class Performance Breakdown

| Class Index | Diagnosis | Support | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| 0 | Actinic keratoses / intraepithelial carcinoma (akiec) | 66 | 0.35 | 0.33 | 0.344 |
| 1 | Basal cell carcinoma (bcc) | 103 | 0.45 | 0.47 | 0.459 |
| 2 | Benign keratosis-like lesions (bkl) | 220 | 0.42 | 0.43 | 0.424 |
| 3 | Dermatofibroma (df) | 23 | 0.07 | 0.43 | 0.117 |
| 4 | Melanoma (mel) | 223 | 0.35 | 0.54 | 0.429 |
| 5 | Melanocytic nevi (nv) | 1341 | 0.92 | 0.73 | 0.814 |
| 6 | Vascular lesions (vasc) | 29 | 0.36 | 0.79 | 0.495 |

### Analysis and Observations

1. **Severe Class Imbalance**: Melanocytic nevi (nv) constitutes ~67% of the dataset (1,341 / 2,005 test samples). Standard accuracy (64.64%) is inflated by strong performance on the majority class (F1: 81.4%). In contrast, balanced accuracy is 53.23% and macro F1 is 44.03%.
2. **Rare Class Vulnerability**: Dermatofibroma (df) has only 23 test samples; while class weighting helped achieve 43% recall, precision remained low (0.07), leading to an F1 of 11.7%. Melanoma (mel) achieved 54% recall with 0.35 precision.
3. **Hardware Assessment**: Training 30 epochs took ~12 minutes on CPU with a lightweight CNN (~469k params).

---

## Federated Baseline: FedAvg with Non-IID Dirichlet Partitioning

### Experiment Setup

- **Dataset**: DermaMNIST (28x28, 7 classes), 7,007 training samples partitioned across 10 simulated hospitals.
- **Partitioning Strategy**: Dirichlet distribution over class proportions ($\alpha = 0.3$, seed 42).
  - High cross-client statistical heterogeneity: e.g. Client 8 holds 2,738 `nv` and 0 `df`/`bcc`; Client 4 holds 124 `bcc`, 51 `vasc`, 29 `df`, but 0 `mel` and 1 `nv`; Client 0 holds 116 `mel` and only 6 `nv`. Full per-client matrix logged in `results/federated_fedavg/partition_report.json`.
- **Federated Algorithm**: Standard FedAvg (McMahan et al., 2017) via Flower simulation backend.
  - 10 total clients, 5 sampled per round (50% participation rate).
  - 20 communication rounds, 3 local training epochs per round (batch size 64, Adam lr=1e-3, weight decay 1e-4).
  - Model architecture: `MiniCNN` (identical to centralized baseline for direct comparability).
  - Loss: CrossEntropyLoss with inverse-frequency class weights computed on the full dataset.
- **Hardware & Runtime**: Local CPU execution, total simulation time: 4.4 minutes (264.5 seconds).

### Comparison: Centralized vs. Federated (FedAvg)

| Metric | Centralized Baseline | Federated (FedAvg, $\alpha=0.3$) | Absolute Degradation ($\Delta$) |
|---|---|---|---|
| **Test Accuracy** | **64.64%** | **14.51%** | **-50.13%** |
| **Balanced Accuracy** | **53.23%** | **27.32%** | **-25.91%** |
| **Macro F1** | **44.03%** | **10.81%** | **-33.22%** |
| **Weighted F1** | **68.23%** | **4.99%** | **-63.24%** |
| **Best Val Accuracy** | **62.91%** | **14.46%** | **-48.45%** |

### Per-Class F1 Score Comparison

| Class Index | Diagnosis | Support | Centralized F1 | FedAvg F1 | $\Delta$ F1 |
|---|---|---|---|---|---|
| 0 | Actinic keratoses (akiec) | 66 | 34.4% | 17.5% | -16.9% |
| 1 | Basal cell carcinoma (bcc) | 103 | 45.9% | 34.3% | -11.6% |
| 2 | Benign keratosis (bkl) | 220 | 42.4% | 0.0% | -42.4% |
| 3 | Dermatofibroma (df) | 23 | 11.7% | 0.0% | -11.7% |
| 4 | Melanoma (mel) | 223 | 42.9% | 23.9% | -19.0% |
| 5 | Melanocytic nevi (nv) | 1341 | 81.4% | 0.0% | -81.4% |
| 6 | Vascular lesions (vasc) | 29 | 49.5% | 0.0% | -49.5% |

### Analysis of Federated Degradation

1. **Severe Client Drift Under Non-IID Skew**:
   - In standard FedAvg, clients perform multiple local SGD steps (3 epochs) on highly biased data distributions.
   - When local data lacks several classes entirely (e.g. Client 4 having 0 `mel`, Client 8 having 0 `df`/`bcc`), local model updates move in conflicting directions in the parameter space.
   - Naive coordinate-wise averaging of these divergent weights leads to catastrophic cancellation in feature representations and classifier head logits.

2. **Collapse on Majority & Zeroed Minority Predictions**:
   - Four out of seven classes (`bkl`, `df`, `nv`, `vasc`) collapsed to 0.0% F1.
   - Because `nv` was concentrated almost entirely in Client 8 (2,738 / 4,693 global samples), client sampling without robust aggregation caused the global model to alternate between predicting `mel` and `bcc`/`akiec`, completely losing calibrated boundaries for `nv` and rare classes.

3. **Motivation for Robust Aggregation & Proximal Regularization**:
   - This dramatic performance cliff (64.64% -> 14.51%) confirms the central thesis of the FedDerm research agenda: standard FedAvg is inadequate for medical imaging with high class imbalance and hospital-level non-IID distribution.
   - This provides the necessary baseline for subsequent evaluations of proximal regularization (FedProx $\mu$-term), control variates (SCAFFOLD), and pretrained feature backbones.
