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
3. **Hardware Assessment**: Training 30 epochs took ~12 minutes on CPU with a lightweight CNN (~469k params). For standard federated simulations (e.g. 10 clients, 100 rounds) and pretrained Vision Transformers (ViT-B/16 with LoRA), GPU acceleration (Google Colab / Kaggle or local CUDA) will be necessary to keep iteration cycles fast.
