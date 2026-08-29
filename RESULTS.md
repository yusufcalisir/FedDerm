# FedDerm Experimental Results

## Centralized Baseline (DermaMNIST)

### Experiment Setup

- **Dataset**: DermaMNIST (MedMNIST v2), 7-class dermatoscopic skin lesion classification derived from HAM10000.
  - Image resolution: 28x28 (3 channels RGB).
  - Sample counts: 7,007 train, 1,003 val, 2,005 test (official splits).
- **Architecture**: `MiniCNN` with `GroupNorm` (num_groups=8 across all 4 convolutional blocks; 468,967 trainable parameters, 0 running statistic buffers).
- **Optimization**: Adam optimizer with initial learning rate 1e-3, weight decay 1e-4, and Cosine Annealing scheduler across 30 epochs. Batch size 64.
- **Loss function**: CrossEntropyLoss with tempered inverse-frequency class weights (exponent=0.3) to balance majority (`nv`) and minority class gradient signals without over-penalizing majority representations.
- **Model Selection**: Best checkpoint selected by validation Macro F1 score to prevent majority-class selection bias.
- **Hardware & Runtime**: Local CPU execution, total training time: 8.9 minutes (535.5 seconds).

### Evaluation Metrics (Official Test Split)

| Metric | Score |
|---|---|
| Test Accuracy | 74.11% |
| Balanced Accuracy | 47.74% |
| Macro F1 | 46.34% |
| Weighted F1 | 73.05% |
| Best Validation Macro F1 | 45.91% |

### Per-Class Performance Breakdown

| Class Index | Diagnosis | Support | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| 0 | Actinic keratoses / intraepithelial carcinoma (akiec) | 66 | 0.41 | 0.36 | 0.387 |
| 1 | Basal cell carcinoma (bcc) | 103 | 0.46 | 0.61 | 0.525 |
| 2 | Benign keratosis-like lesions (bkl) | 220 | 0.44 | 0.42 | 0.432 |
| 3 | Dermatofibroma (df) | 23 | 0.00 | 0.00 | 0.000 |
| 4 | Melanoma (mel) | 223 | 0.41 | 0.31 | 0.355 |
| 5 | Melanocytic nevi (nv) | 1341 | 0.87 | 0.91 | 0.889 |
| 6 | Vascular lesions (vasc) | 29 | 0.60 | 0.72 | 0.656 |

---

## Federated Baseline: FedAvg with Non-IID Dirichlet Partitioning

### Experiment Setup

- **Dataset**: DermaMNIST (28x28, 7 classes), 7,007 training samples partitioned across 10 simulated hospitals.
- **Partitioning Strategy**: Dirichlet distribution over class proportions ($\alpha = 0.3$, seed 42).
- **Federated Algorithm**: FedAvg via Flower simulation backend (10 clients, 5 sampled/round, 20 rounds, 3 local epochs).
- **Hardware & Runtime**: Local CPU execution, total simulation time: 15.5 minutes (928.1 seconds).

### Baseline Metrics

| Metric | Centralized Baseline | Federated FedAvg ($\alpha=0.3$) | Degradation ($\Delta$) |
|---|---|---|---|
| **Test Accuracy** | **74.11%** | **68.28%** | **-5.83%** |
| **Balanced Accuracy** | **47.74%** | **20.65%** | **-27.09%** |
| **Macro F1** | **46.34%** | **17.81%** | **-28.53%** |
| **Weighted F1** | **73.05%** | **57.96%** | **-15.09%** |
| **Best Val Macro F1** | **45.91%** | **17.24%** | **-28.67%** |

---

## Heterogeneity-Robust Aggregation: FedProx

### Proximal Parameter ($\mu$) Sweep Comparison

| Proximal Penalty ($\mu$) | Test Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 | Best Val Macro F1 | Runtime |
|---|---|---|---|---|---|---|
| $\mu = 0.001$ (weak) | 54.31% | **25.42%** | 19.98% | 56.87% | 19.34% | 5.3 min |
| **$\mu = 0.01$ (default, best)** | **67.73%** | 22.03% | **20.36%** | **63.58%** | **20.44%** | 13.1 min |
| $\mu = 0.1$ (strong) | 59.85% | 20.47% | 15.87% | 58.52% | 16.02% | 14.9 min |

---

## Directional Drift Correction: SCAFFOLD

### Four-Way Method Comparison: Centralized vs. FedAvg vs. FedProx vs. SCAFFOLD

| Metric | Centralized Baseline | FedAvg ($\alpha=0.3$) | FedProx ($\mu=0.01$) | SCAFFOLD | Best Federated Method |
|---|---|---|---|---|---|
| **Overall Test Accuracy** | **74.11%** | 68.28% | 67.73% | **68.93%** | **SCAFFOLD (+0.65% vs FedAvg)** |
| **Balanced Accuracy** | **47.74%** | 20.65% | **22.03%** | 19.93% | **FedProx (+1.38% vs FedAvg)** |
| **Macro F1** | **46.34%** | 17.81% | **20.36%** | 17.27% | **FedProx (+2.55% vs FedAvg)** |
| **Weighted F1** | **73.05%** | 57.96% | **63.58%** | 60.66% | **FedProx (+5.62% vs FedAvg)** |
| **Best Val Macro F1** | **45.91%** | 17.24% | **20.44%** | 17.94% | **FedProx (+3.20% vs FedAvg)** |

---

## Differential Privacy via DP-SGD on FedProx (DermaMNIST)

### Experiment Setup

- **Base Aggregation**: FedProx ($\mu=0.01$, Dirichlet $\alpha=0.3$, partition seed 42, 10 clients, 5 sampled/round, 20 rounds, 3 local epochs).
- **DP Mechanism**: Client-side DP-SGD via Opacus `PrivacyEngine`.
  - Gradient clipping: $L_2$ norm threshold $C = 1.0$.
  - Target delta: $\delta = 10^{-5}$.
  - Privacy accounting: Rényi Differential Privacy (RDP) accountant computing server-level subsampled record DP budget $\epsilon$ and local client DP budget.
- **Evaluation Strategy**: Multi-seed evaluation across 3 independent random seeds (42, 43, 44) controlling model weight initialization and Opacus Gaussian noise sampling, while maintaining a fixed Dirichlet client data partition ($\alpha=0.3$).
- **Noise Multiplier Sweep**: Evaluated $\sigma \in [0.3, 0.5, 1.0, 2.0]$.
- **Hardware & Runtime**: CPU execution across all seeds and noise multipliers.

---

### Privacy-Utility Tradeoff Summary Table (3-Seed Aggregated: Mean $\pm$ Std)

> [!NOTE]
> **Superseded Single-Seed Baseline**: The original single-seed exploratory run ($\sigma \in [0.3, 0.5, 1.0, 2.0]$, run once with seed 42) produced high point-estimate variance. The numbers below supersede the single-seed results with rigorous sample mean $\pm$ standard deviation statistics across 3 independent seeds. Server and client $\epsilon$ values are analytically deterministic and identical across all random seeds for a given noise multiplier.

| Noise Multiplier ($\sigma$) | Server $\epsilon$ ($\delta=10^{-5}$) | Client $\epsilon$ | Test Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 | Privacy Regime |
|---|---|---|---|---|---|---|---|
| **0.00 (Non-DP Reference)** | $\infty$ | $\infty$ | **67.73%** | **22.03%** | **20.36%** | **63.58%** | No Privacy |
| **0.30** | 40.41 | 362.23 | 55.81 $\pm$ 9.59% | 18.82 $\pm$ 4.04% | 14.31 $\pm$ 2.92% | 52.36 $\pm$ 1.56% | Very Weak Privacy |
| **0.50** | 8.08 | 93.19 | 58.97 $\pm$ 10.41% | 16.16 $\pm$ 3.72% | 12.48 $\pm$ 1.23% | 52.34 $\pm$ 2.56% | Moderate Privacy |
| **1.00** | 1.06 | 19.06 | 63.44 $\pm$ 5.92% | 17.17 $\pm$ 2.51% | 14.30 $\pm$ 2.63% | 55.01 $\pm$ 2.29% | **Standard Strict Privacy ($\epsilon \le 1.06$)** |
| **2.00** | 0.25 | 6.36 | 59.83 $\pm$ 7.19% | 18.71 $\pm$ 4.40% | 15.22 $\pm$ 3.71% | 54.33 $\pm$ 0.64% | **Very Strict Privacy ($\epsilon \le 0.25$)** |

---

### Per-Class F1 Score Breakdown Under Increasing Privacy Noise (Mean $\pm$ Std)

| Class Index | Diagnosis | Support | Non-DP FedProx | $\sigma=0.3$ ($\epsilon=40.4$) | $\sigma=0.5$ ($\epsilon=8.08$) | $\sigma=1.0$ ($\epsilon=1.06$) | $\sigma=2.0$ ($\epsilon=0.25$) |
|---|---|---|---|---|---|---|---|
| 0 | Actinic keratoses (`akiec`) | 66 | 0.0% | 0.0 $\pm$ 0.0% | 0.0 $\pm$ 0.0% | 0.0 $\pm$ 0.0% | 0.0 $\pm$ 0.0% |
| 1 | Basal cell carcinoma (`bcc`) | 103 | 0.0% | 0.0 $\pm$ 0.0% | 0.0 $\pm$ 0.0% | 0.0 $\pm$ 0.0% | 0.0 $\pm$ 0.0% |
| 2 | Benign keratosis (`bkl`) | 220 | **35.5%** | 20.2 $\pm$ 17.6% | 10.9 $\pm$ 14.6% | 12.1 $\pm$ 20.9% | 20.0 $\pm$ 17.4% |
| 3 | Dermatofibroma (`df`) | 23 | 0.0% | 0.0 $\pm$ 0.0% | 0.0 $\pm$ 0.0% | 0.0 $\pm$ 0.0% | 0.0 $\pm$ 0.0% |
| 4 | Melanoma (`mel`) | 223 | **21.3%** | 6.1 $\pm$ 10.5% | 0.0 $\pm$ 0.0% | 9.3 $\pm$ 16.2% | 10.2 $\pm$ 17.7% |
| 5 | Melanocytic nevi (`nv`) | 1341 | **85.7%** | 74.0 $\pm$ 5.4% | 76.5 $\pm$ 6.2% | 78.7 $\pm$ 2.6% | 76.2 $\pm$ 4.3% |
| 6 | Vascular lesions (`vasc`) | 29 | 0.0% | 0.0 $\pm$ 0.0% | 0.0 $\pm$ 0.0% | 0.0 $\pm$ 0.0% | 0.0 $\pm$ 0.0% |

---

### In-Depth Scientific Discussion: Variance, Monotonicity, and the Collapse Point

1. **Monotonicity and the Majority-Class Accuracy Paradox**:
   - The privacy-utility curve remains non-monotonic across noise levels even after 3-seed aggregation. Raw test accuracy rises from 55.81% at $\sigma=0.3$ to 63.44% at $\sigma=1.0$, while Macro F1 hovers in the narrow 12.48%--15.22% band.
   - This non-monotonicity is directly explained by class imbalance in DermaMNIST (class 5, `nv`, accounts for 66.88% of test samples). Under heavier DP perturbation or uncooperative initializations, the network collapses into a trivial majority-class predictor (predicting `nv` on 100% of samples), producing exactly 66.88% raw accuracy, 14.29% balanced accuracy (1/7), and 11.45% Macro F1 (with `nv` F1 = 80.2% and all other classes at 0.0%).
   - Conversely, when a seed partially learns minority decision boundaries (`bkl` or `mel`), false positives on the majority class reduce raw test accuracy to ~50% while improving balanced accuracy and Macro F1. Thus, higher raw accuracy under DP-SGD often signals class collapse rather than superior generalization.

2. **Run-to-Run Variance Analysis**:
   - **Highest Variance Regime**: $\sigma=0.5$ exhibited the highest run-to-run variance in test accuracy ($\pm 10.41\%$, ranging from 47.18% in seed 42 to 66.88% in seed 44). $\sigma=0.3$ also showed high variance ($\pm 9.59\%$).
   - $\sigma=2.0$ showed the highest variance in Macro F1 ($\pm 3.71\%$) and Balanced Accuracy ($\pm 4.40\%$).
   - **Stability Implications**: With small-sample non-IID shards ($N \approx 700$ samples per client) and 468,967 parameters, DP-SGD is highly sensitive to the initial gradient trajectory. If early noisy steps disrupt the fragile decision boundaries of minority classes, the model locks into the majority-class regime for the remainder of the 20 rounds.

3. **Reconfirmation of the Small-Sample Full-Model Collapse Point**:
   - Across all 4 DP noise levels and all 3 random seeds, **4 out of 7 dermatological classes (`akiec`, `bcc`, `df`, `vasc`) completely collapsed to 0.0% F1 in every single run** (0.00 $\pm$ 0.00%).
   - The remaining minority classes (`bkl` and `mel`) survived only intermittently, suffering substantial degradation compared to Non-DP FedProx (`bkl`: 20.2% vs 35.5%; `mel`: 6.1%--10.2% vs 21.3%).
   - This empirical evidence confirms that full-model training from scratch cannot simultaneously satisfy strict differential privacy ($\epsilon \le 1.06$) and clinically meaningful multi-class skin lesion discrimination.

4. **Empirical Motivation for Parameter-Efficient DP**:
   - Because DP noise magnitude scales with $\sqrt{d}$ (where $d$ is parameter count), reducing the number of DP-perturbed parameters by 1--2 orders of magnitude while utilizing strong frozen representations is essential.
   - This motivates the parameter-efficient adaptation pipeline: **DP-LoRA on a frozen pretrained Vision Transformer backbone**.
