# GPU Execution Guide for DP-LoRA Privacy-Utility Sweep

This guide outlines how to execute the federated DP-LoRA (Vision Transformer ViT-B/16 + LoRA adapters) experiments on a free cloud GPU environment (**Kaggle Notebooks** or **Google Colab**) and reintegrate the experimental results into the local repository.

---

## Benchmark & Expected Runtimes

| Setup | Machine / Hardware | 1 Non-DP Run | Full 12-Run Multi-Seed DP Sweep |
| :--- | :--- | :--- | :--- |
| **Local Baseline** | 8-Core CPU (No CUDA) | ~17.2 hours | ~221.1 hours (~9.2 days) |
| **Cloud GPU Tier** | NVIDIA T4 / P100 (16 GB) | **~4 to 6 minutes** | **~1.0 to 1.5 hours** |

---

## 1. Kaggle Notebooks (Recommended for Unattended Runs)

Kaggle Notebooks support asynchronous background execution via "Save & Run All (Commit)". You do not need to keep your browser window or internet connection open while the sweep runs.

### Step 1: Create a Kaggle Notebook
1. Sign in to [Kaggle](https://www.kaggle.com).
2. Click **Create** -> **New Notebook** (or go to [kaggle.com/code](https://www.kaggle.com/code)).
3. In the notebook editor, click `File` -> `Import Notebook` -> Upload [`notebooks/gpu_dp_lora_sweep.ipynb`](file:///d:/FedDerm/notebooks/gpu_dp_lora_sweep.ipynb).

### Step 2: Configure Notebook Settings (Right Sidebar)
In the right-hand panel under **Notebook settings**:
1. **Accelerator**: Select **GPU T4 x2** or **GPU P100**.
2. **Internet**: Toggle the switch to **On** (required for pip package installation and downloading pretrained ViT weights).
3. **Environment**: Standard (default).

### Step 3: Run Top-to-Bottom in Background
1. In the top-right corner, click **Save Version**.
2. Set **Version Type** to **Save & Run All (Commit)**.
3. Click **Save**.
4. You can now close your browser tab. Kaggle will spin up a dedicated GPU container, execute all cells sequentially, generate all metrics and plots, and save the output.

### Step 4: Download Outputs
1. Once the commit finishes (~1 to 1.5 hours), open your notebook on Kaggle.
2. Go to the **Output** tab of the committed version.
3. Download `dp_lora_fedprox_results.zip`.

---

## 2. Google Colab (Interactive Execution Alternative)

Google Colab provides free access to NVIDIA T4 GPUs in an interactive notebook interface.

### Step 1: Open Notebook in Colab
1. Navigate to [Google Colab](https://colab.research.google.com).
2. Click **Upload** and select [`notebooks/gpu_dp_lora_sweep.ipynb`](file:///d:/FedDerm/notebooks/gpu_dp_lora_sweep.ipynb).

### Step 2: Enable GPU Runtime
1. In the top menu, go to **Runtime** -> **Change runtime type**.
2. Set **Hardware accelerator** to **T4 GPU**.
3. Click **Save**.

### Step 3: Run All Cells
1. In the top menu, click **Runtime** -> **Run all**.
2. Keep the browser tab active while the cells execute (~1 to 1.5 hours).
3. The final cell will automatically trigger a browser download for `dp_lora_fedprox_results.zip`.

---

## 3. Local Results Reintegration

Once the cloud GPU execution is complete and you have downloaded `dp_lora_fedprox_results.zip`:

### Step 1: Extract Archive
Extract `dp_lora_fedprox_results.zip` into the `results/` folder of your local FedDerm workspace.

The resulting directory structure will be:
```text
results/
└── dp_lora_fedprox/
    ├── non_dp_sanity/
    │   ├── best_model.pt
    │   ├── history.json
    │   ├── partition_report.json
    │   └── test_metrics.json
    ├── sigma_0.3/
    │   ├── seed_42/
    │   ├── seed_43/
    │   └── seed_44/
    ├── sigma_0.5/
    │   ├── seed_42/
    │   ├── seed_43/
    │   └── seed_44/
    ├── sigma_1.0/
    │   ├── seed_42/
    │   ├── seed_43/
    │   └── seed_44/
    ├── sigma_2.0/
    │   ├── seed_42/
    │   ├── seed_43/
    │   └── seed_44/
    ├── multiseed_summary.json
    ├── multiseed_summary.csv
    └── privacy_utility_tradeoff_multiseed.png
```

### Step 2: Verify Results
Run the verification check to inspect the imported summary:
```powershell
.venv\Scripts\python -c "import pandas as pd; print(pd.read_csv('results/dp_lora_fedprox/multiseed_summary.csv'))"
```
