import json
from pathlib import Path

with open("results/federated_dp_fedprox_multiseed/multiseed_summary.json") as f:
    data = json.load(f)

for s in data:
    nm = s["noise_multiplier"]
    se = s["server_epsilon"]
    ce = s["client_epsilon"]
    print(f"=== Noise Multiplier: {nm} (Server eps: {se:.2f}, Client eps: {ce:.2f}) ===")
    print(f"Test Acc:     {s['accuracy_mean']:.2f} +- {s['accuracy_std']:.2f}%")
    print(f"Bal Acc:      {s['balanced_accuracy_mean']:.2f} +- {s['balanced_accuracy_std']:.2f}%")
    print(f"Macro F1:     {s['macro_f1_mean']:.2f} +- {s['macro_f1_std']:.2f}%")
    print(f"Weighted F1:  {s['weighted_f1_mean']:.2f} +- {s['weighted_f1_std']:.2f}%")
    print(f"Best Val mF1: {s['best_val_macro_f1_mean']:.2f} +- {s['best_val_macro_f1_std']:.2f}%")
    print("Per-class F1 mean +- std (%):")
    for c, (m, sd) in enumerate(zip(s["per_class_f1_mean"], s["per_class_f1_std"])):
        print(f"  Class {c}: {m:.2f} +- {sd:.2f}%")
    print("Individual runs:")
    for run in s["individual_runs"]:
        pc = [round(x * 100, 1) for x in run.get("per_class_f1", [])]
        print(
            f"  Seed {run.get('seed')}: Acc={run['accuracy']*100:.2f}%, "
            f"BalAcc={run['balanced_accuracy']*100:.2f}%, "
            f"mF1={run['macro_f1']*100:.2f}%, "
            f"wF1={run['weighted_f1']*100:.2f}%, per_class={pc}"
        )
    print()
