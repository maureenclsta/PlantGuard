"""Build the augmentation comparison from the completed experiment logs."""

from __future__ import annotations

import json
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rows = []
    curves = {}
    for run_name, augmentation in (("loq_no_aug", "off"), ("loq_aug", "on")):
        run_dir = ROOT / "results" / run_name
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        log = pd.read_csv(run_dir / "training_log.csv")
        curves[run_name] = log
        best = log.loc[log["val_macro_f1"].idxmax()]
        rows.append({
            "experiment": run_name,
            "augmentation": augmentation,
            "seed": 42,
            "epochs_completed": int(summary["epochs_completed"]),
            "best_epoch": int(summary["best_epoch"]),
            "best_validation_macro_f1": float(summary["best_val_macro_f1"]),
            "validation_accuracy_at_best_epoch": float(best["val_accuracy"]),
            "validation_loss_at_best_epoch": float(best["val_loss"]),
            "training_plus_validation_seconds": float(summary["training_seconds_including_validation"]),
            "test_metrics": "Not evaluated for this run; test used once for validation-selected run only",
        })
    output = ROOT / "results"
    pd.DataFrame(rows).to_csv(output / "augmentation_comparison.csv", index=False)

    # Keep the project-wide result ledger explicit about validation vs test metrics.
    results_path = output / "experiment_results.csv"
    fields = ["experiment_name", "model", "augmentation", "hyperparameters", "accuracy",
              "macro_precision", "macro_recall", "macro_f1", "training_time_seconds",
              "inference_time_ms_per_image", "model_size_mb", "seed", "evaluation_split"]
    existing = []
    if results_path.exists():
        with results_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            existing = list(reader)
    validation_names = {f"baseline_{name}_validation" for name in ("loq_no_aug", "loq_aug")}
    retained = []
    for row in existing:
        row.setdefault("evaluation_split", "test")
        if row.get("experiment_name") not in validation_names:
            retained.append({key: row.get(key, "") for key in fields})
    for result in rows:
        config = json.loads((ROOT / "results" / result["experiment"] / "config.json").read_text(encoding="utf-8"))
        retained.append({
            "experiment_name": f"baseline_{result['experiment']}_validation",
            "model": "BaselineCNN",
            "augmentation": result["augmentation"],
            "hyperparameters": json.dumps({key: config[key] for key in
                ("epochs", "batch_size", "lr", "optimizer", "weight_decay", "imbalance_handling")}, sort_keys=True),
            "accuracy": result["validation_accuracy_at_best_epoch"],
            "macro_precision": "",
            "macro_recall": "",
            "macro_f1": result["best_validation_macro_f1"],
            "training_time_seconds": result["training_plus_validation_seconds"],
            "inference_time_ms_per_image": "",
            "model_size_mb": config["model_size_mb"],
            "seed": 42,
            "evaluation_split": "validation",
        })
    with results_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(retained)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for run_name, label in (("loq_no_aug", "No augmentation"), ("loq_aug", "Training augmentation")):
        log = curves[run_name]
        axes[0].plot(log["epoch"], log["val_macro_f1"], marker="o", label=label)
        axes[1].plot(log["epoch"], log["val_loss"], marker="o", label=label)
    axes[0].set(title="Validation macro F1", xlabel="Epoch", ylabel="Macro F1")
    axes[1].set(title="Validation loss", xlabel="Epoch", ylabel="Loss")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Augmentation comparison (one seed; validation set)")
    fig.tight_layout()
    fig.savefig(output / "figures" / "augmentation_comparison.png", dpi=180)
    plt.close(fig)

    best_no = rows[0]["best_validation_macro_f1"]
    best_aug = rows[1]["best_validation_macro_f1"]
    delta = best_aug - best_no
    notes = f"""# Augmentation experiment notes

Both runs used the same 41-class mapping, fixed split, BaselineCNN, seed 42, five epochs, batch size 16, AdamW (lr 0.001, weight decay 0.0001), inverse-frequency class weights, and RTX 4050 CUDA. Training augmentation was the intended variable.

| Run | Best validation macro F1 | Best epoch | Train + validation time |
|---|---:|---:|---:|
| No augmentation | {best_no:.4f} | {rows[0]['best_epoch']} | {rows[0]['training_plus_validation_seconds']:.1f} s |
| Training augmentation | {best_aug:.4f} | {rows[1]['best_epoch']} | {rows[1]['training_plus_validation_seconds']:.1f} s |

The observed validation difference (augmentation minus no augmentation) was {delta:+.4f} macro F1 in this single seed and five-epoch run. This is descriptive, not evidence that augmentation generally helps or hurts. Both runs reached their best validation score at the final epoch and were still improving, so the epoch limit did not establish convergence. No additional training was started.

Test-set evaluation was performed once, after choosing the no-augmentation checkpoint by validation macro F1. Its metrics are recorded separately in `results/test_evaluation/metrics.json`; they are not a second augmentation comparison because the augmented model was not selected for test evaluation.
"""
    (output / "augmentation_notes.md").write_text(notes, encoding="utf-8")


if __name__ == "__main__":
    main()
