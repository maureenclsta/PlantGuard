"""Evaluate the validation-selected baseline once on the held-out test split.

Run this on the training machine where the best checkpoints are present. The
script refuses to overwrite a prior test evaluation so the test set is not
silently reused during model selection.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.plantguard.data import ArchiveImageDataset, make_loader, make_transform
from src.plantguard.models import BaselineCNN

ARCHIVE = ROOT / "data/raw/Custom Dataset for Plant Species and Disease Detec/Ranveer_Realistic_dataset_2000_1126.zip"
RUN_NAMES = ("loq_no_aug", "loq_aug")
OUTPUT_DIR = ROOT / "results/test_evaluation"


def choose_by_validation():
    """Choose a run using validation macro F1 only; never inspect test data here."""
    runs = []
    for name in RUN_NAMES:
        run_dir = ROOT / "results" / name
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing run summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        runs.append((float(summary["best_val_macro_f1"]), name, run_dir, summary))
    _, name, run_dir, summary = max(runs, key=lambda row: row[0])
    return name, run_dir, summary


def main():
    metrics_path = OUTPUT_DIR / "metrics.json"
    if metrics_path.exists():
        raise FileExistsError(
            f"Test results already exist at {metrics_path}; refusing to evaluate the test set again."
        )

    run_name, run_dir, run_summary = choose_by_validation()
    checkpoint_path = run_dir / "best_checkpoint.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint missing: {checkpoint_path}. It is intentionally excluded from Git; "
            "run this on the training laptop where it was saved."
        )

    mapping_doc = json.loads((ROOT / "models/class_mapping.json").read_text(encoding="utf-8"))
    class_to_idx = mapping_doc["class_to_idx"]
    idx_to_class = {int(index): name for index, name in mapping_doc["idx_to_class"].items()}
    class_names = [idx_to_class[index] for index in range(len(idx_to_class))]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(8, torch.get_num_threads()))

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = BaselineCNN(num_classes=len(class_to_idx)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_data = ArchiveImageDataset(
        ROOT / "dataset/splits/test.csv", ARCHIVE, class_to_idx,
        transform=make_transform(training=False),
    )
    test_loader = make_loader(test_data, batch_size=16, training=False, num_workers=0)

    actual, predicted = [], []
    model_forward_seconds = 0.0
    pipeline_started = time.perf_counter()
    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_started = time.perf_counter()
            logits = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            model_forward_seconds += time.perf_counter() - forward_started
            predicted.extend(logits.argmax(1).cpu().tolist())
            actual.extend(labels.tolist())
    pipeline_seconds = time.perf_counter() - pipeline_started

    accuracy = accuracy_score(actual, predicted)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        actual, predicted, labels=list(range(len(class_names))), average="macro", zero_division=0
    )
    report = classification_report(
        actual, predicted, labels=list(range(len(class_names))), target_names=class_names,
        output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(actual, predicted, labels=list(range(len(class_names))))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "classification_report.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["class_or_average", "precision", "recall", "f1_score", "support"])
        for label in class_names + ["macro avg", "weighted avg"]:
            row = report[label]
            writer.writerow([label, row["precision"], row["recall"], row["f1-score"], row["support"]])

    fig, ax = plt.subplots(figsize=(18, 16))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           xlabel="Predicted class", ylabel="True class", title="Baseline test confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=6)
    plt.setp(ax.get_yticklabels(), fontsize=6)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    params = sum(parameter.numel() for parameter in model.parameters())
    model_size_mb = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()) / (1024 ** 2)
    count = len(actual)
    metrics = {
        "selected_run": run_name,
        "selection_basis": "validation macro F1",
        "validation_macro_f1": run_summary["best_val_macro_f1"],
        "test_images": count,
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "test_model_forward_seconds": model_forward_seconds,
        "model_forward_ms_per_image": model_forward_seconds * 1000 / count,
        "test_pipeline_seconds_including_load_and_preprocessing": pipeline_seconds,
        "pipeline_ms_per_image": pipeline_seconds * 1000 / count,
        "model_size_mb": model_size_mb,
        "parameter_count": params,
        "device": str(device),
        "test_evaluated_once": True,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    experiment_csv = ROOT / "results/experiment_results.csv"
    fields = ["experiment_name", "model", "augmentation", "hyperparameters", "accuracy",
              "macro_precision", "macro_recall", "macro_f1", "training_time_seconds",
              "inference_time_ms_per_image", "model_size_mb", "seed"]
    exists = experiment_csv.exists()
    with experiment_csv.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({
            "experiment_name": f"baseline_{run_name}_selected",
            "model": "BaselineCNN",
            "augmentation": "on" if run_name == "loq_aug" else "off",
            "hyperparameters": json.dumps({"epochs": run_summary["epochs_completed"],
                                           "batch_size": 16, "optimizer": "adamw",
                                           "lr": 0.001, "weight_decay": 0.0001,
                                           "imbalance_handling": "weights"}, sort_keys=True),
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "training_time_seconds": run_summary["training_seconds_including_validation"],
            "inference_time_ms_per_image": metrics["model_forward_ms_per_image"],
            "model_size_mb": model_size_mb,
            "seed": 42,
        })
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
