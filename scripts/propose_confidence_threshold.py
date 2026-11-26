"""Propose a UI confidence warning threshold using validation predictions only."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.plantguard.data import ArchiveImageDataset, make_loader, make_transform
from src.plantguard.models import BaselineCNN

ARCHIVE = ROOT / "data/raw/Custom Dataset for Plant Species and Disease Detec/Ranveer_Realistic_dataset_2000_1126.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-accuracy", type=float, default=0.80,
                        help="Minimum observed accuracy among accepted validation predictions.")
    parser.add_argument("--minimum-coverage", type=float, default=0.10,
                        help="Require at least this fraction of validation images to remain accepted.")
    parser.add_argument("--minimum-support", type=int, default=30,
                        help="Require this many accepted validation images for a candidate threshold.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if not 0 < args.target_accuracy <= 1 or not 0 <= args.minimum_coverage <= 1:
        parser.error("target-accuracy must be in (0, 1] and minimum-coverage in [0, 1].")

    run_name = "loq_no_aug"  # Chosen beforehand by validation macro F1.
    checkpoint_path = ROOT / "results" / run_name / "best_checkpoint.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing validation-selected checkpoint: {checkpoint_path}")
    mapping = json.loads((ROOT / "models/class_mapping.json").read_text(encoding="utf-8"))
    class_to_idx = mapping["class_to_idx"]
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = BaselineCNN(len(class_to_idx)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    data = ArchiveImageDataset(ROOT / "dataset/splits/val.csv", ARCHIVE,
                               class_to_idx, make_transform(training=False))
    loader = make_loader(data, batch_size=16, training=False, num_workers=0)
    records = []
    with torch.inference_mode():
        for images, labels in loader:
            probabilities = model(images.to(device)).softmax(dim=1).cpu()
            confidence, prediction = probabilities.max(dim=1)
            for y, pred, conf in zip(labels.tolist(), prediction.tolist(), confidence.tolist()):
                records.append((int(y), int(pred), float(conf), int(y == pred)))

    candidates = sorted({row[2] for row in records})
    candidates.append(1.000001)  # permits a valid no-accepted-predictions row in the analysis
    rows = []
    n = len(records)
    for threshold in candidates:
        accepted = [row for row in records if row[2] >= threshold]
        if not accepted:
            continue
        rows.append({"threshold": threshold, "accepted": len(accepted),
                     "coverage": len(accepted) / n,
                     "accuracy_among_accepted": sum(row[3] for row in accepted) / len(accepted)})
    out = ROOT / "results" / "validation_confidence_analysis.csv"
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["threshold", "accepted", "coverage", "accuracy_among_accepted"])
        writer.writeheader()
        writer.writerows(rows)
    predictions_path = ROOT / "results" / "validation_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        names = {index: name for name, index in class_to_idx.items()}
        writer.writerow(["true_class", "predicted_class", "confidence", "correct"])
        for y, pred, conf, correct in records:
            writer.writerow([names[y], names[pred], conf, correct])

    # Bin confidence to make the correctness trend visible without implying calibration.
    bins = []
    for lower in [i / 10 for i in range(10)]:
        selected_bin = [row for row in records if lower <= row[2] < lower + 0.1]
        if selected_bin:
            bins.append((lower + 0.05, sum(row[2] for row in selected_bin) / len(selected_bin),
                         sum(row[3] for row in selected_bin) / len(selected_bin), len(selected_bin)))

    viable = [row for row in rows if row["accepted"] >= args.minimum_support
              and row["coverage"] >= args.minimum_coverage
              and row["accuracy_among_accepted"] >= args.target_accuracy]
    selected = min(viable, key=lambda row: row["threshold"]) if viable else None
    config = {
        "threshold": selected["threshold"] if selected else None,
        "status": "proposed" if selected else "no_threshold_met_criteria",
        "source": "validation_predictions.csv",
        "selected_run": run_name,
        "selection_rule": "lowest confidence cutoff with observed accepted accuracy >= target, minimum coverage, and minimum support",
        "target_accuracy": args.target_accuracy,
        "minimum_coverage": args.minimum_coverage,
        "minimum_support": args.minimum_support,
        "validation_images": n,
        "accepted_validation_images": selected["accepted"] if selected else None,
        "validation_coverage": selected["coverage"] if selected else None,
        "validation_accuracy_among_accepted": selected["accuracy_among_accepted"] if selected else None,
        "warning_text": "Unable to confidently identify this image. Please upload a clearer image.",
        "caveat": "A validation-derived threshold is a UI heuristic, not a calibrated probability or guarantee of correctness.",
    }
    (ROOT / "frontend" / "confidence_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if bins:
        axes[0].plot([b[1] for b in bins], [b[2] for b in bins], marker="o", label="Observed accuracy")
        axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration reference")
        for mean_conf, accuracy, _, count in bins:
            axes[0].annotate(str(count), (mean_conf, accuracy), fontsize=7)
    axes[0].set(title="Validation confidence vs correctness", xlabel="Mean top confidence", ylabel="Accuracy")
    axes[0].legend()
    if rows:
        axes[1].plot([row["threshold"] for row in rows], [row["accuracy_among_accepted"] for row in rows], label="Accepted accuracy")
        axes[1].plot([row["threshold"] for row in rows], [row["coverage"] for row in rows], label="Coverage")
        if selected:
            axes[1].axvline(selected["threshold"], linestyle="--", color="black", label="Proposed threshold")
    axes[1].set(title="Threshold trade-off (validation)", xlabel="Minimum confidence", ylabel="Rate")
    axes[1].legend()
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ROOT / "results/figures/validation_confidence_threshold.png", dpi=180)
    plt.close(fig)
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
