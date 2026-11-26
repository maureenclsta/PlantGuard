"""Propose a confidence warning cutoff from validation predictions only."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARCHIVE = ROOT / "data/raw/Custom Dataset for Plant Species and Disease Detec/Ranveer_Realistic_dataset_2000_1126.zip"
PREDICTIONS = ROOT / "results/validation_predictions.csv"


def infer_validation_predictions(device_name: str) -> list[dict]:
    """Run only the validation split; this never reads the test split."""
    import torch

    from src.plantguard.data import ArchiveImageDataset, make_loader, make_transform
    from src.plantguard.models import BaselineCNN

    run_name = "loq_no_aug"  # selected beforehand using validation macro F1
    checkpoint_path = ROOT / "results" / run_name / "best_checkpoint.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing validation-selected checkpoint: {checkpoint_path}")
    mapping = json.loads((ROOT / "models/class_mapping.json").read_text(encoding="utf-8"))
    class_to_idx = mapping["class_to_idx"]
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else
                          "cpu" if device_name == "auto" else device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = BaselineCNN(len(class_to_idx)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    data = ArchiveImageDataset(ROOT / "dataset/splits/val.csv", ARCHIVE,
                               class_to_idx, make_transform(training=False))
    loader = make_loader(data, batch_size=16, training=False, num_workers=0)
    idx_to_class = {index: name for name, index in class_to_idx.items()}
    predictions = []
    with torch.inference_mode():
        for images, labels in loader:
            probabilities = model(images.to(device)).softmax(dim=1).cpu()
            confidence, predicted = probabilities.max(dim=1)
            for truth, guess, score in zip(labels.tolist(), predicted.tolist(), confidence.tolist()):
                predictions.append({"true_class": idx_to_class[truth],
                                    "predicted_class": idx_to_class[guess],
                                    "confidence": float(score), "correct": int(truth == guess)})
    return predictions


def load_predictions(recompute: bool, device: str) -> list[dict]:
    if PREDICTIONS.is_file() and not recompute:
        with PREDICTIONS.open(encoding="utf-8-sig", newline="") as stream:
            return [{"true_class": row["true_class"], "predicted_class": row["predicted_class"],
                     "confidence": float(row["confidence"]), "correct": int(row["correct"])}
                    for row in csv.DictReader(stream)]
    predictions = infer_validation_predictions(device)
    with PREDICTIONS.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["true_class", "predicted_class", "confidence", "correct"])
        writer.writeheader()
        writer.writerows(predictions)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum-coverage", type=float, default=0.10,
                        help="Require at least this validation share to remain accepted.")
    parser.add_argument("--minimum-support", type=int, default=30,
                        help="Require at least this many accepted validation examples.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--recompute", action="store_true",
                        help="Re-run inference on validation instead of using its saved predictions.")
    args = parser.parse_args()
    if not 0 <= args.minimum_coverage <= 1 or args.minimum_support < 1:
        parser.error("minimum-coverage must be in [0, 1] and minimum-support must be positive.")

    predictions = load_predictions(args.recompute, args.device)
    if not predictions:
        raise ValueError("No validation predictions were found or generated.")
    total_correct = sum(row["correct"] for row in predictions)
    total_wrong = len(predictions) - total_correct
    rows = []
    for threshold in sorted({row["confidence"] for row in predictions}):
        accepted = [row for row in predictions if row["confidence"] >= threshold]
        correct = sum(row["correct"] for row in accepted)
        incorrect = len(accepted) - correct
        coverage = len(accepted) / len(predictions)
        correct_recall = correct / total_correct if total_correct else 0.0
        incorrect_accept_rate = incorrect / total_wrong if total_wrong else 0.0
        rows.append({
            "threshold": threshold,
            "accepted": len(accepted),
            "coverage": coverage,
            "accuracy_among_accepted": correct / len(accepted),
            "correct_prediction_recall": correct_recall,
            "incorrect_prediction_accept_rate": incorrect_accept_rate,
            "youden_j": correct_recall - incorrect_accept_rate,
        })
    eligible = [row for row in rows if row["accepted"] >= args.minimum_support
                and row["coverage"] >= args.minimum_coverage]
    selected = max(eligible, key=lambda row: (row["youden_j"], row["threshold"])) if eligible else None
    if selected and selected["youden_j"] <= 0:
        selected = None

    analysis_path = ROOT / "results/validation_confidence_analysis.csv"
    with analysis_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["threshold", "accepted", "coverage", "accuracy_among_accepted",
                                                    "correct_prediction_recall", "incorrect_prediction_accept_rate", "youden_j"])
        writer.writeheader()
        writer.writerows(rows)

    config = {
        "threshold": selected["threshold"] if selected else None,
        "status": "proposed" if selected else "no_threshold_met_criteria",
        "source": "validation_predictions.csv",
        "selected_run": "loq_no_aug",
        "selection_rule": "confidence cutoff maximizing Youden's J for distinguishing correct from incorrect validation predictions, subject to minimum coverage and support",
        "minimum_coverage": args.minimum_coverage,
        "minimum_support": args.minimum_support,
        "validation_images": len(predictions),
        "accepted_validation_images": selected["accepted"] if selected else None,
        "validation_coverage": selected["coverage"] if selected else None,
        "validation_accuracy_among_accepted": selected["accuracy_among_accepted"] if selected else None,
        "youden_j": selected["youden_j"] if selected else None,
        "warning_text": "Unable to confidently identify this image. Please upload a clearer image.",
        "caveat": "Validation-derived UI heuristic; not a calibrated probability or guarantee of correctness.",
    }
    (ROOT / "frontend/confidence_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    bins = []
    for lower in [i / 10 for i in range(10)]:
        part = [row for row in predictions if lower <= row["confidence"] < lower + 0.1]
        if part:
            bins.append((sum(r["confidence"] for r in part) / len(part),
                         sum(r["correct"] for r in part) / len(part), len(part)))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if bins:
        axes[0].plot([row[0] for row in bins], [row[1] for row in bins], marker="o", label="Observed accuracy")
        axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration reference")
        for confidence, accuracy, count in bins:
            axes[0].annotate(str(count), (confidence, accuracy), fontsize=7)
    axes[0].set(title="Validation confidence vs correctness", xlabel="Mean top confidence", ylabel="Accuracy")
    axes[0].legend()
    axes[1].plot([r["threshold"] for r in rows], [r["accuracy_among_accepted"] for r in rows], label="Accepted accuracy")
    axes[1].plot([r["threshold"] for r in rows], [r["coverage"] for r in rows], label="Coverage")
    axes[1].plot([r["threshold"] for r in rows], [r["youden_j"] for r in rows], label="Youden's J")
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
