"""Train the scratch CNN baseline with validation-only model selection."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.plantguard.data import (ArchiveImageDataset, class_weights, make_loader,
                                make_transform, make_weighted_sampler)
from src.plantguard.models import BaselineCNN

DEFAULT_ARCHIVE = ROOT / "data/raw/Custom Dataset for Plant Species and Disease Detec/Ranveer_Realistic_dataset_2000_1126.zip"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=("adamw", "adam", "sgd"), default="adamw")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--augmentation", choices=("on", "off"), default="on")
    parser.add_argument("--imbalance-handling", choices=("none", "weights", "sampler"), default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models/baseline")
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=min(8, torch.get_num_threads()))
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--mapping", type=Path, default=ROOT / "models/class_mapping.json")
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, actual, predicted = 0.0, [], []
    started = time.perf_counter()
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            actual.extend(labels.detach().cpu().tolist())
            predicted.extend(logits.argmax(1).detach().cpu().tolist())
    elapsed = time.perf_counter() - started
    return {
        "loss": total_loss / max(1, len(actual)),
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": f1_score(actual, predicted, average="macro", zero_division=0),
        "seconds": elapsed,
    }


def save_curves(history: pd.DataFrame, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, train_col, val_col, title in [
        (axes[0], "train_loss", "val_loss", "Loss"),
        (axes[1], "train_accuracy", "val_accuracy", "Accuracy"),
        (axes[2], None, "val_macro_f1", "Validation macro F1"),
    ]:
        if train_col:
            ax.plot(history.epoch, history[train_col], label="train")
        ax.plot(history.epoch, history[val_col], label="validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    seed_everything(args.seed)
    torch.set_num_threads(max(1, args.num_threads))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    mapping_doc = json.loads(args.mapping.read_text(encoding="utf-8"))
    class_to_idx = mapping_doc["class_to_idx"]
    num_classes = len(class_to_idx)
    train_csv, val_csv = ROOT / "dataset/splits/train.csv", ROOT / "dataset/splits/val.csv"
    train_transform = make_transform(training=args.augmentation == "on")
    eval_transform = make_transform(training=False)
    train_data = ArchiveImageDataset(train_csv, args.archive, class_to_idx, transform=train_transform)
    val_data = ArchiveImageDataset(val_csv, args.archive, class_to_idx, transform=eval_transform)
    sampler = None
    if args.imbalance_handling == "sampler":
        sampler = make_weighted_sampler([row["class_name"] for row in train_data.rows], class_to_idx)
    train_loader = make_loader(train_data, args.batch_size, training=True,
                               num_workers=args.num_workers, sampler=sampler)
    val_loader = make_loader(val_data, args.batch_size, training=False, num_workers=args.num_workers)

    model = BaselineCNN(num_classes).to(device)
    weights = None
    if args.imbalance_handling == "weights":
        weights = class_weights([row["class_name"] for row in train_data.rows], class_to_idx).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizers = {
        "adamw": torch.optim.AdamW,
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
    }
    optimizer_kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
    if args.optimizer == "sgd":
        optimizer_kwargs["momentum"] = 0.9
    optimizer = optimizers[args.optimizer](model.parameters(), **optimizer_kwargs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update({"device_resolved": str(device), "input_size": [224, 224],
                   "num_classes": num_classes, "class_to_idx": class_to_idx,
                   "parameter_count": sum(p.numel() for p in model.parameters()),
                   "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2),
                   "selection_metric": "validation macro F1"})
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")

    history, best_f1, best_epoch, stale = [], -1.0, None, 0
    overall_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, criterion, device, optimizer)
        val_stats = run_epoch(model, val_loader, criterion, device)
        row = {"epoch": epoch, "train_loss": train_stats["loss"],
               "train_accuracy": train_stats["accuracy"], "train_seconds": train_stats["seconds"],
               "val_loss": val_stats["loss"], "val_accuracy": val_stats["accuracy"],
               "val_macro_f1": val_stats["macro_f1"], "val_seconds": val_stats["seconds"]}
        history.append(row)
        pd.DataFrame(history).to_csv(args.output_dir / "training_log.csv", index=False)
        print(json.dumps(row), flush=True)
        if val_stats["macro_f1"] > best_f1:
            best_f1, best_epoch, stale = val_stats["macro_f1"], epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "val_macro_f1": best_f1, "class_to_idx": class_to_idx},
                       args.output_dir / "best_checkpoint.pt")
        else:
            stale += 1
        if stale >= args.patience:
            print(f"Early stopping after epoch {epoch}; best epoch={best_epoch}", flush=True)
            break
    total_seconds = time.perf_counter() - overall_started
    history_frame = pd.DataFrame(history)
    save_curves(history_frame, args.output_dir / "training_curves.png")
    summary = {"epochs_completed": len(history), "best_epoch": best_epoch,
               "best_val_macro_f1": best_f1, "training_seconds_including_validation": total_seconds,
               "mean_train_epoch_seconds": float(history_frame.train_seconds.mean()),
               "mean_validation_epoch_seconds": float(history_frame.val_seconds.mean()),
               "device": str(device), "test_set_evaluated": False}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
