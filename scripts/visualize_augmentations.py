"""Sample the real PyTorch training transform from the archive-backed dataset."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.plantguard.data import ArchiveImageDataset, make_loader, make_transform

ARCHIVE = ROOT / "data/raw/Custom Dataset for Plant Species and Disease Detec/Ranveer_Realistic_dataset_2000_1126.zip"
MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    rgb = (tensor.cpu() * STD + MEAN).clamp(0, 1)
    array = (rgb.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    return Image.fromarray(array, mode="RGB")


def render(archive_path: Path, split_csv: Path, mapping_path: Path, output: Path, seed: int = 42):
    """Render paired eval and stochastic training transforms for real samples."""
    random.seed(seed)
    torch.manual_seed(seed)
    rows = pd.read_csv(split_csv)
    class_to_idx = json.loads(mapping_path.read_text(encoding="utf-8"))["class_to_idx"]
    train_transform = make_transform(training=True)
    eval_transform = make_transform(training=False)
    train_data = ArchiveImageDataset(split_csv, archive_path, class_to_idx, transform=train_transform)
    eval_data = ArchiveImageDataset(split_csv, archive_path, class_to_idx, transform=eval_transform)

    # Exercise the configured PyTorch DataLoader before rendering individual pairs.
    loader = make_loader(train_data, batch_size=4, training=True, num_workers=0)
    batch_images, batch_labels = next(iter(loader))
    print(f"PyTorch DataLoader batch: images={tuple(batch_images.shape)}, labels={tuple(batch_labels.shape)}")

    preferred = ["grape_healthy", "groundnut_healthy", "guava_healthy", "soyabean_healthy"]
    indexes = []
    for label in preferred:
        matches = rows.index[rows.class_name == label].tolist()
        if matches:
            indexes.append(random.choice(matches))
    if len(indexes) < 4:
        remaining = [idx for idx in range(len(rows)) if idx not in indexes]
        indexes.extend(random.sample(remaining, 4 - len(indexes)))

    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 17) if font_path.exists() else ImageFont.load_default()
    cell_w, cell_h, margin, image_h = 420, 350, 18, 280
    canvas = Image.new("RGB", (margin + 2 * (cell_w + margin), 58 + 4 * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 16), "Actual samples from ArchiveImageDataset with train_transform", fill="#182b3a", font=font)
    for row_idx, sample_idx in enumerate(indexes):
        eval_tensor, _ = eval_data[sample_idx]
        train_tensor, _ = train_data[sample_idx]
        original = tensor_to_pil(eval_tensor)
        augmented = tensor_to_pil(train_tensor)
        y = 58 + row_idx * cell_h
        x1, x2 = margin, margin + cell_w + margin
        draw.text((x1, y), str(rows.iloc[sample_idx].class_name), fill="#273746", font=font)
        original.thumbnail((cell_w, image_h))
        augmented.thumbnail((cell_w, image_h))
        canvas.paste(original, (x1 + (cell_w - original.width) // 2, y + 27))
        canvas.paste(augmented, (x2 + (cell_w - augmented.width) // 2, y + 27))
        draw.text((x1, y + image_h + 34), "Evaluation transform", fill="#273746", font=font)
        draw.text((x2, y + image_h + 34), "Training transform sample", fill="#273746", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(f"Saved real-transform preview: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--split", type=Path, default=ROOT / "dataset/splits/train.csv")
    parser.add_argument("--mapping", type=Path, default=ROOT / "models/class_mapping.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/figures/augmentation_preview.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    render(args.archive, args.split, args.mapping, args.output, args.seed)
