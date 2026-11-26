"""Inspect the source archive, deduplicate exact bytes, and save stratified splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "data/raw/Custom Dataset for Plant Species and Disease Detec/Ranveer_Realistic_dataset_2000_1126.zip"
SEED = 42


def inspect(archive_path: Path):
    rows = []
    hashes = defaultdict(list)
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir() or Path(info.filename).suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
                continue
            image_bytes = archive.read(info.filename)
            digest = hashlib.sha256(image_bytes).hexdigest()
            label = Path(info.filename).parent.name
            with Image.open(__import__("io").BytesIO(image_bytes)) as image:
                image.load()
                width, height, fmt = image.width, image.height, image.format
            rows.append({"path": info.filename, "class_name": label, "sha256": digest,
                         "width": width, "height": height, "format": fmt,
                         "aspect_ratio": width / height})
            hashes[digest].append(info.filename)
    rows.sort(key=lambda row: row["path"])
    duplicates = {digest: sorted(paths) for digest, paths in hashes.items() if len(paths) > 1}
    return rows, duplicates


def split_rows(rows, seed=SEED):
    """Drop every member of an exact duplicate group, then stratify."""
    by_hash = defaultdict(list)
    for row in rows:
        by_hash[row["sha256"]].append(row)
    unique, duplicate_records = [], []
    for digest, group in by_hash.items():
        group.sort(key=lambda row: row["path"])
        if len(group) > 1:
            for duplicate in group:
                duplicate_records.append({"sha256": digest, "path": duplicate["path"],
                                          "class_name": duplicate["class_name"],
                                          "decision": "excluded_exact_duplicate_group"})
        else:
            unique.append(group[0])
    unique.sort(key=lambda row: row["path"])
    labels = [row["class_name"] for row in unique]
    train_val, test = train_test_split(unique, test_size=0.15, random_state=seed,
                                       stratify=labels)
    train_labels = [row["class_name"] for row in train_val]
    train, val = train_test_split(train_val, test_size=(0.15 / 0.85),
                                  random_state=seed, stratify=train_labels)
    splits = {"train": sorted(train, key=lambda row: row["path"]),
              "val": sorted(val, key=lambda row: row["path"]),
              "test": sorted(test, key=lambda row: row["path"])}
    return splits, duplicate_records


def save_csv(path: Path, records, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records, columns=columns).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if not args.archive.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {args.archive}")

    dataset_dir = ROOT / "dataset"
    rows, _ = inspect(args.archive)
    splits, duplicate_records = split_rows(rows, args.seed)
    raw_counts = pd.Series([r["class_name"] for r in rows]).value_counts().sort_index()
    save_csv(dataset_dir / "class_distribution.csv",
             [{"class_name": name, "image_count": int(count)} for name, count in raw_counts.items()],
             ["class_name", "image_count"])
    duplicate_hashes = {record["sha256"] for record in duplicate_records}
    for row in rows:
        excluded = row["sha256"] in duplicate_hashes
        row["modeling_included"] = not excluded
        row["exclusion_reason"] = "exact_duplicate_group" if excluded else ""
    save_csv(dataset_dir / "image_inventory.csv", rows,
             ["path", "class_name", "sha256", "width", "height", "format", "aspect_ratio",
              "modeling_included", "exclusion_reason"])
    save_csv(dataset_dir / "duplicate_report.csv", duplicate_records,
             ["sha256", "path", "class_name", "decision"])

    included_rows = [row for row in rows if row["modeling_included"]]
    class_names = sorted({row["class_name"] for row in included_rows})
    mapping = {name: index for index, name in enumerate(class_names)}
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "class_mapping.json").write_text(
        json.dumps({"class_to_idx": mapping, "idx_to_class": {str(v): k for k, v in mapping.items()}}, indent=2) + "\n",
        encoding="utf-8")
    prepared_counts = pd.Series([row["class_name"] for row in included_rows]).value_counts().sort_index()
    save_csv(dataset_dir / "prepared_class_distribution.csv",
             [{"class_name": name, "image_count": int(count)} for name, count in prepared_counts.items()],
             ["class_name", "image_count"])

    split_dir = dataset_dir / "splits"
    for name, records in splits.items():
        save_csv(split_dir / f"{name}.csv", records, ["path", "class_name", "sha256"])
    path_sets = {name: {row["path"] for row in records} for name, records in splits.items()}
    assert not (path_sets["train"] & path_sets["val"])
    assert not (path_sets["train"] & path_sets["test"])
    assert not (path_sets["val"] & path_sets["test"])
    hashes_by_split = {name: {row["sha256"] for row in records} for name, records in splits.items()}
    assert not (hashes_by_split["train"] & hashes_by_split["val"])
    assert not (hashes_by_split["train"] & hashes_by_split["test"])
    assert not (hashes_by_split["val"] & hashes_by_split["test"])
    print(f"Decoded inventory: {len(rows)} images, {len(class_names)} classes")
    print(f"Included after exact-hash group exclusion: {sum(map(len, splits.values()))} images")
    print("Split sizes:", {name: len(records) for name, records in splits.items()})
    print(f"Excluded duplicate group members: {len(duplicate_records)}; split path/hash overlap: 0")


if __name__ == "__main__":
    main()
