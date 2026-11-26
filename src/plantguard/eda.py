"""Exploratory dataset figures built from the verified archive inventory."""

from __future__ import annotations

import io
import math
import zipfile
import textwrap
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # Pillow fallback lets EDA figures be built in minimal environments.
    plt = None


def _font(size=16, bold=False):
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def _save_horizontal_bars(rows, output: Path, title: str, value_key: str, as_percent=False):
    rows = sorted(rows, key=lambda row: row[value_key])
    if plt is not None:
        labels = [row["class_name"] for row in rows]
        values = [row[value_key] for row in rows]
        fig, ax = plt.subplots(figsize=(12, max(7, len(rows) * 0.32)))
        ax.barh(labels, values, color="#4385a8")
        ax.set_title(title)
        ax.set_xlabel("Percent of all images" if as_percent else "Image count")
        ax.grid(axis="x", alpha=0.2)
        for y, value in enumerate(values):
            ax.text(value, y, f" {value:.2f}%" if as_percent else f" {value:,}",
                    va="center", fontsize=7)
        fig.tight_layout()
        fig.savefig(output, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return
    width, left, right, top, row_h = 1500, 500, 90, 100, 34
    height = top + row_h * len(rows) + 70
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, label_font, value_font = _font(28, True), _font(15), _font(14)
    draw.text((left, 25), title, fill="#182b3a", font=title_font)
    max_value = max(row[value_key] for row in rows) or 1
    plot_w = width - left - right
    for index, row in enumerate(rows):
        y = top + index * row_h
        draw.text((20, y + 5), row["class_name"], fill="#273746", font=label_font)
        bar_w = int(plot_w * row[value_key] / max_value)
        draw.rounded_rectangle((left, y + 2, left + bar_w, y + row_h - 5), radius=5, fill="#4385a8")
        value = f"{row[value_key]:.2f}%" if as_percent else f"{row[value_key]:,}"
        draw.text((left + bar_w + 8, y + 6), value, fill="#273746", font=value_font)
    image.save(output)


def _save_sample_grid(archive_path: Path, inventory: pd.DataFrame, output: Path):
    chosen = inventory.sort_values("path").groupby("class_name", sort=True).nth(0).reset_index()
    columns, cell_w, cell_h, image_h, margin = 5, 250, 220, 150, 14
    rows = math.ceil(len(chosen) / columns)
    canvas = Image.new("RGB", (margin + columns * (cell_w + margin), margin + rows * (cell_h + margin)), "white")
    draw = ImageDraw.Draw(canvas)
    label_font = _font(12, True)
    with zipfile.ZipFile(archive_path) as archive:
        for index, row in chosen.iterrows():
            x = margin + (index % columns) * (cell_w + margin)
            y = margin + (index // columns) * (cell_h + margin)
            with archive.open(row["path"]) as member:
                source = Image.open(io.BytesIO(member.read())).convert("RGB")
            source.thumbnail((cell_w, image_h))
            canvas.paste(source, (x + (cell_w - source.width) // 2, y))
            label = row["class_name"]
            lines = textwrap.wrap(label, width=35)
            for line_number, line in enumerate(lines):
                draw.text((x, y + image_h + 3 + line_number * 14), line,
                          fill="#182b3a", font=label_font)
            count_y = y + image_h + 4 + len(lines) * 14
            draw.text((x, count_y), f"n={int((inventory['class_name'] == label).sum())}",
                      fill="#4d626e", font=label_font)
    canvas.save(output)


def _save_size_and_ratio(inventory: pd.DataFrame, output: Path):
    if plt is not None:
        dimensions = inventory.groupby(["width", "height"]).size().sort_values(ascending=False)
        ratios = inventory["aspect_ratio"].astype(float)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].barh([f"{w} x {h}" for w, h in dimensions.index[::-1]],
                     dimensions.values[::-1], color="#4385a8")
        axes[0].set_title("Image dimensions")
        axes[0].set_xlabel("Images")
        axes[1].hist(ratios, bins=12, color="#68a77a", edgecolor="white")
        axes[1].set_title("Aspect ratio (width / height)")
        axes[1].set_xlabel("Width / height")
        axes[1].set_ylabel("Images")
        axes[1].text(0.03, 0.97,
                     f"median={ratios.median():.3f}\nmean={ratios.mean():.3f}\nmin={ratios.min():.3f}\nmax={ratios.max():.3f}",
                     transform=axes[1].transAxes, va="top")
        fig.tight_layout()
        fig.savefig(output, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return
    width, height = 1500, 850
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font, label_font = _font(28, True), _font(17)
    draw.text((60, 28), "Image dimensions and aspect ratio", fill="#182b3a", font=title_font)
    dim_counts = inventory.groupby(["width", "height"]).size().sort_values(ascending=False)
    top_dims = dim_counts.head(8)
    draw.text((80, 95), "Most common dimensions", fill="#273746", font=_font(20, True))
    max_n = int(top_dims.max())
    for i, (dims, count) in enumerate(top_dims.items()):
        y = 145 + i * 42
        draw.text((80, y), f"{dims[0]} × {dims[1]}", fill="#273746", font=label_font)
        bar_w = int(390 * count / max_n)
        draw.rectangle((250, y + 2, 250 + bar_w, y + 25), fill="#4385a8")
        draw.text((660, y), f"{count:,} ({count / len(inventory):.1%})", fill="#273746", font=label_font)
    ratios = inventory["aspect_ratio"].astype(float)
    bins = [0.0, .5, .65, .8, .9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0]
    histogram, edges = pd.cut(ratios, bins=bins, right=False, include_lowest=True).value_counts(sort=False).values, bins
    draw.text((820, 95), "Aspect ratio (width / height)", fill="#273746", font=_font(20, True))
    max_h = max(histogram) or 1
    for i, count in enumerate(histogram):
        x = 840 + i * 58
        bar_h = int(430 * count / max_h)
        draw.rectangle((x, 650 - bar_h, x + 35, 650), fill="#68a77a")
        draw.text((x - 3, 665), f"{edges[i]:g}", fill="#273746", font=_font(11))
    draw.text((820, 710), f"median={ratios.median():.3f}   mean={ratios.mean():.3f}   min={ratios.min():.3f}   max={ratios.max():.3f}", fill="#273746", font=label_font)
    draw.text((80, 555), f"Unique dimensions: {len(dim_counts)}", fill="#273746", font=label_font)
    draw.text((80, 590), f"Dimension range: {int(inventory.width.min())}–{int(inventory.width.max())} px × {int(inventory.height.min())}–{int(inventory.height.max())} px", fill="#273746", font=label_font)
    canvas.save(output)


def run_eda(archive_path: str | Path, inventory_csv: str | Path,
            distribution_csv: str | Path, figures_dir: str | Path,
            stats_csv: str | Path):
    """Create count/percentage plots, per-class sample grid and size analysis."""
    archive_path, figures_dir = Path(archive_path), Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    inventory = pd.read_csv(inventory_csv)
    distribution = pd.read_csv(distribution_csv)
    total = int(distribution.image_count.sum())
    records = distribution.to_dict("records")
    for row in records:
        row["percentage"] = 100 * row["image_count"] / total
    _save_horizontal_bars(records, figures_dir / "class_distribution_counts.png",
                          "Image count by folder label", "image_count")
    _save_horizontal_bars(records, figures_dir / "class_distribution_percent.png",
                          "Class share of all images (%)", "percentage", as_percent=True)
    _save_sample_grid(archive_path, inventory, figures_dir / "sample_images_by_class.png")
    _save_size_and_ratio(inventory, figures_dir / "image_size_aspect_ratio.png")
    sizes = inventory.groupby(["width", "height"], as_index=False).agg(image_count=("path", "count"))
    ratios = inventory["aspect_ratio"]
    sizes.to_csv(stats_csv, index=False)
    return {
        "images": total, "classes": len(distribution),
        "imbalance_ratio": float(distribution.image_count.max() / distribution.image_count.min()),
        "aspect_ratio_min": float(ratios.min()), "aspect_ratio_max": float(ratios.max()),
        "aspect_ratio_mean": float(ratios.mean()), "aspect_ratio_median": float(ratios.median()),
        "unique_dimensions": int(len(sizes)),
    }
