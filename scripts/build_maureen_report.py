"""Compile Maureen's verified project artifacts into Markdown and self-contained HTML."""

from __future__ import annotations

import base64
import csv
import html
import json
import re
from pathlib import Path

import mistune

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def figure(path: str, caption: str) -> str:
    return f"![{caption}]({path})\n\n*Figure: {caption}. Source file: `{path}`.*\n\n"


def section(key: str, title: str, takeaway: str, body: str) -> str:
    return f"# {title}\n\n> **Key takeaways.** {takeaway}\n\n{body.strip()}\n"


def build_sections() -> list[tuple[str, str]]:
    distribution = read_csv(ROOT / "dataset/class_distribution.csv")
    prepared = read_csv(ROOT / "dataset/prepared_class_distribution.csv")
    metrics = json.loads((ROOT / "results/test_evaluation/metrics.json").read_text(encoding="utf-8"))
    per_class = read_csv(ROOT / "results/test_evaluation/classification_report.csv")
    run_no = json.loads((ROOT / "results/loq_no_aug/summary.json").read_text(encoding="utf-8"))
    run_aug = json.loads((ROOT / "results/loq_aug/summary.json").read_text(encoding="utf-8"))
    notes = (ROOT / "results/augmentation_notes.md").read_text(encoding="utf-8").split("# Augmentation experiment notes", 1)[-1]
    class_table = "| Verified folder label | Raw images | Included after duplicate removal |\n|---|---:|---:|\n"
    prepared_counts = {row["class_name"]: int(row["image_count"]) for row in prepared}
    for row in distribution:
        class_table += f"| `{row['class_name']}` | {int(row['image_count'])} | {prepared_counts.get(row['class_name'], 0)} |\n"
    per_class_table = "| Class | Precision | Recall | F1 | Test support |\n|---|---:|---:|---:|---:|\n"
    for row in per_class:
        if row["class_or_average"] in {"macro avg", "weighted avg"}:
            continue
        per_class_table += (f"| `{row['class_or_average']}` | {float(row['precision']):.3f} | "
                            f"{float(row['recall']):.3f} | {float(row['f1_score']):.3f} | {int(float(row['support']))} |\n")
    background = section(
        "01", "Problem background", "PlantGuard AI is an academic image-classification prototype for visible plant conditions; it is not expert diagnosis.",
        "The project explores whether a leaf photograph can support a preliminary classification of plant and visible condition. Real-world appearance varies with lighting, background, and weather, while different conditions can look similar. The intended workflow is image upload, model inference, confidence display, and an optional visual explanation. The system must describe only classes represented by the verified dataset and must not imply that a Grad-CAM map precisely segments disease.\n\n"
        "The team documentation describes transfer learning and a FastAPI application as project-wide goals. Maureen's completed work covers dataset preparation, a scratch CNN baseline, augmentation comparison, and a Streamlit frontend prepared for the API contract."
    )
    dataset = section(
        "02", "Dataset and exploratory analysis", "The inspected version 2 archive contains 9,469 JPGs in 41 combined species-condition folders; ten exact duplicate copies with conflicting labels were excluded from modeling.",
        "Source: [Mendeley Data, version 2](https://data.mendeley.com/datasets/bsr2vzhrzr/2), DOI [10.17632/bsr2vzhrzr.2](https://doi.org/10.17632/bsr2vzhrzr.2). The project specification and bundled dataset README state CC BY 4.0. The archive README identifies itself as version 1.0, a source/version documentation mismatch. The archive has four species prefixes (grape, groundnut, guava, and source-spelled `soyabean`), but its labels are 41 folder-level combined categories rather than separate species and disease fields.\n\n"
        + class_table + "\n"
        + figure("../results/figures/class_distribution_counts.png", "Image counts by verified class")
        + figure("../results/figures/sample_images_by_class.png", "Representative samples from the dataset")
        + "All 9,469 JPG files were fully decoded; none were unreadable. Exact duplicate checking used SHA-256 on image file bytes: five groups (ten files), each duplicated across two Groundnut folders with conflicting labels. Both copies from every group were excluded. SHA-256 will miss visually similar files encoded differently; no perceptual duplicate search was completed. No image-level plant, session, or group ID metadata was found. The dataset bundle does not include a separate resized training copy; dimensions vary, predominantly 2000x1126 and 634x1126. The full original class names and counts above preserve source spelling. The raw class max/min ratio was 57.6:1. The archive's Groundnut summary table omits a folder present in its folder tree and archive; Papaya is mentioned in the bundled crop notes but no Papaya class exists."
    )
    preprocessing = section(
        "03", "Preprocessing and split", "The 9,459-image modeling pool was stratified by its 41 exact labels into fixed 70/15/15 train, validation, and test CSVs; path and SHA-256 overlap were checked as zero.",
        "Evaluation uses RGB, resize the shorter edge to 256, center crop to 224x224, then ImageNet mean/std normalization. The shared `models/preprocessing_config.json` remains 224x224. Training-only augmentation uses a mild 0.95-1.0 random scale crop, horizontal flip, and restrained color jitter; validation/test transforms do not augment. Horizontal flips preserve left/right orientation-independent leaf symptoms.\n\n"
        "The split uses seed 42 and exact folder class stratification: 6,621 train, 1,419 validation, and 1,419 test images. The ten files in five exact duplicate groups were removed before splitting. Since the archive has no plant/session IDs, group-aware splitting was not possible; visually related but non-identical images may still cross splits. Inverse-frequency class weights were used in both full baseline runs."
    )
    baseline_body = (
        "The from-scratch baseline has four Conv2D (3x3)-BatchNorm-ReLU-MaxPool blocks with channels 16, 32, 64, and 96, followed by adaptive global average pooling, dropout (0.30), and a 41-class linear layer. Parameter count was 83,161; float32 parameter size was 0.317 MB.\n\n"
        "| Layer | Configuration | Parameters |\n|---|---|---:|\n| Input | RGB 224x224 | - |\n| Conv block 1 | Conv 3x3, 3->16; BatchNorm; ReLU; 2x2 max pool | 464 |\n| Conv block 2 | Conv 3x3, 16->32; BatchNorm; ReLU; 2x2 max pool | 4,672 |\n| Conv block 3 | Conv 3x3, 32->64; BatchNorm; ReLU; 2x2 max pool | 18,560 |\n| Conv block 4 | Conv 3x3, 64->96; BatchNorm; ReLU; 2x2 max pool | 55,488 |\n| Head | Global average pool; dropout 0.30; linear 96->41 | 3,977 |\n| Total | Four convolutional blocks and classification head | 83,161 |\n\n"
        "| Training setting | Value |\n|---|---|\n| Epochs | 5 per run; best epoch 5 |\n| Batch size | 16 |\n| Optimizer | AdamW |\n| Learning rate / weight decay | 0.001 / 0.0001 |\n| Imbalance handling | Inverse-frequency class weights |\n| Seed / device | 42 / CUDA RTX 4050 |\n| Model selection | Validation macro F1; test evaluated once after selection |\n\n"
        "Both runs used seed 42, batch 16, AdamW, learning rate 0.001, weight decay 0.0001, class weights, and five epochs on RTX 4050 CUDA. The best checkpoint was selected by validation macro F1. The no-augmentation run reached 0.4863 validation macro F1 at epoch 5 in 786.5 seconds; the augmentation run reached 0.3525 at epoch 5 in 808.8 seconds.\n\n"
        + figure("../results/loq_no_aug/training_curves.png", "No-augmentation training and validation curves")
        + figure("../results/loq_aug/training_curves.png", "Augmented training and validation curves")
        + f"The no-augmentation checkpoint was selected using validation macro F1 ({metrics['validation_macro_f1']:.4f}) and evaluated once on the held-out test set (n={metrics['test_images']}). Test accuracy was {metrics['accuracy']:.4f}, macro precision {metrics['macro_precision']:.4f}, macro recall {metrics['macro_recall']:.4f}, and macro F1 {metrics['macro_f1']:.4f}. Model-forward time was {metrics['model_forward_ms_per_image']:.3f} ms/image on CUDA; full data loading and preprocessing pipeline time was {metrics['pipeline_ms_per_image']:.3f} ms/image.\n\n"
        + "| Test class metrics | Precision | Recall | F1 | Support |\n|---|---:|---:|---:|---:|\n"
        + "| Macro average | " + f"{metrics['macro_precision']:.3f} | {metrics['macro_recall']:.3f} | {metrics['macro_f1']:.3f} | {metrics['test_images']} |\n\n"
        + "Full per-class results are listed below. The saved confusion matrix is a PNG; the evaluation did not export the raw cell counts, so the exact top confused class pairs are **NOT YET AVAILABLE**.\n\n"
        + figure("../results/test_evaluation/confusion_matrix.png", "One-time test confusion matrix for the validation-selected no-augmentation model")
        + per_class_table
        + "\nValidation macro F1 and validation accuracy rose through epoch 5 for the no-augmentation run, and validation macro F1 rose each epoch for the augmented run. The five-epoch cap did not establish a plateau. The curves do not show a clear divergence within this short run, but they are insufficient to claim convergence."
    )
    baseline = section("04", "Baseline CNN", "The five-epoch no-augmentation CNN achieved 0.4918 test macro F1 after validation-only model selection; the short run was still improving at its final epoch.", baseline_body)
    augmentation = section(
        "05", "Augmentation experiment", "With one seed and five epochs, validation macro F1 was lower in the augmented run; this is a preliminary single-run observation, not a general conclusion.",
        notes + "\nThe results table and plot use validation metrics for both runs. The held-out test set was used once for the validation-selected no-augmentation checkpoint only; no test score is reported for the augmented run."
        + figure("../results/figures/augmentation_comparison.png", "Validation macro F1 and loss for the augmentation comparison")
    )
    threshold = json.loads((ROOT / "frontend/confidence_config.json").read_text(encoding="utf-8"))
    threshold_text = (f"A validation-derived threshold of {threshold['threshold']:.3f} is proposed. "
                      f"It was selected using the documented validation acceptance rule, with observed accepted accuracy {threshold.get('validation_accuracy_among_accepted'):.3f} and coverage {threshold.get('validation_coverage'):.3f}.") if threshold.get("threshold") is not None else "The confidence threshold is **NOT YET AVAILABLE** because validation predictions have not yet been generated from the selected checkpoint. `scripts/propose_confidence_threshold.py` derives it from validation only; the UI reads the value from `frontend/confidence_config.json`."
    frontend = section(
        "06", "Frontend", "The Streamlit upload and result flow is implemented for the evolving FastAPI contract, with mock mode clearly marked and real predictions sourced only from the API.",
        "The interface accepts JPG/JPEG/PNG files, checks that uploads can be decoded, previews the image, and sends it to `POST /predict`. It displays predicted class, confidence, optional status, and an optional Grad-CAM image or placeholder. Missing optional fields do not crash the display. Mock mode is controlled by `PLANTGUARD_MOCK_MODE=true` and prominently labels simulated output. The API base URL is read from `PLANTGUARD_API_BASE_URL`.\n\n"
        + threshold_text + "\n\nA screenshot was not supplied; insert an application screenshot here after backend integration. The disclaimer reads: AI prediction, not expert diagnosis. See `docs/api_contract.md` for the expected fields."
    )
    limitations = section(
        "07", "Limitations (Maureen's work)", "Results are limited by the source labels, severe class imbalance, lack of group metadata, one random seed, and a short scratch-CNN run.",
        "The 41 labels combine species and conditions and retain source spelling errors; the model cannot produce separate species and disease outputs without additional verified annotations. The folder names do not provide validated disease ground truth beyond the dataset's labeling. Class frequencies range widely, including small classes with limited test support.\n\n"
        "No plant/session grouping metadata was supplied, so residual leakage among related but non-identical images is possible. Only one split and seed were used. The baseline is a small CNN trained for five epochs, and its validation scores were still increasing at the final epoch. The reported test metrics apply to one checkpoint selected by validation macro F1. No robustness evaluation, external field validation, or multi-seed uncertainty analysis is included. Confidence values are not calibrated probabilities. The frontend is contract-ready but has not been integrated with Disca's real backend."
    )
    return [
        ("01_problem_background.md", background), ("02_dataset.md", dataset),
        ("03_preprocessing_split.md", preprocessing), ("04_baseline_cnn.md", baseline),
        ("05_augmentation_experiment.md", augmentation), ("06_frontend.md", frontend),
        ("07_limitations_maureen.md", limitations),
    ]


def markdown_to_self_contained_html(markdown_text: str) -> str:
    renderer = mistune.create_markdown(plugins=["table", "strikethrough"])
    content = renderer(markdown_text)
    def embed(match):
        src = html.unescape(match.group(1))
        path = (REPORT / src).resolve()
        try:
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            return f'<img src="data:{mime};base64,{data}"'
        except OSError:
            return match.group(0)
    content = re.sub(r'<img src="([^"]+)"', embed, content)
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Maureen's PlantGuard AI Sections</title>
<style>
@page { size: A4; margin: 12mm; } body { font: 10pt/1.42 Georgia, 'Times New Roman', serif; color:#202a31; max-width: 900px; margin: 2rem auto; padding: 0 1.25rem; }
h1,h2,h3 { font-family: Arial, sans-serif; color:#164a3b; line-height:1.2; page-break-after:avoid; } h1 { border-bottom:2px solid #4e8b68; padding-bottom:.25em; margin-top:1.2em; }
blockquote { background:#eef6f0; border-left:4px solid #4e8b68; padding:.5rem 1rem; margin:1rem 0; }
table { border-collapse:collapse; width:100%; font:7pt Arial,sans-serif; page-break-inside:auto; } th,td { border:1px solid #cbd5d1; padding:2px 4px; vertical-align:top; } th { background:#e7f0eb; }
img { display:block; max-width:100%; height:auto; margin:.5rem auto; } figure { page-break-inside:avoid; } p { orphans:3; widows:3; } a { color:#14664c; }
@media print { body { max-width:none; margin:0; padding:0; font-size:8.8pt; } h1 { margin-top:.7em; } }
</style></head><body>""" + content + "</body></html>"


def main() -> None:
    REPORT.mkdir(exist_ok=True)
    sections = build_sections()
    for filename, text in sections:
        (REPORT / filename).write_text(text.strip() + "\n", encoding="utf-8")
    toc = "# Contents\n\n" + "\n".join(f"- {name[3:-3].replace('_', ' ').title()}" for name, _ in sections)
    combined = "# PlantGuard AI: Maureen's Project Sections\n\n" + toc + "\n\n" + "\n\n---\n\n".join(text for _, text in sections)
    (REPORT / "maureen_sections.md").write_text(combined, encoding="utf-8")
    (REPORT / "maureen_sections.html").write_text(markdown_to_self_contained_html(combined), encoding="utf-8")

    metrics = json.loads((ROOT / "results/test_evaluation/metrics.json").read_text(encoding="utf-8"))
    baseline = json.loads((ROOT / "results/loq_no_aug/summary.json").read_text(encoding="utf-8"))
    augmented = json.loads((ROOT / "results/loq_aug/summary.json").read_text(encoding="utf-8"))
    summary = f"""# Maureen's results summary

## Key findings

- Verified archive: 9,469 images, 41 combined folder labels; five exact duplicate pairs were dropped in full, leaving 9,459 images.
- Fixed stratified split: 6,621 train / 1,419 validation / 1,419 test. Path and SHA-256 overlap: zero.
- Best validation run: no augmentation, macro F1 {baseline['best_val_macro_f1']:.4f} at epoch {baseline['best_epoch']}; augmented run {augmented['best_val_macro_f1']:.4f} at epoch {augmented['best_epoch']}.
- Validation-selected no-augmentation test result (evaluated once): accuracy {metrics['accuracy']:.4f}; macro precision {metrics['macro_precision']:.4f}; macro recall {metrics['macro_recall']:.4f}; macro F1 {metrics['macro_f1']:.4f}.
- The five-epoch runs were still improving at the last epoch; one seed is not enough to generalize the augmentation comparison.

## Main comparison

| Experiment | Best validation macro F1 | Test macro F1 |
|---|---:|---:|
| No augmentation | {baseline['best_val_macro_f1']:.4f} | {metrics['macro_f1']:.4f} |
| Training augmentation | {augmented['best_val_macro_f1']:.4f} | Not evaluated (test reserved for selected model) |

## Figures

![Verified class counts](../results/figures/class_distribution_counts.png)

![Training augmentation comparison](../results/figures/augmentation_comparison.png)

![Selected baseline test confusion matrix](../results/test_evaluation/confusion_matrix.png)

![Dataset sample grid](../results/figures/sample_images_by_class.png)

Detailed metrics: `results/experiment_results.csv`, `results/test_evaluation/metrics.json`, and `results/test_evaluation/classification_report.csv`.
"""
    (REPORT / "results_summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
