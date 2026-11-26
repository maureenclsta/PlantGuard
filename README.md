# PlantGuard AI

PlantGuard AI is a course prototype for classifying the verified plant-condition labels in the selected image dataset. The application is an AI prediction tool, not an expert agricultural diagnosis.

## Dataset

Download Mendeley Data version 2 (DOI 10.17632/bsr2vzhrzr.2) and place the archive at:

```text
data/raw/Custom Dataset for Plant Species and Disease Detec/Ranveer_Realistic_dataset_2000_1126.zip
```

Do not push the dataset archive or extracted data to Git. See [dataset/README_dataset.md](dataset/README_dataset.md) for verified labels, counts, license notes, duplicates, and limitations.

## Setup

Use Python 3.10 or newer. Install PyTorch and torchvision using the official selector for the machine's CPU/CUDA setup, then install project dependencies:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -r requirements.txt
```

On Linux/macOS, activate with `source .venv/bin/activate`. The Python scripts detect CUDA automatically unless a device is specified.

## Data inspection and preparation

```bat
python scripts\prepare_data.py
python scripts\run_eda.py
python scripts\visualize_augmentations.py
```

The preparation script decodes and hashes archive images, removes both members of conflicting exact duplicate pairs, writes deterministic stratified split CSVs, and checks path and SHA-256 overlap. Rebuild the two notebooks with:

```bat
jupyter nbconvert --to notebook --execute --inplace notebooks\01_dataset_eda.ipynb notebooks\02_preprocessing.ipynb
```

## Baseline training and evaluation

The completed RTX 4050 runs used five epochs each. To reproduce them on a suitable CUDA system:

```bat
python scripts\train_baseline.py --epochs 5 --batch-size 16 --lr 0.001 --optimizer adamw --weight-decay 0.0001 --augmentation off --imbalance-handling weights --seed 42 --device auto --output-dir results\loq_no_aug --patience 3 --num-workers 0
python scripts\train_baseline.py --epochs 5 --batch-size 16 --lr 0.001 --optimizer adamw --weight-decay 0.0001 --augmentation on --imbalance-handling weights --seed 42 --device auto --output-dir results\loq_aug --patience 3 --num-workers 0
```

Training writes logs, config, curves, and a best checkpoint. Checkpoints are ignored by Git. The validation-selected baseline test evaluation is a one-time operation; do not repeat it after `results/test_evaluation/metrics.json` exists:

```bat
python scripts\evaluate_baseline.py
```

Compare the existing completed runs without training:

```bat
python scripts\compare_augmentation.py
```

For a validation-only confidence threshold, run this on the machine with the `results/loq_no_aug/best_checkpoint.pt` checkpoint:

```bat
python scripts\propose_confidence_threshold.py
```

## Frontend

```bat
set PLANTGUARD_MOCK_MODE=true
streamlit run frontend\app.py
```

Mock mode is clearly labelled and simulates the interface only. For backend integration, set `PLANTGUARD_MOCK_MODE=false` and `PLANTGUARD_API_BASE_URL` to the FastAPI server. The contract is documented in [docs/api_contract.md](docs/api_contract.md). Maureen's frontend does not implement or modify `backend/`.

## Tests

Run the lightweight repository checks with:

```bat
python -m unittest discover -s tests -v
```

## Report

Build Maureen's report sections and self-contained HTML from the saved results:

```bat
python scripts\build_maureen_report.py
python scripts\export_maureen_pdf.py
```

The generated files are in `report/`. The HTML embeds its figures and includes print CSS; ReportLab exports the same Markdown and figures to PDF.
