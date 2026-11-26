# Maureen workstream handoff

## For Alicia

- `models/class_mapping.json`: 41 exact verified labels and indices.
- `models/preprocessing_config.json`: shared RGB 224x224 evaluation input, resize/crop, and ImageNet normalization contract.
- `dataset/splits/{train,val,test}.csv`: deterministic seed 42 splits; the shared dataset archive is not committed.
- `dataset/README_dataset.md`: verified source, labels, duplicates, split, and residual leakage risk.

## For Disca

- `docs/api_contract.md`: frontend request and response contract for `POST /predict`.
- `frontend/app.py`: Streamlit UI with configurable API URL and clearly labelled mock mode.
- `src/plantguard/api_client.py`: defensive response parser.
- `frontend/confidence_config.json`: threshold proposed from the saved validation predictions; see the confidence caveat in the file and report.

## Report and result files

- `report/01_problem_background.md` through `report/07_limitations_maureen.md` and `report/maureen_sections.md` are Maureen's report sections, ready for team integration.
- `report/maureen_sections.html` is self-contained and print-friendly.
- `report/results_summary.md` is the short handoff overview.
- `results/loq_no_aug/` and `results/loq_aug/` contain five-epoch configs, training logs, summaries, and curves.
- `results/augmentation_comparison.csv`, `results/figures/augmentation_comparison.png`, and `results/augmentation_notes.md` compare validation results.
- `results/test_evaluation/` contains the one-time selected-model test metrics, class report, and confusion matrix.
- Checkpoints are excluded from Git. Validation predictions, threshold analysis, and its figure were generated from the selected checkpoint on the RTX 4050 laptop; this was validation inference, not training.

## PDF export

Build the self-contained HTML and PDF from the repository root:

```bat
python scripts\build_maureen_report.py
python scripts\export_maureen_pdf.py
```

## Remaining Maureen-specific gaps

- Obtain a real screenshot after integration with the backend.
- Export and visually review the PDF on a machine with a PDF engine.
- The test evaluator saved a confusion matrix PNG but not raw cell counts, so exact most-confused class pairs are not extracted.
- No multi-seed augmentation results or additional baseline training are included.
