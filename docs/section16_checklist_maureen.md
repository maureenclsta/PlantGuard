# Section 16 checklist: Maureen workstream

Evidence is recorded from the repository artifacts. Items belonging to Alicia, Disca, or the whole team are marked outside Maureen's scope.

## Dataset

- [x] Dataset downloaded and accessible during inspection: `dataset/README_dataset.md`, `dataset/image_inventory.csv`.
- [x] DOI and license recorded: `dataset/README_dataset.md` (license stated in project specification and bundled dataset README; confirm against Mendeley record before redistribution).
- [x] Actual labels verified: `dataset/class_distribution.csv`, `models/class_mapping.json`.
- [x] Class distribution calculated: `dataset/class_distribution.csv`, `results/figures/class_distribution_counts.png`.
- [x] Corrupt/duplicate checks completed: `dataset/image_inventory.csv`, `dataset/duplicate_report.csv`, `dataset/README_dataset.md`.
- [x] Train/validation/test split documented: `dataset/splits/*.csv`, `dataset/README_dataset.md`.
- [x] Leakage risk considered: duplicate SHA-256 overlap is zero; no plant/session identifiers were found, so residual group leakage remains possible.

## AI and experiments

- [x] CNN baseline trained: `results/loq_no_aug/`, `results/loq_aug/` (five epochs each).
- [ ] Pretrained model(s) trained: Alicia's scope.
- [ ] Fine-tuning strategy documented: Alicia's scope.
- [x] Augmentation experiment completed: `results/augmentation_comparison.csv`; one seed, five epochs, interpretation limited accordingly.
- [x] Metrics calculated: one-time test results in `results/test_evaluation/metrics.json` and class report CSV.
- [x] Confusion matrix generated: `results/test_evaluation/confusion_matrix.png`.
- [x] Training curves reviewed: `results/loq_*/training_curves.png`, `results/baseline_notes.md`.
- [ ] Exact error-pair analysis: raw confusion cell values were not exported; visual PNG is available.
- [ ] Robustness test completed: not part of Maureen's completed runs.
- [x] Best Maureen baseline selected using validation data: `results/test_evaluation/metrics.json`.

## Application

- [ ] FastAPI backend runs: Disca's scope.
- [x] Image upload, preview, Analyze, result, confidence, error states implemented: `frontend/app.py`.
- [ ] Real prediction endpoint integration verified: pending Disca's backend.
- [ ] Real Grad-CAM integration verified: pending Disca's backend.
- [x] Validation-derived low-confidence threshold proposed: `frontend/confidence_config.json`, `results/validation_confidence_analysis.csv`, and `results/figures/validation_confidence_threshold.png`.
- [ ] End-to-end test completed: pending backend integration.

## Deliverables and shared work

- [x] Maureen report sections and summary generated under `report/`.
- [ ] PDF export: no PDF engine available in the build environment; command is in `docs/maureen_handoff.md`.
- [ ] Presentation and demo video: shared team deliverables, outside this workstream.
- [x] Reproduction instructions added to `README.md`.
