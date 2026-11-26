"""Lightweight checks for Maureen's reproducibility and report contracts."""

from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.plantguard.api_client import PredictionResponseError, parse_prediction_response
from src.plantguard.data import make_transform


def rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class ProjectChecks(unittest.TestCase):
    def test_split_path_and_hash_disjointness(self):
        splits = [rows(ROOT / "dataset/splits" / f"{name}.csv") for name in ("train", "val", "test")]
        for left in range(3):
            for right in range(left + 1, 3):
                self.assertFalse({r["path"] for r in splits[left]} & {r["path"] for r in splits[right]})
                self.assertFalse({r["sha256"] for r in splits[left]} & {r["sha256"] for r in splits[right]})

    def test_class_mapping_matches_prepared_dataset(self):
        mapping = json.loads((ROOT / "models/class_mapping.json").read_text(encoding="utf-8"))["class_to_idx"]
        prepared = {row["class_name"] for row in rows(ROOT / "dataset/prepared_class_distribution.csv")}
        self.assertEqual(set(mapping), prepared)
        self.assertEqual(sorted(mapping.values()), list(range(len(mapping))))

    def test_eval_transform_shape_and_normalization(self):
        transform = make_transform(training=False)
        output = transform(Image.new("RGB", (300, 300), (0, 0, 0)))
        self.assertEqual(tuple(output.shape), (3, 224, 224))
        expected = (-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225)
        for actual, value in zip(output[:, 0, 0].tolist(), expected):
            self.assertAlmostEqual(actual, value, places=5)

    def test_frontend_response_parser(self):
        parsed = parse_prediction_response({"predicted_class": "grape_healthy", "confidence": 0.7})
        self.assertEqual(parsed["predicted_class"], "grape_healthy")
        self.assertFalse(parsed["gradcam_available"])
        with self.assertRaises(PredictionResponseError):
            parse_prediction_response({"predicted_class": "x", "confidence": 2})

    def test_report_test_metrics_match_results_csv(self):
        metrics = json.loads((ROOT / "results/test_evaluation/metrics.json").read_text(encoding="utf-8"))
        records = rows(ROOT / "results/experiment_results.csv")
        matching = [row for row in records if row["experiment_name"] == "baseline_loq_no_aug_selected"]
        self.assertEqual(len(matching), 1)
        for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
            self.assertAlmostEqual(float(matching[0][key]), float(metrics[key]), places=12)
        self.assertEqual(matching[0]["evaluation_split"], "test")
        for run_name in ("loq_no_aug", "loq_aug"):
            summary = json.loads((ROOT / "results" / run_name / "summary.json").read_text(encoding="utf-8"))
            validation_rows = [row for row in records if row["experiment_name"] == f"baseline_{run_name}_validation"]
            self.assertEqual(len(validation_rows), 1)
            self.assertEqual(validation_rows[0]["evaluation_split"], "validation")
            self.assertAlmostEqual(float(validation_rows[0]["macro_f1"]), summary["best_val_macro_f1"], places=12)
        report = (ROOT / "report/04_baseline_cnn.md").read_text(encoding="utf-8")
        self.assertIn(f"{metrics['macro_f1']:.4f}", report)


if __name__ == "__main__":
    unittest.main()
