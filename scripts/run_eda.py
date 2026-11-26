"""Build dataset EDA outputs from files created by prepare_data.py."""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.plantguard.eda import run_eda

archive = ROOT / "data/raw/Custom Dataset for Plant Species and Disease Detec/Ranveer_Realistic_dataset_2000_1126.zip"
result = run_eda(archive, ROOT / "dataset/image_inventory.csv", ROOT / "dataset/class_distribution.csv",
                 ROOT / "results/figures", ROOT / "dataset/image_size_stats.csv")
print(json.dumps(result, indent=2))
