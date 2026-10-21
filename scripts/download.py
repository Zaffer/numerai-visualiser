"""Download the Numerai Classic v6.0 (Unity) dataset files into data/."""
import sys
from pathlib import Path

from numerapi import NumerAPI

VERSION = "v6.0"
FILES = [
    "features.json",
    "train.parquet",
    "validation.parquet",
    "live.parquet",
    "train_benchmark_models.parquet",
    "validation_benchmark_models.parquet",
]

napi = NumerAPI()
out = Path(__file__).resolve().parent.parent / "data"
for name in sys.argv[1:] or FILES:
    dest = out / VERSION / name
    print(f"downloading {VERSION}/{name}", flush=True)
    napi.download_dataset(f"{VERSION}/{name}", str(dest))
    print(f"  -> {dest} ({dest.stat().st_size / 1e9:.2f} GB)", flush=True)
