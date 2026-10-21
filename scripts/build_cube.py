"""Encode the v6.0 dataset as a uint8 cube for the browser viewer.

Every cell becomes one byte on a shared 0..254 colour ramp (255 = null):
  - 5-bin features (0..4)          -> 0, 64, 127, 191, 254
  - targets (0, .25, .5, .75, 1)   -> same bytes as the bins
  - continuous features            -> min/max scaled to 0..254 (min/max over all rows)

Output (data/cube/):
  meta.json          tickers, countries, dates, columns, per-date row offsets
  row_ticker.u16     ticker index of every row, rows ordered by (date, ticker)
  values/NNNN.u8     one file per date: rows x n_columns bytes, row-major
"""
import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "v6.0"
OUT = ROOT / "data" / "cube"
SPLITS = ["train", "validation", "live"]
NULL = 255
BIN_BYTES = [0, 64, 127, 191, 254]

meta = json.loads((SRC / "features.json").read_text())
FEATURES = [f for f in meta["feature_sets"]["all"] if f != "feature_country"]
TARGETS = [t for t in meta["targets"] if t != "target"]  # `target` duplicates target_jupiter_60
COLUMNS = FEATURES + TARGETS

lf = pl.concat([pl.scan_parquet(SRC / f"{s}.parquet") for s in SPLITS])
schema = lf.collect_schema()
binned = {c for c in COLUMNS if schema[c] == pl.Int8}
continuous = [c for c in FEATURES if c not in binned]

print("computing min/max for", len(continuous), "continuous columns")
mm = lf.select(*[pl.col(c).min().alias(f"{c}|min") for c in continuous],
               *[pl.col(c).max().alias(f"{c}|max") for c in continuous]).collect().row(0, named=True)
ranges = {c: (float(mm[f"{c}|min"]), float(mm[f"{c}|max"])) for c in continuous}


def encode(c):
    if c in binned:
        return pl.col(c).replace_strict([0, 1, 2, 3, 4], BIN_BYTES, default=NULL, return_dtype=pl.UInt8)
    if c in TARGETS:  # targets are 5-level too (0, .25, .5, .75, 1): use the same bytes as the bins
        return ((pl.col(c) * 4).round().cast(pl.Int8)
                .replace_strict([0, 1, 2, 3, 4], BIN_BYTES, default=NULL, return_dtype=pl.UInt8).fill_null(NULL))
    lo, hi = ranges[c]
    scaled = ((pl.col(c) - lo) / (hi - lo) * 254).round().clip(0, 254).cast(pl.UInt8)
    return scaled.fill_null(NULL).fill_nan(NULL).alias(c)


ids = lf.select("numerai_ticker", "date", "feature_country", "data_type").collect()
tickers = ids["numerai_ticker"].unique().sort()
dates = ids["date"].unique().sort()
tick_idx = {t: i for i, t in enumerate(tickers)}
# Most recent country per ticker
country = (ids.sort("date").group_by("numerai_ticker").agg(pl.col("feature_country").last())
           .sort("numerai_ticker")["feature_country"].to_list())
first_seen = ids.group_by("numerai_ticker").agg(pl.col("date").min()).sort("numerai_ticker")["date"].to_list()
data_type = ids.group_by("date").agg(pl.col("data_type").first()).sort("date")["data_type"].to_list()

(OUT / "values").mkdir(parents=True, exist_ok=True)
row_ticker = []
offsets = [0]
# Process in date batches to bound memory
BATCH = 64
for b in range(0, len(dates), BATCH):
    batch_dates = dates[b:b + BATCH]
    df = (lf.filter(pl.col("date").is_in(batch_dates.to_list()))
          .select("date", pl.col("numerai_ticker").replace_strict(tick_idx, return_dtype=pl.UInt16).alias("t"),
                  *[encode(c) for c in COLUMNS])
          .sort("date", "t").collect())
    for j, d in enumerate(batch_dates):
        part = df.filter(pl.col("date") == d)
        vals = np.ascontiguousarray(part.select(COLUMNS).to_numpy(), dtype=np.uint8)
        (OUT / "values" / f"{b + j:04d}.u8").write_bytes(vals.tobytes())
        row_ticker.append(part["t"].to_numpy())
        offsets.append(offsets[-1] + part.height)
    print(f"dates {b}..{b + len(batch_dates) - 1} done", flush=True)

np.concatenate(row_ticker).astype("<u2").tofile(OUT / "row_ticker.u16")
(OUT / "meta.json").write_text(json.dumps({
    "null": NULL,
    "tickers": tickers.to_list(),
    "country": country,
    "first_seen": first_seen,
    "dates": dates.to_list(),
    "date_type": data_type,
    "row_offsets": offsets,
    "features": FEATURES,
    "targets": TARGETS,
    "columns": [{"name": c, "kind": "cont" if c in ranges else "bin",
                 **({"min": ranges[c][0], "max": ranges[c][1]} if c in ranges else {})} for c in COLUMNS],
}))
print(f"rows {offsets[-1]:,}  tickers {len(tickers):,}  dates {len(dates):,}  columns {len(COLUMNS)}")
