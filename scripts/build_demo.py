"""Build the viewer's built-in demo: one slice per axis of the Numerai v6.0 cube.

  uv run python scripts/build_demo.py [--date 2020-03-20] [--column feature_market_cap_factor] [--ticker "AAPL US"]

Writes viewer/demo/:
  meta.json     tickers, countries, dates, columns (with min/max), and the chosen slices
  rows.bin.gz   which (ticker, date) rows exist: uint32 row count per date, then per date the
                ticker ids (alphabetical order) as uint16 deltas
  values.bin.gz bytes on the viewer's 0..254 ramp (255 = null), three blocks back to back:
                A  every column for the rows on the chosen date (ticker order), column-major
                B  every column for the chosen ticker's other rows (date order), column-major
                C  the chosen column for every remaining row, in (date, ticker) order
The encoding matches viewer/ingest-worker.js.
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "v6.0"
OUT = ROOT / "viewer" / "demo"
SPLITS = ["train", "validation", "live"]
BIN = np.array([0, 64, 127, 191, 254], dtype=np.uint8)
NULL = 255

ap = argparse.ArgumentParser()
ap.add_argument("--date", default="2020-03-20")
ap.add_argument("--column", default="feature_market_cap_factor")
ap.add_argument("--ticker", default="AAPL US")
args = ap.parse_args()

lf = pl.concat([pl.scan_parquet(SRC / f"{s}.parquet") for s in SPLITS])
schema = lf.collect_schema()
names = list(schema.names())
features = [c for c in names if c.startswith("feature_") and c != "feature_country"]
targets = [c for c in names if c.startswith("target_")]
columns = features + targets
kind = {c: "target" if c in targets else "bin" if schema[c] == pl.Int8 else "cont" for c in columns}
cont = [c for c in columns if kind[c] == "cont"]
mm = lf.select(*[pl.col(c).min().alias(f"{c}|lo") for c in cont], *[pl.col(c).max().alias(f"{c}|hi") for c in cont]).collect().row(0, named=True)
lo = {c: float(mm[f"{c}|lo"]) for c in cont}
hi = {c: float(mm[f"{c}|hi"]) for c in cont}


def encode(c, values):
    v = np.asarray(values, dtype=np.float64)
    out = np.full(v.shape, NULL, dtype=np.uint8)
    ok = ~np.isnan(v)
    if kind[c] == "cont":
        out[ok] = np.clip(np.floor((v[ok] - lo[c]) * (254 / (hi[c] - lo[c])) + 0.5), 0, 254)
    else:
        k = v[ok] * 4 if kind[c] == "target" else v[ok]
        k = np.floor(k + 0.5).astype(np.int64)
        good = (k >= 0) & (k <= 4)
        enc = np.full(k.shape, NULL, dtype=np.uint8)
        enc[good] = BIN[k[good]]
        out[ok] = enc
    return out


ids = lf.select("numerai_ticker", "date", "feature_country", "data_type").collect()
tickers = ids["numerai_ticker"].unique().sort().to_list()
dates = ids["date"].unique().sort().to_list()
assert args.date in dates and args.ticker in tickers and args.column in columns
t_id = {t: i for i, t in enumerate(tickers)}
d_id = {d: i for i, d in enumerate(dates)}
per_t = ids.sort("date").group_by("numerai_ticker").agg(pl.col("feature_country").last(), pl.col("date").min().alias("first")).sort("numerai_ticker")
date_type = ids.group_by("date").agg(pl.col("data_type").first()).sort("date")["data_type"].to_list()

# Every row in (date, ticker) order, which is the order the viewer walks rows.bin.
rows = (lf.select(pl.col("date").replace_strict(d_id, return_dtype=pl.UInt16).alias("d"),
                  pl.col("numerai_ticker").replace_strict(t_id, return_dtype=pl.UInt16).alias("t"),
                  pl.col(args.column).cast(pl.Float64).alias("v"))
        .sort("d", "t").collect())
d = rows["d"].to_numpy().astype(np.int64)
t = rows["t"].to_numpy().astype(np.int64)
d0, t0 = d_id[args.date], t_id[args.ticker]

counts = np.bincount(d, minlength=len(dates)).astype("<u4")
deltas = np.diff(t, prepend=0)
deltas[np.r_[0, np.cumsum(counts)[:-1]]] = t[np.r_[0, np.cumsum(counts)[:-1]]]  # first ticker of each date is absolute
rows_bin = counts.tobytes() + deltas.astype("<u2").tobytes()


def block(frame):
    return b"".join(encode(c, frame[c].cast(pl.Float64).to_numpy()).tobytes() for c in columns)


full = lf.select("numerai_ticker", "date", *columns)
A = full.filter(pl.col("date") == args.date).sort("numerai_ticker").collect()
B = full.filter((pl.col("numerai_ticker") == args.ticker) & (pl.col("date") != args.date)).sort("date").collect()
in_c = (d != d0) & (t != t0)
C = encode(args.column, rows["v"].to_numpy()[in_c])
assert A.height + B.height + int(in_c.sum()) == rows.height
values_bin = block(A) + block(B) + C.tobytes()

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "rows.bin.gz").write_bytes(gzip.compress(rows_bin, 9, mtime=0))
(OUT / "values.bin.gz").write_bytes(gzip.compress(values_bin, 9, mtime=0))
(OUT / "meta.json").write_text(json.dumps({
    "source": f"demo: {args.date} · {args.column.removeprefix('feature_').removeprefix('target_')} · {args.ticker}",
    "rows": rows.height,
    "tickers": tickers,
    "country": per_t["feature_country"].to_list(),
    "first_seen": per_t["first"].to_list(),
    "dates": dates,
    "date_type": date_type,
    "features": features,
    "targets": targets,
    "columns": [{"name": c, "kind": kind[c], **({"min": lo[c], "max": hi[c]} if c in lo else {})} for c in columns],
    "demo": {"date": d0, "col": columns.index(args.column), "tick": t0, "rowsA": A.height, "rowsB": B.height},
}, separators=(",", ":")))
for f in sorted(OUT.iterdir()):
    print(f"{f.name}: {f.stat().st_size / 1e6:.2f} MB")
print(f"rows {rows.height:,}  A {A.height}  B {B.height}  C {int(in_c.sum()):,}  values {len(values_bin) / 1e6:.1f} MB raw")
