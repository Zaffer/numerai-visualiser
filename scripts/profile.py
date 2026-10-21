"""Print a basic numerical profile of the Numerai v6.0 (Unity) dataset."""
import json
from pathlib import Path

import polars as pl

DATA = Path(__file__).resolve().parent.parent / "data" / "v6.0"
meta = json.loads((DATA / "features.json").read_text())
FEATURES = meta["feature_sets"]["all"]
SIGNALS = [f for f in meta["feature_sets"]["signals"] if f != "feature_country"]
OBFUSCATED = [f for f in FEATURES if f not in meta["feature_sets"]["signals"]]
TARGETS = meta["targets"]


def section(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def profile_split(name):
    path = DATA / f"{name}.parquet"
    lf = pl.scan_parquet(path)
    schema = lf.collect_schema()
    section(f"{name}  ({path.stat().st_size / 1e9:.2f} GB on disk)")
    print(f"columns: {len(schema)}")
    print("dtypes:", dict(pl.Series([str(t) for t in schema.values()]).value_counts().iter_rows()))
    non_feat = [c for c in schema if not c.startswith(("feature_", "target"))]
    print("sample ids:", lf.select(non_feat).head(3).collect().rows())
    if "data_type" in schema: print("data_type:", dict(lf.group_by("data_type").len().collect().iter_rows()))
    print("non-feature/target columns:", non_feat)

    df = lf.select("numerai_ticker", "date", "feature_country").collect()
    print(f"rows: {df.height:,}")
    print(f"unique tickers: {df['numerai_ticker'].n_unique():,}")
    dates = df["date"].unique().sort()
    print(f"dates: {len(dates):,}  ({dates.min()} -> {dates.max()})")
    if len(dates) > 1:
        gaps = dates.str.to_date().diff().drop_nulls().dt.total_days()
        print("date gap (days) counts:", dict(gaps.value_counts().sort("count", descending=True).head(5).iter_rows()))
    per_date = df.group_by("date").len()["len"]
    print(f"tickers per date: min {per_date.min():,}  median {per_date.median():,.0f}  max {per_date.max():,}")
    if len(dates) > 1:
        by_year = (df.group_by("date").len()
                   .with_columns(pl.col("date").str.to_date().dt.year().alias("year"))
                   .group_by("year").agg(pl.col("len").mean().round(0).alias("mean_tickers_per_date"), pl.len().alias("dates"))
                   .sort("year"))
        print(by_year)
        life = df.group_by("numerai_ticker").len()["len"]
        print(f"dates per ticker: min {life.min()}  median {life.median():.0f}  max {life.max()}")
    countries = df["feature_country"].value_counts().sort("count", descending=True)
    print(f"countries: {countries.height}  top 10:")
    print(countries.head(10).with_columns((pl.col("count") / df.height * 100).round(1).alias("pct")))

    # Feature values / nulls (one pass over numeric features)
    num_feats = [f for f in FEATURES if f in schema and f != "feature_country"]
    obf = [f for f in OBFUSCATED if f in schema]
    stats = lf.select(
        *[pl.col(f).null_count().alias(f) for f in num_feats]
    ).collect().transpose(include_header=True, column_names=["nulls"])
    stats = stats.with_columns((pl.col("nulls") / df.height * 100).round(2).alias("null_pct"))
    print(f"numeric features: {len(num_feats)}  ({len(obf)} obfuscated, {len([s for s in SIGNALS if s in schema])} signals)")
    print(f"features with any nulls: {(stats['nulls'] > 0).sum()}")
    print(stats.filter(pl.col("nulls") > 0).sort("null_pct", descending=True).head(25))
    sample_cols = obf[:: max(1, len(obf) // 50)]
    vals = lf.select(sample_cols).collect().unpivot()["value"].value_counts().sort("value")
    print(f"obfuscated feature value counts (sample of {len(sample_cols)} cols):")
    print(vals.with_columns((pl.col("count") / pl.col("count").sum() * 100).round(2).alias("pct")))
    sig = lf.select([s for s in SIGNALS if s in schema]).collect()
    print("signals features: describe")
    with pl.Config(tbl_cols=-1, tbl_width_chars=250):
        print(sig.describe().transpose(include_header=True, header_name="feature", column_names="statistic").select("feature", "null_count", "mean", "std", "min", "50%", "max"))

    tgts = [t for t in TARGETS if t in schema]
    if tgts:
        t = lf.select(tgts).collect()
        tstats = pl.DataFrame({
            "target": tgts,
            "null_pct": [round(t[c].null_count() / t.height * 100, 2) for c in tgts],
            "n_unique": [t[c].drop_nulls().n_unique() for c in tgts],
            "values": [str(sorted(t[c].drop_nulls().unique().to_list())[:6]) for c in tgts],
            "mean": [round(t[c].mean() or float("nan"), 4) for c in tgts],
        })
        with pl.Config(tbl_rows=-1, fmt_str_lengths=60):
            print(tstats)
        print("target == target_jupiter_60:", t.select((pl.col("target") == pl.col("target_jupiter_60")).all()).item())
        main = (lf.select("date", "target").collect().group_by("date")
                .agg(pl.col("target").is_null().mean().alias("null_frac")).sort("date"))
        null_dates = main.filter(pl.col("null_frac") > 0)
        print(f"dates with null `target`: {null_dates.height}", null_dates.tail(15) if null_dates.height else "")
    return df


if __name__ == "__main__":
    ids = {s: profile_split(s) for s in ["train", "validation", "live"]}
    section("overlap")
    tick = {s: set(d["numerai_ticker"].unique()) for s, d in ids.items()}
    print(f"tickers train&val: {len(tick['train'] & tick['validation']):,}  val&live: {len(tick['validation'] & tick['live']):,}  live not in train/val: {len(tick['live'] - tick['train'] - tick['validation']):,}")
    print(f"train last date {ids['train']['date'].max()}  val first date {ids['validation']['date'].min()}")
    for b in ["train_benchmark_models", "validation_benchmark_models"]:
        lf = pl.scan_parquet(DATA / f"{b}.parquet")
        cols = lf.collect_schema().names()
        section(b)
        print(f"rows: {lf.select(pl.len()).collect().item():,}  columns ({len(cols)}): {cols}")
