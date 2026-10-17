"""Clean the split files, check data quality and compute class weights.

Reads train/val/test(/train_subset_N).csv from data/processed (written by
03_make_splits.py) and writes, next to them:

    <name>_clean.csv       original columns + text_light + text_heavy (+ meta features if --meta)
    quality_report.json    empties, duplicates, label conflicts, train->val/test leakage
    pos_weights.json       per-label pos_weight for weighted BCE

    python scripts/03b_preprocess.py
    python scripts/03b_preprocess.py --meta
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from toxicity import load_config
from toxicity.config import TEXT_COL
from toxicity.preprocess import (
    clean_series,
    compute_pos_weights,
    drop_duplicates_resolving_labels,
    leakage,
    meta_features,
    quality_report,
)


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default=cfg["data"]["processed_dir"])
    parser.add_argument("--meta", action="store_true", help="Also write hand-made meta features")
    parser.add_argument("--dedupe-train", action="store_true", help="Drop duplicate texts in train files")
    args = parser.parse_args()
    folder = Path(args.dir)

    files = sorted(p for p in folder.glob("*.csv") if not p.stem.endswith("_clean") and p.stem != "split_stats")
    if not files:
        raise SystemExit(f"No split files in {folder}. Run scripts/03_make_splits.py first.")

    raw = {p.stem: pd.read_csv(p, keep_default_na=False) for p in files}
    report = {"quality": {name: quality_report(df) for name, df in raw.items()}}
    if "train" in raw:
        report["leakage_rows_in_train"] = {
            name: leakage(raw["train"], df) for name, df in raw.items() if name != "train" and not name.startswith("train_")
        }

    for name, df in raw.items():
        if args.dedupe_train and name.startswith("train"):
            df = drop_duplicates_resolving_labels(df)
        out = df.copy()
        out["text_light"] = clean_series(df[TEXT_COL], "light")
        out["text_heavy"] = clean_series(df[TEXT_COL], "heavy")
        if args.meta:
            out = pd.concat([out, meta_features(df[TEXT_COL])], axis=1)
        out.to_csv(folder / f"{name}_clean.csv", index=False)
        print(f"{name}: {len(out):,} rows -> {name}_clean.csv")

    if "train" in raw:
        tc = cfg["train"]
        report["pos_weights"] = compute_pos_weights(raw["train"], tc["pos_weight_power"], tc["pos_weight_max"])
        (folder / "pos_weights.json").write_text(json.dumps(report["pos_weights"], indent=2))
    (folder / "quality_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "quality"}, indent=2))
    print(f"\nFull report: {folder / 'quality_report.json'}")


if __name__ == "__main__":
    main()
