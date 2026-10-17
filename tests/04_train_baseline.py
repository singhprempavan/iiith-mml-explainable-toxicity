"""Train the TF-IDF + logistic regression baseline (one binary model per label).

Uses the cleaned text from 03b_preprocess.py (`text_heavy` column) and the
settings in configs/default.yaml (`baseline:`). Writes to --out:
model.joblib, val_predictions.csv, test_predictions.csv, metrics_val.csv.
Feed that folder to scripts/06_tune_thresholds.py.

    python scripts/04_train_baseline.py --out runs/baseline_tfidf_lr
    python scripts/04_train_baseline.py --train train_subset_30000_clean.csv --out runs/quick
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

from toxicity import LABELS, load_config
from toxicity.metrics import evaluate
from toxicity.predictions import save_predictions

TEXT = "text_heavy"


def build_vectorizers(bc: dict) -> list[TfidfVectorizer]:
    vecs = [
        TfidfVectorizer(
            ngram_range=(1, 2), max_features=bc["word_max_features"], min_df=bc["min_df"],
            sublinear_tf=True, strip_accents="unicode",
        )
    ]
    if bc["use_char"]:
        vecs.append(
            TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 5), max_features=bc["char_max_features"],
                min_df=bc["min_df"], sublinear_tf=True,
            )
        )
    return vecs


def main() -> None:
    cfg = load_config()
    d = Path(cfg["data"]["processed_dir"])
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", default="train_clean.csv")
    parser.add_argument("--val", default="val_clean.csv")
    parser.add_argument("--test", default="test_clean.csv")
    parser.add_argument("--dir", default=str(d))
    parser.add_argument("--out", default="runs/baseline_tfidf_lr")
    args = parser.parse_args()

    read = lambda n: pd.read_csv(Path(args.dir) / n, keep_default_na=False)
    train, val, test = read(args.train), read(args.val), read(args.test)
    bc = cfg["baseline"]

    vecs = build_vectorizers(bc)
    X_train = hstack([v.fit_transform(train[TEXT]) for v in vecs]).tocsr()
    transform = lambda df: hstack([v.transform(df[TEXT]) for v in vecs]).tocsr()

    clf = MultiOutputClassifier(
        LogisticRegression(C=bc["C"], class_weight=bc["class_weight"], solver="liblinear", max_iter=200),
        n_jobs=-1,
    ).fit(X_train, train[LABELS].to_numpy())

    proba = lambda X: np.column_stack([p[:, 1] for p in clf.predict_proba(X)])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    val_prob, test_prob = proba(transform(val)), proba(transform(test))
    save_predictions(out / "val_predictions.csv", val, val_prob)
    save_predictions(out / "test_predictions.csv", test, test_prob)
    joblib.dump({"vectorizers": vecs, "classifier": clf, "text_column": TEXT}, out / "model.joblib")

    metrics = evaluate(val[LABELS].to_numpy(), val_prob, 0.5)
    metrics.to_csv(out / "metrics_val.csv")
    print(metrics.round(4))
    print(f"\nSaved model and predictions to {out}/")


if __name__ == "__main__":
    main()
