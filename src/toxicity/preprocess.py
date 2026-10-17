"""Text cleaning, data-quality checks and class weights for the Jigsaw data.

Two cleaning modes, because the two model families need different input:

* ``light``  for transformers. Keeps case, punctuation and emoji (the tokenizer
  and the model use them). Only removes noise: HTML, URLs, IPs, repeated
  characters, extra whitespace.
* ``heavy``  for TF-IDF / linear models. Also lowercases, undoes leetspeak
  (``sh1t``) and masked profanity (``f.u.c.k``), expands contractions and drops
  punctuation.

Both modes keep the text close to what the user wrote, so word-level
explanations (LIME / SHAP / attention) still point at real words.
"""

from __future__ import annotations

import html
import re
import unicodedata

import numpy as np
import pandas as pd

from .config import ID_COL, LABELS, TEXT_COL

# ---------------------------------------------------------------- regexes
_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_WIKI_USER = re.compile(r"\[\[User:[^\]|]*(?:\|[^\]]*)?\]\]", re.I)
_HTML_TAG = re.compile(r"<[^>]+>")
_REPEAT_CHAR = re.compile(r"(.)\1{2,}")        # sooooo -> soo
_REPEAT_PUNCT = re.compile(r"([!?.,])\1+")      # !!!! -> !
_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9\s']")

# Only tokens that mix letters with digits/symbols are de-leeted (sh1t, a$$), so plain numbers stay.
_LEET_TOKEN = re.compile(r"(?=\S*[a-z])(?=\S*[0-9@$])\S+")
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})

# "n't" is handled by the explicit pairs first so "won't" -> "will not".
_CONTRACTIONS = [
    ("won't", "will not"), ("can't", "cannot"), ("n't", " not"),
    ("'re", " are"), ("'ve", " have"), ("'ll", " will"), ("'d", " would"), ("'m", " am"),
]

# f.u.c.k / f u c k / fuuuck -> fuck. Letters may be separated by punctuation or spaces.
_SEP = r"[\W_]*"
_MASKED = [
    (re.compile(_SEP.join(f"{c}+" for c in word), re.I), word)
    for word in ("fuck", "shit", "bitch", "asshole", "bastard", "cunt", "dick", "slut", "whore")
]

MODES = ("light", "heavy")


def _base_clean(text: object) -> str:
    text = "" if text is None else str(text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = _HTML_TAG.sub(" ", text)
    text = text.replace("\\n", " ")                  # literal backslash-n present in the data
    text = _WIKI_USER.sub(" USER ", text)
    text = _URL.sub(" URL ", text)
    text = _EMAIL.sub(" EMAIL ", text)
    text = _IP.sub(" IPADDR ", text)
    return text


def clean_text(text: object, mode: str = "light") -> str:
    """Clean one comment. ``mode`` is ``"light"`` or ``"heavy"``."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    text = _base_clean(text)
    if mode == "light":
        text = _REPEAT_PUNCT.sub(r"\1", text)
        text = _REPEAT_CHAR.sub(r"\1\1", text)
        return _WS.sub(" ", text).strip()

    for pattern, word in _MASKED:                    # before lowercasing / leet mapping
        text = pattern.sub(f" {word} ", text)
    text = _LEET_TOKEN.sub(lambda m: m.group(0).translate(_LEET), text.lower())
    for old, new in _CONTRACTIONS:
        text = text.replace(old, new)
    text = _REPEAT_CHAR.sub(r"\1\1", text)
    text = _NON_ALNUM.sub(" ", text)
    return _WS.sub(" ", text).strip()


def clean_series(texts: pd.Series, mode: str = "light") -> pd.Series:
    return texts.map(lambda t: clean_text(t, mode))


# ---------------------------------------------------------------- features
def meta_features(texts: pd.Series) -> pd.DataFrame:
    """Hand-made signals computed on RAW text (case and punctuation still present)."""
    s = texts.fillna("").astype(str)
    n_chars = s.str.len().clip(lower=1)
    words = s.str.split()
    return pd.DataFrame(
        {
            "n_chars": s.str.len(),
            "n_words": words.str.len(),
            "caps_ratio": s.map(lambda x: sum(c.isupper() for c in x)) / n_chars,
            "n_exclaim": s.str.count("!"),
            "n_question": s.str.count(r"\?"),
            "has_url": s.str.contains(_URL).astype(int),
            "unique_word_ratio": words.map(lambda w: len(set(x.lower() for x in w)) / max(len(w), 1)),
        },
        index=texts.index,
    )


# ---------------------------------------------------------------- quality
def quality_report(df: pd.DataFrame, text_col: str = TEXT_COL) -> dict:
    """Counts of problems worth knowing about before training."""
    labels = df[LABELS]
    dup_mask = df[text_col].duplicated(keep=False)
    # Same text but different labels = label noise.
    conflicting = (
        df[dup_mask].groupby(text_col)[LABELS].nunique().gt(1).any(axis=1).sum() if dup_mask.any() else 0
    )
    return {
        "rows": int(len(df)),
        "empty_text": int((df[text_col].str.strip() == "").sum()),
        "duplicate_rows": int(df[text_col].duplicated().sum()),
        "duplicate_texts_with_conflicting_labels": int(conflicting),
        "rows_with_no_label": int((labels.sum(axis=1) == 0).sum()),
        "rows_with_any_label": int((labels.sum(axis=1) > 0).sum()),
        "max_chars": int(df[text_col].str.len().max()),
        "median_words": float(df[text_col].str.split().str.len().median()),
    }


def leakage(train: pd.DataFrame, other: pd.DataFrame, text_col: str = TEXT_COL) -> int:
    """How many rows of ``other`` have a text that also appears in ``train``."""
    return int(other[text_col].isin(set(train[text_col])).sum())


def drop_duplicates_resolving_labels(df: pd.DataFrame, text_col: str = TEXT_COL) -> pd.DataFrame:
    """Keep one row per text; if copies disagree, a label is 1 when ANY copy has it."""
    if not df[text_col].duplicated().any():
        return df.reset_index(drop=True)
    agg = {ID_COL: "first", **{label: "max" for label in LABELS}}
    return df.groupby(text_col, sort=False, as_index=False).agg(agg)[df.columns.tolist()]


# ---------------------------------------------------------------- weights
def compute_pos_weights(
    df: pd.DataFrame, power: float = 0.5, max_weight: float | None = None
) -> dict[str, float]:
    """pos_weight = (negatives / positives) ** power, per label (config: train.pos_weight_power).

    power=1.0 is full inverse frequency; 0.5 is a softer, usually more stable choice.
    """
    pos = df[LABELS].sum().clip(lower=1)
    neg = len(df) - df[LABELS].sum()
    weights = (neg / pos) ** power
    if max_weight is not None:
        weights = weights.clip(upper=max_weight)
    return {label: float(round(weights[label], 4)) for label in LABELS}
