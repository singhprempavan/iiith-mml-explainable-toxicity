import pandas as pd
import pytest

from toxicity import LABELS
from toxicity.preprocess import (
    clean_text, compute_pos_weights, drop_duplicates_resolving_labels,
    leakage, meta_features, quality_report,
)


def test_light_keeps_case_and_removes_noise():
    out = clean_text("YOU are SOOOOO bad!!!! see http://x.com <b>now</b>\\n ok", "light")
    assert "YOU" in out and "SOOO" not in out and "!!" not in out
    assert "URL" in out and "<b>" not in out and "\\n" not in out


def test_heavy_lowercases_and_fixes_obfuscation():
    assert "shit" in clean_text("this is sh1t", "heavy")
    assert "fuck" in clean_text("F.U.C.K you", "heavy")
    assert clean_text("You won't!!", "heavy") == "you will not"


def test_heavy_keeps_plain_numbers():
    out = clean_text("in 2010 he was a$$ and sh1t", "heavy")
    assert "2010" in out and "ass" in out and "shit" in out


def test_masks_wiki_user_ip_email():
    out = clean_text("[[User:Bob]] at 192.168.0.1 mail a@b.com", "light")
    assert out == "USER at IPADDR mail EMAIL"


def test_bad_mode_and_non_string_input():
    with pytest.raises(ValueError):
        clean_text("x", "medium")
    assert clean_text(None) == "" or isinstance(clean_text(None), str)
    assert clean_text(123, "light") == "123"


def _df(rows):
    return pd.DataFrame(rows, columns=["id", "comment_text", *LABELS])


def test_quality_and_conflicts():
    df = _df([["1", "a b", 1, 0, 0, 0, 0, 0], ["2", "a b", 0, 0, 0, 0, 0, 0], ["3", "", 0, 0, 0, 0, 0, 0]])
    rep = quality_report(df)
    assert rep["duplicate_rows"] == 1 and rep["duplicate_texts_with_conflicting_labels"] == 1
    assert rep["empty_text"] == 1


def test_dedupe_takes_max_label():
    df = _df([["1", "a b", 1, 0, 0, 0, 0, 0], ["2", "a b", 0, 0, 0, 1, 0, 0]])
    out = drop_duplicates_resolving_labels(df)
    assert len(out) == 1 and out.loc[0, "toxic"] == 1 and out.loc[0, "threat"] == 1


def test_leakage_count():
    a = _df([["1", "x", 0, 0, 0, 0, 0, 0]])
    b = _df([["2", "x", 0, 0, 0, 0, 0, 0], ["3", "y", 0, 0, 0, 0, 0, 0]])
    assert leakage(a, b) == 1


def test_pos_weights():
    df = _df([[str(i), "t", int(i == 0), 0, 0, 0, 0, 0] for i in range(5)])
    assert compute_pos_weights(df, power=1.0)["toxic"] == 4.0
    assert compute_pos_weights(df, power=1.0, max_weight=2.0)["toxic"] == 2.0
    assert compute_pos_weights(df, power=0.5)["threat"] == pytest.approx(5 ** 0.5, rel=1e-3)  # no positives -> clipped to 1


def test_meta_features():
    m = meta_features(pd.Series(["HELLO there!!", ""]))
    assert m.loc[0, "n_exclaim"] == 2 and m.loc[0, "caps_ratio"] > 0.3 and m.loc[1, "n_words"] == 0
