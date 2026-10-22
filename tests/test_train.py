"""Tests for the split and the metrics in src/train.py. No database needed."""

import numpy as np
import pandas as pd
import pytest

from features import TARGET, TIME_COLUMN
from train import TEST_START, VALID_START, evaluate, time_split


def test_time_split_keeps_the_future_out_of_training():
    times = pd.to_datetime(["2017-06-01", "2018-02-28 23:59", VALID_START,
                            "2018-04-30 23:59", TEST_START, "2018-08-01"], format="ISO8601")
    df = pd.DataFrame({TIME_COLUMN: times, TARGET: 0})

    train, valid, test = time_split(df)

    assert len(train) + len(valid) + len(test) == len(df)  # every order used once
    assert train[TIME_COLUMN].max() < pd.Timestamp(VALID_START)
    assert valid[TIME_COLUMN].min() == pd.Timestamp(VALID_START)
    assert valid[TIME_COLUMN].max() < pd.Timestamp(TEST_START)
    assert test[TIME_COLUMN].min() == pd.Timestamp(TEST_START)


def test_evaluate_top_fraction_by_hand():
    # 10 orders, 2 late. Flagging the top 20% = the 2 highest scores,
    # which are one late order (0.9) and one on-time order (0.8).
    y = [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    scores = [0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2]

    result = evaluate(y, scores, top_fraction=0.2)

    assert result["late_rate"] == pytest.approx(0.2)
    assert result["precision_top20%"] == pytest.approx(0.5)  # 1 of 2 flagged is late
    assert result["recall_top20%"] == pytest.approx(0.5)     # 1 of 2 late orders caught


def test_evaluate_perfect_and_random_scores():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.07).astype(int)

    perfect = evaluate(y, y.astype(float))
    random = evaluate(y, rng.random(len(y)))

    assert perfect["avg_precision"] == pytest.approx(1.0)
    # A random ranking scores about the late rate (why we compare to 0.046).
    assert random["avg_precision"] == pytest.approx(y.mean(), abs=0.02)
