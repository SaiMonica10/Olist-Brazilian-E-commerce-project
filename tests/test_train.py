"""Tests for the split and the metrics in src/train.py. No database needed."""

import numpy as np
import pandas as pd
import pytest

from features import TARGET, TIME_COLUMN
from train import TEST_START, VALID_START, backtest_splits, evaluate, simple_rules, time_split


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


def test_backtest_splits_never_train_on_the_future():
    times = pd.date_range("2017-01-01", "2018-08-31", freq="D")
    df = pd.DataFrame({TIME_COLUMN: times, TARGET: 0})

    splits = list(backtest_splits(df))

    assert len(splits) == 5
    for _, train, valid, test in splits:
        assert train[TIME_COLUMN].max() < valid[TIME_COLUMN].min()
        assert valid[TIME_COLUMN].max() < test[TIME_COLUMN].min()
    # The last window is the main split.
    _, train, valid, test = splits[-1]
    main = time_split(df)
    assert [len(train), len(valid), len(test)] == [len(part) for part in main]


def _orders(states, promised_days, late):
    return pd.DataFrame({"customer_state": pd.Categorical(states, categories=["SP", "RJ", "AL"]),
                         "promised_days": promised_days, TARGET: late})


def test_simple_rules_by_hand():
    # SP: 1 of 4 late (25%), all with long promises. RJ: 2 of 2 late (100%), short promises.
    train = _orders(["SP"] * 4 + ["RJ"] * 2, [30, 30, 30, 30, 10, 10], [1, 0, 0, 0, 1, 1])
    test = _orders(["SP", "RJ", "AL", "SP"], [30, 10, 20, 10], [0, 0, 0, 0])

    rules = simple_rules(train, test)

    assert list(rules["rule: shortest promise"]) == [-30, -10, -20, -10]
    # AL never appears in training, so it gets the overall late rate (3 of 6).
    assert list(rules["rule: riskiest state"]) == pytest.approx([0.25, 1.0, 0.5, 0.25])
    # SP with a short promise was never seen, so it falls back to SP's rate.
    assert list(rules["rule: state x promise table"]) == pytest.approx([0.25, 1.0, 0.5, 0.25])


def test_simple_rules_ignore_test_outcomes():
    train = _orders(["SP", "RJ"], [30, 10], [0, 1])
    test = _orders(["SP", "RJ"], [30, 10], [0, 0])

    before = simple_rules(train, test)
    after = simple_rules(train, test.assign(**{TARGET: [1, 1]}))

    for name in before:
        pd.testing.assert_series_equal(before[name], after[name])
