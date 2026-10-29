"""Train and evaluate late-delivery models on a time-based split.

Usage:
    uv run python src/train.py

Decisions:
    * Time-based split (notebook 02: lateness comes in bursts, and the model
      will score future orders). Train on the past, validate on the next
      months, test once on the most recent months.
    * No gap between the splits. Training orders bought just before a cutoff
      were delivered after it (6.4% of train), so their outcome was not yet
      known at the cutoff. Checked: a 30-day gap gives the same test average
      precision (0.093 vs 0.094), so the gap is left out.
    * Known limitation: early stopping watches Mar-Apr 2018, a lateness burst
      (11.8% late vs 6.6% in train), so the number of trees it picks is fragile.
    * Main metric: average precision (area under the precision-recall curve).
      Only ~7% of orders are late, so accuracy is useless and ROC-AUC looks
      flattering. A model that guesses at random scores the late rate.
    * Ops view: precision and recall when the top 10% riskiest orders are
      flagged, assuming ops can only act on a limited number of orders.
    * The LightGBM model is saved to models/ for the API. It is the model the
      test scores describe (trained before Mar 2018), not a refit on all data.
    * Simple rules as baselines: rules ops could follow without a model, built
      from the two top SHAP reasons (promised days, customer state). Their
      late-rate tables are learned from train only, the same data LightGBM
      learns from.
    * Walk-forward backtest: one test window can mislead (lateness comes in
      bursts), so every method is also scored on five consecutive windows.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, TIME_COLUMN, load_features

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Purchases before VALID_START train the model; VALID_START..TEST_START tune it;
# TEST_START onwards is touched once, for the final score.
VALID_START = "2018-03-01"
TEST_START = "2018-05-01"

TOP_FRACTION = 0.10  # share of orders ops can review

# Backtest windows as (valid_start, test_start, test_end): train on everything
# before valid_start, early-stop on valid, score on test. The last window is
# the main split above; 2018-09-01 is just after the last purchase.
BACKTEST_WINDOWS = [
    ("2017-07-01", "2017-09-01", "2017-11-01"),
    ("2017-09-01", "2017-11-01", "2018-01-01"),
    ("2017-11-01", "2018-01-01", "2018-03-01"),
    ("2018-01-01", "2018-03-01", "2018-05-01"),
    (VALID_START, TEST_START, "2018-09-01"),
]

# Promised-days buckets for the state x promise rule (edges are inclusive on the right).
PROMISE_BUCKETS = [-1, 14, 21, 28, 35, np.inf]

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "late_risk_lgbm.joblib"


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by purchase time into train, validation and test."""
    t = df[TIME_COLUMN]
    train = df[t < VALID_START]
    valid = df[(t >= VALID_START) & (t < TEST_START)]
    test = df[t >= TEST_START]
    return train, valid, test


def backtest_splits(df: pd.DataFrame):
    """Yield (window label, train, valid, test) for each backtest window."""
    t = df[TIME_COLUMN]
    for valid_start, test_start, test_end in BACKTEST_WINDOWS:
        label = f"{test_start[:7]}..{pd.Timestamp(test_end) - pd.Timedelta(days=1):%Y-%m}"
        yield (label, df[t < valid_start], df[(t >= valid_start) & (t < test_start)],
               df[(t >= test_start) & (t < test_end)])


def simple_rules(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, pd.Series]:
    """Risk scores from rules of thumb (higher = riskier), learned from train only."""
    def with_bucket(d):
        return d.assign(promise_bucket=pd.cut(d["promised_days"], PROMISE_BUCKETS))

    train, test = with_bucket(train), with_bucket(test)

    def past_late_rate(columns):
        rates = train.groupby(columns, observed=True)[TARGET].mean().rename("rate")
        return test.join(rates, on=columns)["rate"].astype(float)

    # A state or state x bucket not seen in training falls back to the broader rate.
    state = past_late_rate(["customer_state"]).fillna(train[TARGET].mean())
    return {
        "rule: shortest promise": -test["promised_days"],
        "rule: riskiest state": state,
        "rule: state x promise table": past_late_rate(["customer_state", "promise_bucket"]).fillna(state),
    }


def backtest(df: pd.DataFrame) -> pd.DataFrame:
    """Score the simple rules and LightGBM on every backtest window."""
    rows = []
    for window, train, valid, test in backtest_splits(df):
        scores = simple_rules(train, test)
        scores["lightgbm"] = fit_lightgbm(train, valid).predict_proba(test[FEATURES])[:, 1]
        for method, s in scores.items():
            rows.append({"window": window, "method": method, **evaluate(test[TARGET], s)})
    result = pd.DataFrame(rows)
    result["ap_vs_random"] = result["avg_precision"] / result["late_rate"]
    return result


def evaluate(y_true, scores, top_fraction: float = TOP_FRACTION) -> dict:
    """Ranking metrics, plus precision/recall when the top-scored orders are flagged."""
    y_true = np.asarray(y_true)
    n_flagged = int(round(top_fraction * len(y_true)))
    flagged = np.argsort(-np.asarray(scores))[:n_flagged]
    caught = y_true[flagged].sum()
    return {
        "orders": len(y_true),
        "late_rate": y_true.mean(),
        "avg_precision": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
        f"precision_top{top_fraction:.0%}": caught / n_flagged,
        f"recall_top{top_fraction:.0%}": caught / y_true.sum(),
    }


def build_logistic():
    """Baseline: logistic regression. Needs missing values filled, categories
    one-hot encoded, and numbers on a common scale."""
    preprocess = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                              StandardScaler()), NUMERIC_FEATURES),
        # Missing categories become their own column; categories with fewer
        # than 50 training orders are grouped as "infrequent".
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50),
         CATEGORICAL_FEATURES),
    ])
    return make_pipeline(preprocess, LogisticRegression(max_iter=2000))


def build_lightgbm():
    """Gradient-boosted trees. Handles missing values and pandas categories
    natively, so no preprocessing is needed."""
    from lightgbm import LGBMClassifier  # imported here: needs libomp on macOS

    return LGBMClassifier(
        n_estimators=2000,       # upper limit; early stopping picks the real number
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=50,    # each leaf needs 50+ orders, against overfitting
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )


def fit_lightgbm(train: pd.DataFrame, valid: pd.DataFrame):
    """Fit LightGBM, stopping when validation average precision stops improving."""
    from lightgbm import early_stopping

    model = build_lightgbm()
    model.fit(train[FEATURES], train[TARGET],
              eval_X=valid[FEATURES], eval_y=valid[TARGET],
              eval_metric="average_precision",
              callbacks=[early_stopping(100, verbose=False)])
    return model


def main() -> None:
    df = load_features()
    train, valid, test = time_split(df)
    for name, part in [("train", train), ("valid", valid), ("test", test)]:
        print(f"{name:5s}: {len(part):6,} orders, {part[TARGET].mean():.1%} late, "
              f"{part[TIME_COLUMN].min():%Y-%m} to {part[TIME_COLUMN].max():%Y-%m}")

    results = {"random guess": evaluate(test[TARGET], np.random.default_rng(42).random(len(test)))}
    for name, scores in simple_rules(train, test).items():
        results[name] = evaluate(test[TARGET], scores)

    logistic = build_logistic().fit(train[FEATURES], train[TARGET])
    results["logistic regression"] = evaluate(test[TARGET], logistic.predict_proba(test[FEATURES])[:, 1])

    lgbm = fit_lightgbm(train, valid)
    print(f"LightGBM stopped at {lgbm.best_iteration_} trees")
    results["lightgbm"] = evaluate(test[TARGET], lgbm.predict_proba(test[FEATURES])[:, 1])

    print("\nTest set (May - Aug 2018):")
    print(pd.DataFrame(results).T.round(3).to_string())

    print("\nBacktest (each column is a test window; mean is across windows):")
    bt = backtest(df)
    for metric in ["ap_vs_random", f"recall_top{TOP_FRACTION:.0%}"]:
        table = bt.pivot(index="method", columns="window", values=metric)
        table["mean"] = table.mean(axis=1)
        print(f"\n{metric}:\n{table.round(3).to_string()}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(lgbm, MODEL_PATH)
    print(f"\nSaved LightGBM model to {MODEL_PATH.relative_to(MODEL_PATH.parent.parent)}")


if __name__ == "__main__":
    main()
