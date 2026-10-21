"""Train and evaluate late-delivery models on a time-based split.

Usage:
    uv run python src/train.py

Decisions:
    * Time-based split (notebook 02: lateness comes in bursts, and the model
      will score future orders). Train on the past, validate on the next
      months, test once on the most recent months.
    * Main metric: average precision (area under the precision-recall curve).
      Only ~7% of orders are late, so accuracy is useless and ROC-AUC looks
      flattering. A model that guesses at random scores the late rate.
    * Ops view: precision and recall when the top 10% riskiest orders are
      flagged, assuming ops can only act on a limited number of orders.
"""

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


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by purchase time into train, validation and test."""
    t = df[TIME_COLUMN]
    train = df[t < VALID_START]
    valid = df[(t >= VALID_START) & (t < TEST_START)]
    test = df[t >= TEST_START]
    return train, valid, test


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

    logistic = build_logistic().fit(train[FEATURES], train[TARGET])
    results["logistic regression"] = evaluate(test[TARGET], logistic.predict_proba(test[FEATURES])[:, 1])

    lgbm = fit_lightgbm(train, valid)
    print(f"LightGBM stopped at {lgbm.best_iteration_} trees")
    results["lightgbm"] = evaluate(test[TARGET], lgbm.predict_proba(test[FEATURES])[:, 1])

    print("\nTest set (May - Aug 2018):")
    print(pd.DataFrame(results).T.round(3).to_string())


if __name__ == "__main__":
    main()
