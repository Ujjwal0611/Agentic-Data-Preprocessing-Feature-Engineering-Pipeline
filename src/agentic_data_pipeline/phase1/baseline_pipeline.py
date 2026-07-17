"""
baseline_pipeline.py

Phase 1 of the Agentic Data Preprocessing & Feature Engineering Pipeline.

Generates a deliberately messy synthetic classification dataset and trains
a naive XGBoost baseline on it, using only a minimal fail-safe (dropping
string columns) rather than real preprocessing. This establishes the
"before" F1-score that our Phase 2 local-LLM agent must beat.

Run with:
    uv run python -m agentic_data_pipeline.baseline_pipeline
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("baseline_pipeline")

RANDOM_SEED: int = 42
N_ROWS: int = 1000
NAN_FRACTION: float = 0.12
OUTLIER_FRACTION: float = 0.03
INCOME_OUTLIER_MULTIPLIER: float = 15.0
AGE_OUTLIER_MULTIPLIER: float = 5.0
TEST_SIZE: float = 0.2

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
RAW_CSV_PATH = DATA_DIR / "raw_messy_dataset.csv"
METRICS_JSON_PATH = DATA_DIR / "metrics.json"


@dataclass(frozen=True)
class BaselineMetrics:
    """Typed container so downstream code (and tomorrow's agent) has a
    guaranteed, documented shape to read metrics from."""

    baseline_f1_score: float
    baseline_accuracy: float
    n_train_rows: int
    n_test_rows: int
    n_features_used: int
    columns_dropped_by_failsafe: list[str]

    def to_dict(self) -> dict:
        return {
            "baseline_f1_score": self.baseline_f1_score,
            "baseline_accuracy": self.baseline_accuracy,
            "n_train_rows": self.n_train_rows,
            "n_test_rows": self.n_test_rows,
            "n_features_used": self.n_features_used,
            "columns_dropped_by_failsafe": self.columns_dropped_by_failsafe,
        }


def generate_messy_dataset(n_rows: int = N_ROWS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate a synthetic 'loan approval' dataset with deliberately injected
    messiness: missing values, outliers, fuzzy-spelled categories, duplicate
    rows, and illogical cross-column date orderings.

    Args:
        n_rows: number of rows to generate.
        seed: seed for reproducibility.

    Returns:
        A messy DataFrame with columns: age, income, tenure_years, credit_score,
        employment_type, application_date, approval_date, target.
    """
    rng = np.random.default_rng(seed)

    age = rng.normal(loc=40, scale=12, size=n_rows).round(1)
    income = rng.normal(loc=65_000, scale=18_000, size=n_rows).round(2)
    tenure_years = rng.exponential(scale=4, size=n_rows).round(2)
    credit_score = rng.normal(loc=680, scale=55, size=n_rows).round(0)

    # --- Fuzzy-spelled categorical: same 5 real categories, multiple spellings ---
    clean_types = rng.choice(
        ["full-time", "part-time", "self-employed", "unemployed", "contract"],
        size=n_rows,
        p=[0.45, 0.15, 0.15, 0.10, 0.15],
    )
    spelling_variants: dict[str, list[str]] = {
        "full-time": ["full-time", "Full-Time", "FULL TIME", "full time ", "fulltime"],
        "part-time": ["part-time", "Part-Time", "part time"],
        "self-employed": ["self-employed", "Self Employed", "self employed"],
        "unemployed": ["unemployed", "Unemployed"],
        "contract": ["contract", "Contract "],
    }
    employment_type = np.array(
        [rng.choice(spelling_variants[t]) for t in clean_types]
    )

    # Build a genuine, learnable linear signal
    linear_signal = (
        0.00003 * income
        + 0.05 * tenure_years
        + 0.01 * credit_score
        - 0.02 * age
    )
    noise = rng.normal(loc=0, scale=1.5, size=n_rows)
    probability = 1 / (1 + np.exp(-(linear_signal - linear_signal.mean() + noise)))
    target = (probability > np.median(probability)).astype(int)

    # --- Cross-column date logic: application_date should precede approval_date ---
    base_dates = pd.Timestamp("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 500, n_rows), unit="D"
    )
    approval_offset_days = rng.integers(1, 20, n_rows)
    approval_dates = base_dates + pd.to_timedelta(approval_offset_days, unit="D")
    # Inject ~5% of rows where approval comes before application (data error)
    date_logic_bug_mask = rng.random(n_rows) < 0.05
    approval_dates_reported = approval_dates.where(
        ~date_logic_bug_mask,
        base_dates - pd.to_timedelta(rng.integers(1, 10, n_rows), unit="D"),
    )

    df = pd.DataFrame(
        {
            "age": age,
            "income": income,
            "tenure_years": tenure_years,
            "credit_score": credit_score,
            "employment_type": employment_type,
            "application_date": base_dates.strftime("%Y-%m-%d"),
            "approval_date": approval_dates_reported.strftime("%Y-%m-%d"),
            "target": target,
        }
    )

    # --- Inject missing values (NaNs) ---
    numeric_cols_for_nans = ["age", "income", "tenure_years", "credit_score"]
    n_nans_per_col = int(n_rows * NAN_FRACTION)
    for col in numeric_cols_for_nans:
        nan_indices = rng.choice(n_rows, size=n_nans_per_col, replace=False)
        df.loc[nan_indices, col] = np.nan

    # --- Inject unscaled outliers ---
    n_outliers = int(n_rows * OUTLIER_FRACTION)
    income_outlier_idx = rng.choice(n_rows, size=n_outliers, replace=False)
    df.loc[income_outlier_idx, "income"] = (
        df.loc[income_outlier_idx, "income"] * INCOME_OUTLIER_MULTIPLIER
    )
    age_outlier_idx = rng.choice(n_rows, size=n_outliers, replace=False)
    df.loc[age_outlier_idx, "age"] = df.loc[age_outlier_idx, "age"] * AGE_OUTLIER_MULTIPLIER

    # --- Duplicate rows: append ~2% exact duplicates ---
    n_duplicates = int(n_rows * 0.02)
    duplicate_rows = df.sample(n=n_duplicates, random_state=seed)
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    logger.info(
        "Generated messy dataset: %d rows (incl. %d duplicates), "
        "%d NaNs/col, %d outliers/col, ~5%% date-logic bug.",
        len(df), n_duplicates, n_nans_per_col, n_outliers,
    )
    return df


def baseline_failsafe_drop_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal fail-safe so the naive baseline doesn't crash on text columns.

    This is intentionally NOT real preprocessing — it exists only so Phase 1
    can produce a runnable 'before' score. Phase 2's LLM agent replaces this
    with real encoding logic.
    """
    string_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    string_cols = [c for c in string_cols if c != "target"]
    if string_cols:
        logger.warning(
            "Fail-safe triggered: dropping unencoded string column(s) %s. "
            "This is a placeholder -- Phase 2's agent must do better.",
            string_cols,
        )
        df = df.drop(columns=string_cols)
    return df


def train_and_evaluate_baseline(df: pd.DataFrame) -> BaselineMetrics:
    """Train a naive XGBoost baseline on the (fail-safed) messy data."""
    if "target" not in df.columns:
        raise ValueError("Expected a 'target' column in the input DataFrame.")

    working_df = baseline_failsafe_drop_strings(df.copy())

    dropped_cols = [c for c in df.columns if c not in working_df.columns]

    X = working_df.drop(columns=["target"])
    y = working_df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    model = XGBClassifier(
        random_state=RANDOM_SEED,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = BaselineMetrics(
        baseline_f1_score=round(float(f1_score(y_test, predictions)), 4),
        baseline_accuracy=round(float(accuracy_score(y_test, predictions)), 4),
        n_train_rows=int(len(X_train)),
        n_test_rows=int(len(X_test)),
        n_features_used=int(X.shape[1]),
        columns_dropped_by_failsafe=dropped_cols,
    )
    logger.info("Baseline results: %s", metrics.to_dict())
    return metrics


def export_metrics(metrics: BaselineMetrics, path: Path = METRICS_JSON_PATH) -> None:
    """Write metrics to disk as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    logger.info("Metrics exported to %s", path)


def main() -> None:
    """Entry point: generate data, save raw CSV, train baseline, export metrics."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = generate_messy_dataset()
    df.to_csv(RAW_CSV_PATH, index=False)
    logger.info("Raw messy dataset saved to %s", RAW_CSV_PATH)

    metrics = train_and_evaluate_baseline(df)
    export_metrics(metrics)

    print("\n--- PHASE 1 BASELINE COMPLETE ---")
    print(json.dumps(metrics.to_dict(), indent=2))


if __name__ == "__main__":
    main()