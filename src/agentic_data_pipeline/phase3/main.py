"""
main.py

Phase 3 orchestrator: the outer state machine that cycles through every
strategy in STRATEGY_LIBRARY, cleans train/test INDEPENDENTLY under each one
via the Phase 2 self-correcting agent, aligns any one-hot column divergence,
retrains XGBoost, and tracks F1 against the Phase 1 baseline. Every strategy
is attempted (not stopped early on first success) so the executive report
can show a genuine comparison, not just "the first thing that worked."

Run with:
    uv run python -m agentic_data_pipeline.phase3.main
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

from agentic_data_pipeline.data_io import DataIOManager
from agentic_data_pipeline.phase2.agent import run_self_correcting_cleaning
from agentic_data_pipeline.phase2.error_reporter import (
    ErrorReporter,
    FailureCategory,
    RemediationSuggestion,
)
from agentic_data_pipeline.phase3.leakage_guard import (
    LeakageGuardError,
    SplitResult,
    align_train_test_columns,
    assert_cleaned_independently,
    split_before_cleaning,
)
from agentic_data_pipeline.phase3.reporter import StrategyRunResult, build_executive_report
from agentic_data_pipeline.phase3.strategies import STRATEGY_LIBRARY, CleaningStrategy

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("phase3_main")

# parents[0]=phase3, [1]=agentic_data_pipeline, [2]=src, [3]=project root
DATA_DIR = Path(__file__).resolve().parents[3] / "data"
RAW_CSV_PATH = DATA_DIR / "raw_messy_dataset.csv"
METRICS_JSON_PATH = DATA_DIR / "metrics.json"
TARGET_COLUMN = "target"
MAX_AGENT_RETRIES = 5

error_reporter = ErrorReporter(run_name="phase3_orchestrator")
io_manager = DataIOManager()


def _load_baseline_f1() -> float:
    """Read Phase 1's baseline_f1_score from data/metrics.json."""
    if not METRICS_JSON_PATH.exists():
        raise FileNotFoundError(
            f"{METRICS_JSON_PATH} not found. Run Phase 1's baseline_pipeline first: "
            f"uv run python -m agentic_data_pipeline.phase1.baseline_pipeline"
        )
    with METRICS_JSON_PATH.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    return float(metrics["baseline_f1_score"])


def _train_and_score(
    cleaned_train: pd.DataFrame, cleaned_test: pd.DataFrame, target_column: str
) -> tuple[float, float]:
    """Train XGBoost on cleaned_train, evaluate on cleaned_test.

    Returns:
        (f1_score, accuracy), both rounded to 4 decimal places.
    """
    X_train = cleaned_train.drop(columns=[target_column])
    y_train = cleaned_train[target_column]
    X_test = cleaned_test.drop(columns=[target_column])
    y_test = cleaned_test[target_column]

    model = XGBClassifier(random_state=42, eval_metric="logloss")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return (
        round(float(f1_score(y_test, predictions)), 4),
        round(float(accuracy_score(y_test, predictions)), 4),
    )


def run_strategy(
    strategy: CleaningStrategy,
    split: SplitResult,
    output_dirs: dict,
    target_column: str = TARGET_COLUMN,
) -> StrategyRunResult:
    """Run ONE cleaning strategy end-to-end. Never raises -- every failure
    mode (train cleaning fails, test cleaning fails, column alignment fails)
    is captured into a StrategyRunResult so the outer loop can continue to
    the next strategy instead of crashing the whole run.
    """
    logger.info("=== STRATEGY: %s ===", strategy.name)
    logger.info("%s", strategy.description)
    strategy_dir = output_dirs["strategies"] / strategy.name
    strategy_dir.mkdir(parents=True, exist_ok=True)

    # Clean train and test INDEPENDENTLY -- this is the whole point of
    # splitting before cleaning (see leakage_guard.py module docstring).
    train_result = run_self_correcting_cleaning(
        split.train_df,
        target_column=target_column,
        max_retries=MAX_AGENT_RETRIES,
        strategy=strategy,
    )
    if not train_result.succeeded:
        logger.warning(
            "Strategy '%s' failed cleaning TRAIN split: %s", strategy.name, train_result.failure_reason
        )
        error_reporter.report_failure(
            test_name=f"Phase 3 strategy: {strategy.name} (train split)",
            dataset_name=f"phase3_{strategy.name}_train",
            n_rows=len(split.train_df),
            attempts_used=train_result.attempts_used,
            max_attempts=MAX_AGENT_RETRIES,
            failure_category=FailureCategory.AGENT_EXHAUSTION,
            final_error_type="Unknown",
            final_error_message=train_result.failure_reason or "Unknown",
            remediation_suggestions=[RemediationSuggestion.RETRY_WITH_DIFFERENT_MODEL],
            human_notes=f"Strategy '{strategy.name}' could not clean the train split.",
        )
        return StrategyRunResult(
            strategy_name=strategy.name, succeeded=False, f1_score=None, accuracy=None,
            beat_baseline=None, n_train_rows=None, n_test_rows=None, n_features=None,
            agent_attempts_used=train_result.attempts_used,
            failure_reason=f"Train cleaning failed: {train_result.failure_reason}",
        )

    test_result = run_self_correcting_cleaning(
        split.test_df,
        target_column=target_column,
        max_retries=MAX_AGENT_RETRIES,
        strategy=strategy,
    )
    total_attempts = train_result.attempts_used + test_result.attempts_used
    if not test_result.succeeded:
        logger.warning(
            "Strategy '%s' failed cleaning TEST split: %s", strategy.name, test_result.failure_reason
        )
        error_reporter.report_failure(
            test_name=f"Phase 3 strategy: {strategy.name} (test split)",
            dataset_name=f"phase3_{strategy.name}_test",
            n_rows=len(split.test_df),
            attempts_used=test_result.attempts_used,
            max_attempts=MAX_AGENT_RETRIES,
            failure_category=FailureCategory.AGENT_EXHAUSTION,
            final_error_type="Unknown",
            final_error_message=test_result.failure_reason or "Unknown",
            remediation_suggestions=[RemediationSuggestion.RETRY_WITH_DIFFERENT_MODEL],
            human_notes=f"Strategy '{strategy.name}' could not clean the test split.",
        )
        return StrategyRunResult(
            strategy_name=strategy.name, succeeded=False, f1_score=None, accuracy=None,
            beat_baseline=None, n_train_rows=None, n_test_rows=None, n_features=None,
            agent_attempts_used=total_attempts,
            failure_reason=f"Test cleaning failed: {test_result.failure_reason}",
        )

    # Cheap guard against a future refactor accidentally re-merging
    # train/test before this point (see leakage_guard.py docstring).
    assert_cleaned_independently(train_result.cleaned_df, test_result.cleaned_df)

    try:
        aligned_train, aligned_test = align_train_test_columns(
            train_result.cleaned_df, test_result.cleaned_df, target_column=target_column
        )
    except LeakageGuardError as exc:
        logger.error("Strategy '%s' failed column alignment: %s", strategy.name, exc)
        error_reporter.report_failure(
            test_name=f"Phase 3 strategy: {strategy.name} (column alignment)",
            dataset_name=f"phase3_{strategy.name}",
            n_rows=len(split.train_df) + len(split.test_df),
            attempts_used=total_attempts,
            max_attempts=MAX_AGENT_RETRIES,
            failure_category=FailureCategory.SANDBOX_RESTRICTION,
            final_error_type="LeakageGuardError",
            final_error_message=str(exc),
        )
        return StrategyRunResult(
            strategy_name=strategy.name, succeeded=False, f1_score=None, accuracy=None,
            beat_baseline=None, n_train_rows=None, n_test_rows=None, n_features=None,
            agent_attempts_used=total_attempts,
            failure_reason=f"Column alignment failed: {exc}",
        )

    aligned_train.to_csv(strategy_dir / "train_cleaned.csv", index=False)
    aligned_test.to_csv(strategy_dir / "test_cleaned.csv", index=False)

    f1, accuracy = _train_and_score(aligned_train, aligned_test, target_column)

    result = StrategyRunResult(
        strategy_name=strategy.name,
        succeeded=True,
        f1_score=f1,
        accuracy=accuracy,
        beat_baseline=None,  # filled in by main() once baseline is loaded
        n_train_rows=len(aligned_train),
        n_test_rows=len(aligned_test),
        n_features=aligned_train.shape[1] - 1,  # exclude target
        agent_attempts_used=total_attempts,
    )

    with (strategy_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)

    logger.info("Strategy '%s' scored F1=%.4f, accuracy=%.4f", strategy.name, f1, accuracy)
    return result


def main() -> None:
    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(
            f"{RAW_CSV_PATH} not found. Run Phase 1's baseline_pipeline first: "
            f"uv run python -m agentic_data_pipeline.phase1.baseline_pipeline"
        )

    baseline_f1 = _load_baseline_f1()
    logger.info("Baseline F1 to beat: %.4f", baseline_f1)

    raw_df = pd.read_csv(RAW_CSV_PATH)
    n_before = len(raw_df)
    raw_df = raw_df.drop_duplicates(keep="first").reset_index(drop=True)
    logger.info("Deduplication: %d -> %d rows before split.", n_before, len(raw_df))
    split = split_before_cleaning(raw_df, target_column=TARGET_COLUMN)

    phase3_out = DATA_DIR / "phase3_outputs"
    output_dirs = {
        "strategies": phase3_out / "strategies",
        "reports": phase3_out / "reports",
    }
    for d in output_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    strategy_results: list[StrategyRunResult] = []
    for strategy in STRATEGY_LIBRARY:
        result = run_strategy(strategy, split, output_dirs, target_column=TARGET_COLUMN)

        if result.succeeded and result.f1_score is not None:
            # StrategyRunResult is frozen -- rebuild with beat_baseline filled in
            result = StrategyRunResult(**{**asdict(result), "beat_baseline": result.f1_score >= baseline_f1})

        strategy_results.append(result)
        io_manager.append_run_history("phase3", result.to_dict())

    error_reporter.export_all()
    comparison_path, executive_path = build_executive_report(
        baseline_f1=baseline_f1,
        strategy_results=strategy_results,
        output_dir=output_dirs["reports"],
    )

    logger.info("Phase 3 complete.")
    print("\n--- PHASE 3 COMPLETE ---")
    print(f"Strategies attempted: {len(strategy_results)}")
    print(f"Strategies succeeded: {sum(1 for r in strategy_results if r.succeeded)}")
    print(f"Executive report: {executive_path}")
    print(f"Comparison JSON: {comparison_path}")


if __name__ == "__main__":
    main()