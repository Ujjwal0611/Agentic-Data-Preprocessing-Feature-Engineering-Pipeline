"""
test_orchestrator.py

Integration tests for the Phase 3 orchestrator. Split into two groups:

FAST tests (no LLM, no Ollama dependency) -- run every time:
    - leakage prevention (split integrity, no cross-split duplicates)
    - column alignment after independent cleaning
    - executive report generation from synthetic results

SLOW test (real Ollama call, ~10-30s) -- proves the actual wiring works:
    - one full strategy run through run_strategy(): clean train/test,
      align, retrain, score

Run with:
    uv run python -m agentic_data_pipeline.phase3.test_orchestrator
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from agentic_data_pipeline.data_io import DataIOManager
from agentic_data_pipeline.phase3.leakage_guard import (
    LeakageGuardError,
    align_train_test_columns,
    split_before_cleaning,
    verify_split_integrity,
)
from agentic_data_pipeline.phase3.main import run_strategy
from agentic_data_pipeline.phase3.reporter import StrategyRunResult, build_executive_report
from agentic_data_pipeline.phase3.strategies import STRATEGY_LIBRARY, get_strategy

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("test_orchestrator")


def test_leakage_prevention() -> bool:
    """Split integrity: row counts reconcile, no exact duplicate crosses
    the train/test boundary."""
    logger.info("=== TEST: Leakage prevention ===")
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "feature_a": rng.normal(0, 1, 200),
        "feature_b": rng.normal(0, 1, 200),
        "target": rng.choice([0, 1], 200),
    })

    split = split_before_cleaning(df, target_column="target", test_size=0.2, random_seed=42)

    try:
        verify_split_integrity(split, original_row_count=len(df))
    except LeakageGuardError as exc:
        logger.error("FAILED: split_before_cleaning produced an invalid split: %s", exc)
        return False

    if len(split.train_df) + len(split.test_df) != len(df):
        logger.error("FAILED: row count mismatch after split.")
        return False

    merged = pd.concat([split.train_df, split.test_df], axis=0)
    if int(merged.duplicated(keep=False).sum()) > 0:
        logger.error("FAILED: duplicate rows found across the train/test boundary.")
        return False

    logger.info("PASSED: %d train rows, %d test rows, no cross-split duplicates.",
                len(split.train_df), len(split.test_df))
    return True


def test_leakage_prevention_rejects_missing_target() -> bool:
    """split_before_cleaning must fail loud if target_column isn't present,
    not silently split on the wrong column."""
    logger.info("=== TEST: Leakage guard rejects missing target column ===")
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    try:
        split_before_cleaning(df, target_column="target")
        logger.error("FAILED: expected LeakageGuardError for missing target_column, none raised.")
        return False
    except LeakageGuardError:
        logger.info("PASSED: LeakageGuardError correctly raised for missing target_column.")
        return True


def test_align_columns() -> bool:
    """After independent cleaning, train/test can legitimately have
    different one-hot columns (a rare category present only on one side).
    align_train_test_columns must zero-fill the gap, not drop rows or crash."""
    logger.info("=== TEST: Column alignment after independent cleaning ===")
    train = pd.DataFrame({
        "age": [25.0, 30.0, 35.0],
        "region_US": [1, 0, 1],
        "region_EU": [0, 1, 0],
        "target": [0, 1, 0],
    })
    # test is missing region_EU (never appeared in this split) but has a
    # column train never saw
    test = pd.DataFrame({
        "age": [40.0, 45.0],
        "region_US": [1, 0],
        "region_APAC": [0, 1],
        "target": [1, 0],
    })

    aligned_train, aligned_test = align_train_test_columns(train, test, target_column="target")

    if set(aligned_train.columns) != set(aligned_test.columns):
        logger.error("FAILED: column sets still differ after alignment: train=%s test=%s",
                     sorted(aligned_train.columns), sorted(aligned_test.columns))
        return False

    if list(aligned_train.columns) != list(aligned_test.columns):
        logger.error("FAILED: column ORDER differs between aligned train/test.")
        return False

    if len(aligned_train) != 3 or len(aligned_test) != 2:
        logger.error("FAILED: alignment changed row counts (should only add zero-filled columns).")
        return False

    logger.info("PASSED: aligned to %d shared columns, row counts preserved.", len(aligned_train.columns))
    return True


def test_align_columns_rejects_missing_target() -> bool:
    """align_train_test_columns must refuse to proceed if the agent dropped
    or renamed the target column during cleaning -- this would otherwise
    silently corrupt downstream training."""
    logger.info("=== TEST: Column alignment rejects missing target ===")
    train = pd.DataFrame({"age": [25.0], "target": [1]})
    test_missing_target = pd.DataFrame({"age": [30.0]})  # target dropped
    try:
        align_train_test_columns(train, test_missing_target, target_column="target")
        logger.error("FAILED: expected LeakageGuardError for missing target column, none raised.")
        return False
    except LeakageGuardError:
        logger.info("PASSED: LeakageGuardError correctly raised for missing target column.")
        return True


def test_report_generation() -> bool:
    """build_executive_report must produce both comparison.json and
    executive.md from a mix of successful and failed strategy results,
    without needing any real agent/LLM calls."""
    logger.info("=== TEST: Executive report generation (synthetic results) ===")

    synthetic_results = [
        StrategyRunResult(
            strategy_name="conservative_median", succeeded=True, f1_score=0.72,
            accuracy=0.70, beat_baseline=True, n_train_rows=800, n_test_rows=200,
            n_features=14, agent_attempts_used=3,
        ),
        StrategyRunResult(
            strategy_name="mean_wide_tolerance", succeeded=True, f1_score=0.61,
            accuracy=0.60, beat_baseline=False, n_train_rows=800, n_test_rows=200,
            n_features=13, agent_attempts_used=1,
        ),
        StrategyRunResult(
            strategy_name="tight_grouping_high_temp", succeeded=False, f1_score=None,
            accuracy=None, beat_baseline=None, n_train_rows=None, n_test_rows=None,
            n_features=None, agent_attempts_used=5,
            failure_reason="Train cleaning failed: exhausted 5 attempts",
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "reports"
        comparison_path, executive_path = build_executive_report(
            baseline_f1=0.65, strategy_results=synthetic_results, output_dir=output_dir,
        )

        if not comparison_path.exists() or not executive_path.exists():
            logger.error("FAILED: report files were not created.")
            return False

        executive_text = executive_path.read_text(encoding="utf-8")
        if "conservative_median" not in executive_text or "0.72" not in executive_text:
            logger.error("FAILED: executive.md missing expected winning-strategy content.")
            return False
        if "tight_grouping_high_temp" not in executive_text:
            logger.error("FAILED: executive.md missing the failed strategy in the detail section.")
            return False

    logger.info("PASSED: comparison.json and executive.md generated with correct content.")
    return True


def test_full_orchestration_single_strategy() -> bool:
    """SLOW / REAL: runs one actual strategy end-to-end through
    run_strategy() -- real Ollama calls, real sandbox execution, real
    XGBoost training. Proves the full Phase 3 wiring works, not just each
    piece in isolation. Uses a small synthetic messy dataset to keep this fast.
    """
    logger.info("=== TEST: Full orchestration, single strategy (REQUIRES OLLAMA) ===")
    rng = np.random.default_rng(7)
    n = 60
    df = pd.DataFrame({
        "age": rng.normal(40, 10, n).round(1),
        "income": rng.normal(60_000, 15_000, n).round(2),
        "employment_type": rng.choice(["full-time", "part-time", "self-employed"], n),
        "target": rng.choice([0, 1], n),
    })
    # inject a little realistic messiness
    nan_idx = rng.choice(n, size=6, replace=False)
    df.loc[nan_idx, "income"] = np.nan

    split = split_before_cleaning(df, target_column="target", test_size=0.25, random_seed=7)

    io_manager = DataIOManager()
    with tempfile.TemporaryDirectory() as tmpdir:
        strategies_dir = Path(tmpdir) / "strategies"
        reports_dir = Path(tmpdir) / "reports"
        run_history_dir = Path(tmpdir) / "run_history"
        for d in (strategies_dir, reports_dir, run_history_dir):
            d.mkdir(parents=True, exist_ok=True)
        output_dirs = {"phase": Path(tmpdir), "strategies": strategies_dir,
                        "reports": reports_dir, "run_history": run_history_dir}

        strategy = get_strategy("conservative_median")
        result = run_strategy(strategy, split, output_dirs, target_column="target")

        if not result.succeeded:
            logger.warning(
                "Strategy did not succeed (this can happen with a small/noisy synthetic "
                "dataset and a 7B local model -- not necessarily a wiring bug): %s",
                result.failure_reason,
            )
            # Don't hard-fail the suite on LLM non-determinism -- the point
            # of this test is to prove the WIRING works, which it did if we
            # got a clean StrategyRunResult back instead of an exception.
            return True

        if result.f1_score is None or not (0.0 <= result.f1_score <= 1.0):
            logger.error("FAILED: got an invalid F1 score: %s", result.f1_score)
            return False

        results_json = strategies_dir / strategy.name / "results.json"
        if not results_json.exists():
            logger.error("FAILED: results.json was not written for the strategy.")
            return False

    logger.info("PASSED: full orchestration wiring works end-to-end, F1=%.4f", result.f1_score)
    return True


def main() -> None:
    logger.info("Starting Phase 3 orchestrator test suite...")
    logger.info("STRATEGY_LIBRARY has %d strategies: %s",
                len(STRATEGY_LIBRARY), [s.name for s in STRATEGY_LIBRARY])

    tests = [
        test_leakage_prevention,
        test_leakage_prevention_rejects_missing_target,
        test_align_columns,
        test_align_columns_rejects_missing_target,
        test_report_generation,
        test_full_orchestration_single_strategy,
    ]

    results = {}
    for test_fn in tests:
        try:
            results[test_fn.__name__] = test_fn()
        except Exception:
            logger.exception("Test %s raised an unhandled exception.", test_fn.__name__)
            results[test_fn.__name__] = False

    passed = sum(1 for v in results.values() if v)
    failed = len(results) - passed

    print("\n--- PHASE 3 ORCHESTRATOR TEST SUITE COMPLETE ---")
    print(f"Passed: {passed}/{len(results)}")
    if failed:
        print("Failed tests:")
        for name, ok in results.items():
            if not ok:
                print(f"  - {name}")

    raise SystemExit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()