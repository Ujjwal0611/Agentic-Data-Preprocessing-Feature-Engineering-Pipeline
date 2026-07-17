"""
test_leakage_guard.py

Run with:
    uv run python -m agentic_data_pipeline.test_leakage_guard
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from agentic_data_pipeline.phase3.leakage_guard  import (
    LeakageGuardError,
    align_train_test_columns,
    split_before_cleaning,
    verify_split_integrity,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_leakage_guard")


def _make_raw_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "age": rng.normal(40, 10, n),
        "income": rng.normal(60000, 15000, n),
        "target": rng.integers(0, 2, n),
    })


def test_split_is_disjoint_and_complete():
    logger.info("=== TEST: split produces disjoint, complete train/test ===")
    raw = _make_raw_df(200)
    result = split_before_cleaning(raw, target_column="target", test_size=0.25, random_seed=1)
    assert len(result.train_df) + len(result.test_df) == 200
    verify_split_integrity(result, original_row_count=200)
    logger.info("PASSED: %d train / %d test, no overlap.", len(result.train_df), len(result.test_df))


def test_missing_target_column_raises():
    logger.info("=== TEST: missing target column raises LeakageGuardError ===")
    raw = _make_raw_df(50).drop(columns=["target"])
    try:
        split_before_cleaning(raw, target_column="target")
        logger.warning("FAILED: expected LeakageGuardError, none raised.")
    except LeakageGuardError:
        logger.info("PASSED: LeakageGuardError raised as expected.")


def test_cross_split_duplicate_detection():
    logger.info("=== TEST: manually-forged cross-split duplicate is detected ===")
    from agentic_data_pipeline.phase3.leakage_guard  import SplitResult
    dup_row = pd.DataFrame({"age": [40.0], "income": [60000.0], "target": [1]})
    train = pd.concat([dup_row, dup_row], ignore_index=True)  # 2 rows, includes dup_row twice
    test = pd.concat([dup_row], ignore_index=True)            # same row again on the "other side"
    bad_split = SplitResult(train_df=train, test_df=test, target_column="target", random_seed=1)
    try:
        verify_split_integrity(bad_split, original_row_count=3)
        logger.warning("FAILED: expected LeakageGuardError for cross-split duplicate.")
    except LeakageGuardError as exc:
        logger.info("PASSED: correctly detected cross-split contamination: %s", exc)


def test_align_columns_after_independent_cleaning():
    logger.info("=== TEST: column alignment after independently-cleaned splits diverge ===")
    cleaned_train = pd.DataFrame({
        "age": [1.0, 2.0], "employment_full-time": [1, 0], "target": [0, 1],
    })
    cleaned_test = pd.DataFrame({
        "age": [3.0], "employment_part-time": [1], "target": [1],
    })
    aligned_train, aligned_test = align_train_test_columns(cleaned_train, cleaned_test, "target")
    assert set(aligned_train.columns) == set(aligned_test.columns)
    assert aligned_test["employment_full-time"].iloc[0] == 0
    assert aligned_train["employment_part-time"].iloc[0] == 0
    logger.info("PASSED: columns aligned, gaps zero-filled. Final columns: %s", list(aligned_train.columns))


def test_align_missing_target_raises():
    logger.info("=== TEST: missing target column during alignment raises ===")
    cleaned_train = pd.DataFrame({"age": [1.0], "target": [0]})
    cleaned_test = pd.DataFrame({"age": [2.0]})  # target dropped by a buggy cleaning attempt
    try:
        align_train_test_columns(cleaned_train, cleaned_test, "target")
        logger.warning("FAILED: expected LeakageGuardError, none raised.")
    except LeakageGuardError:
        logger.info("PASSED: LeakageGuardError raised as expected.")


def main():
    logger.info("Starting leakage_guard test suite...")
    test_split_is_disjoint_and_complete()
    test_missing_target_column_raises()
    test_cross_split_duplicate_detection()
    test_align_columns_after_independent_cleaning()
    test_align_missing_target_raises()
    logger.info("leakage_guard test suite complete.")


if __name__ == "__main__":
    main()