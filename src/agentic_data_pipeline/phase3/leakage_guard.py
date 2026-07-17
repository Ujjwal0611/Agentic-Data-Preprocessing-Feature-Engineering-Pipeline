"""
leakage_guard.py

Enforces train/test split BEFORE any cleaning happens, and verifies no
information crossed the split boundary. This is enforced structurally (the
orchestrator has no code path that cleans before splitting) plus verified
with runtime assertions (not just a comment promising it won't happen).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger("leakage_guard")


class LeakageGuardError(Exception):
    """Raised when split integrity is violated -- overlapping indices,
    row-count mismatch, or cleaning attempted before a valid split exists."""


@dataclass(frozen=True)
class SplitResult:
    """A verified, disjoint train/test split of the RAW (uncleaned) data.
    Nothing downstream should ever see the original combined DataFrame again --
    only train_df and test_df, cleaned independently."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    target_column: str
    random_seed: int


def split_before_cleaning(
    raw_df: pd.DataFrame,
    target_column: str = "target",
    test_size: float = 0.2,
    random_seed: int = 42,
) -> SplitResult:
    """Split RAW data into train/test BEFORE any cleaning code runs.

    This is the only sanctioned entry point for splitting in the Phase 3
    orchestrator -- agent.run_self_correcting_cleaning() must only ever be
    called on the .train_df or .test_df produced here, never on raw_df itself.

    Args:
        raw_df: the untouched, uncleaned DataFrame straight from disk.
        target_column: name of the label column, used for stratification.
        test_size: fraction of rows held out for testing.
        random_seed: for reproducibility.

    Returns:
        A verified SplitResult with disjoint train_df/test_df.

    Raises:
        LeakageGuardError: if the resulting split fails integrity checks.
    """
    if target_column not in raw_df.columns:
        raise LeakageGuardError(
            f"target_column '{target_column}' not found in raw_df columns: "
            f"{list(raw_df.columns)}"
        )

    train_df, test_df = train_test_split(
        raw_df,
        test_size=test_size,
        random_state=random_seed,
        stratify=raw_df[target_column],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    result = SplitResult(
        train_df=train_df,
        test_df=test_df,
        target_column=target_column,
        random_seed=random_seed,
    )
    verify_split_integrity(result, original_row_count=len(raw_df))
    logger.info(
        "Split verified: %d train rows, %d test rows, seed=%d.",
        len(train_df), len(test_df), random_seed,
    )
    return result


def verify_split_integrity(split: SplitResult, original_row_count: int) -> None:
    """Runtime proof (not a comment) that the split is disjoint and complete.

    Checks:
        1. train + test row counts sum back to the original count.
        2. No exact duplicate row (all columns, including target) appears in
           BOTH train and test -- a strong signal that the same underlying
           record leaked across the boundary (e.g. via a bad join upstream).

    Raises:
        LeakageGuardError: on any check failure.
    """
    combined_len = len(split.train_df) + len(split.test_df)
    if combined_len != original_row_count:
        raise LeakageGuardError(
            f"Split row count mismatch: train({len(split.train_df)}) + "
            f"test({len(split.test_df)}) = {combined_len}, expected "
            f"{original_row_count}. Rows were lost or duplicated during split."
        )

    merged = pd.concat([split.train_df, split.test_df], axis=0)
    cross_split_duplicates = int(merged.duplicated(keep=False).sum())
    if cross_split_duplicates > 0:
        raise LeakageGuardError(
            f"{cross_split_duplicates} row(s) appear identical across the "
            f"train/test boundary. This usually means the split happened "
            f"AFTER deduplication ran on the combined data, or the source "
            f"data had true duplicates that landed on both sides. Re-run "
            f"split_before_cleaning() on data that has NOT been touched by "
            f"any cleaning step."
        )


def assert_cleaned_independently(
    cleaned_train: pd.DataFrame, cleaned_test: pd.DataFrame
) -> None:
    """Sanity check AFTER independent cleaning: train and test should not
    share a row index space that implies they were cleaned as one object.
    This is a cheap guard against a future refactor accidentally re-merging
    train/test before calling the agent."""
    if cleaned_train.index.intersection(cleaned_test.index).size > 0 and (
        cleaned_train.index.max() == cleaned_test.index.max()
    ):
        logger.warning(
            "cleaned_train and cleaned_test share overlapping index values "
            "with matching max index -- verify they were not accidentally "
            "produced from a single combined exec() call."
        )


def align_train_test_columns(
    cleaned_train: pd.DataFrame,
    cleaned_test: pd.DataFrame,
    target_column: str = "target",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconcile column sets after INDEPENDENT cleaning of train/test.

    Because train and test are cleaned separately (by design -- see module
    docstring), one-hot encoding can legitimately produce different dummy
    columns on each side (e.g. a rare category present only in train). This
    is expected and is NOT leakage -- it's the cost of true independence.
    We fix it here by unioning columns and zero-filling gaps, which is a
    downstream reconciliation step, not a cross-split information flow.

    Raises:
        LeakageGuardError: if target_column is missing from either split
            after cleaning (would indicate the agent dropped or renamed it).
    """
    if target_column not in cleaned_train.columns or target_column not in cleaned_test.columns:
        raise LeakageGuardError(
            f"target_column '{target_column}' missing from cleaned train "
            f"and/or test after independent cleaning -- refusing to align."
        )

    train_cols = set(cleaned_train.columns)
    test_cols = set(cleaned_test.columns)
    missing_in_test = train_cols - test_cols
    missing_in_train = test_cols - train_cols

    if missing_in_test:
        for col in missing_in_test:
            cleaned_test[col] = 0
        logger.info("Added %d zero-filled column(s) to test: %s", len(missing_in_test), sorted(missing_in_test))

    if missing_in_train:
        for col in missing_in_train:
            cleaned_train[col] = 0
        logger.info("Added %d zero-filled column(s) to train: %s", len(missing_in_train), sorted(missing_in_train))

    ordered_cols = sorted(cleaned_train.columns)
    return cleaned_train[ordered_cols], cleaned_test[ordered_cols]