"""
test_edge_cases_reporting.py - updated with error_reporter integration
                                + organized DataIOManager output structure

Edge case testing for the agent, focused on genuine data DEFECTS the agent
must handle. Captures all failures to structured error reports for human
review (via ErrorReporter, unchanged) AND writes a per-test, per-run
organized record of every test (pass or fail) via DataIOManager, so results
can be inspected without digging through logs.

Run with:
    uv run python -m agentic_data_pipeline.phase2.test_edge_cases_reporting

Output (new, in addition to whatever ErrorReporter already writes):
    data/phase2_outputs/
    ├── edge_cases/{summary.json, summary.md, test_<id>.json, ...}
    ├── attempt_logs/{test_id}/{attempt_N.txt, metadata.json}
    └── run_metadata/run_YYYYMMDD_HHMMSS.json
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from agentic_data_pipeline.data_io import DataIOManager,TestResult
from agentic_data_pipeline.phase2.agent import run_self_correcting_cleaning
from agentic_data_pipeline.phase2.error_reporter import (
    ErrorReporter,
    FailureCategory,
    RemediationSuggestion,
)
from agentic_data_pipeline.phase2.profiler import profile_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("edge_case_tests")

# Initialize error reporter (unchanged - still the system of record for
# human-facing failure reports)
error_reporter = ErrorReporter(run_name="edge_case_suite")

# New: organized output manager. Creates data/phase2_outputs/{edge_cases,
# attempt_logs, run_metadata}/ at import time, same as error_reporter above.
io_manager = DataIOManager()
output_dirs = io_manager.create_test_run_dirs("phase2")


def _assert_fully_numeric_except_target(df: pd.DataFrame, target_column: str = "target") -> bool:
    """Verify every column except the target is numeric."""
    non_target_cols = [c for c in df.columns if c != target_column]
    non_numeric = [c for c in non_target_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        logger.warning("FAILED encoding check: non-numeric columns remain: %s", non_numeric)
        return False
    logger.info("PASSED encoding check: all %d non-target columns are numeric.", len(non_target_cols))
    return True


def _make_test_result(
    test_id: str,
    status: str,
    attempts: int,
    error_message: Optional[str] = None,
    error_category: Optional[str] = None,
    notes: Optional[str] = None,
) -> TestResult:
    """Build a TestResult for DataIOManager. duration_seconds filled in by caller."""
    return TestResult(
        test_id=test_id,
        status=status,
        attempts=attempts,
        duration_seconds=0.0,
        error_message=error_message,
        error_category=error_category,
        notes=notes,
    )


def test_all_nulls_column() -> tuple[TestResult, list[str]]:
    """A column that is entirely NaN - agent should drop it or leave it inert."""
    logger.info("=== TEST: All-NaN column ===")
    df = pd.DataFrame({
        "valid_col": [1.0, 2.0, 3.0, 4.0, 5.0],
        "all_nan_col": [np.nan] * 5,
        "employment": ["full-time"] * 5,
        "target": [0, 1, 0, 1, 0],
    })
    result = run_self_correcting_cleaning(df)
    logger.info("Result: success=%s, attempts=%d", result.succeeded, result.attempts_used)

    if not result.succeeded:
        error_reporter.report_failure(
            test_name="All-NaN column",
            dataset_name="edge_case_all_nan_5row",
            n_rows=len(df),
            attempts_used=result.attempts_used,
            max_attempts=5,
            failure_category=FailureCategory.AGENT_EXHAUSTION,
            final_error_type=type(result.failure_reason).__name__ if result.failure_reason else "Unknown",
            final_error_message=result.failure_reason or "Unknown",
            remediation_suggestions=[RemediationSuggestion.INSPECT_DATA],
            human_notes="LLM struggled with all-NaN column handling.",
        )
        return (
            _make_test_result(
                "all_nan_column", "FAIL", result.attempts_used,
                error_message=str(result.failure_reason or "Unknown"),
                error_category=FailureCategory.AGENT_EXHAUSTION.name,
            )
        )

    logger.info("Cleaned shape: %s", result.cleaned_df.shape)
    _assert_fully_numeric_except_target(result.cleaned_df)
    return (
        _make_test_result(
            "all_nan_column", "PASS", result.attempts_used,
            notes=f"Cleaned shape: {result.cleaned_df.shape}",
        )
    )


def test_single_row() -> tuple[TestResult, list[str]]:
    """A dataset with only 1 row - degenerate statistics (IQR=0, etc.)."""
    logger.info("=== TEST: Single-row dataset ===")
    df = pd.DataFrame({
        "age": [35.0], "income": [75000.0], "tenure_years": [5.0],
        "credit_score": [720.0], "employment_type": ["full-time"],
        "application_date": ["2024-01-15"], "approval_date": ["2024-01-20"],
        "target": [1],
    })
    result = run_self_correcting_cleaning(df)
    logger.info("Result: success=%s, attempts=%d", result.succeeded, result.attempts_used)

    if not result.succeeded:
        error_reporter.report_failure(
            test_name="Single-row dataset",
            dataset_name="edge_case_single_row_1row",
            n_rows=len(df),
            attempts_used=result.attempts_used,
            max_attempts=5,
            failure_category=FailureCategory.AGENT_EXHAUSTION,
            final_error_type="Unknown",
            final_error_message=result.failure_reason or "Unknown",
        )
        return (
            _make_test_result(
                "single_row", "FAIL", result.attempts_used,
                error_message=str(result.failure_reason or "Unknown"),
                error_category=FailureCategory.AGENT_EXHAUSTION.name,
            )
        )

    return (
        _make_test_result("single_row", "PASS", result.attempts_used)
        
    )


def test_all_duplicates() -> tuple[TestResult, list[str]]:
    """Every row is a duplicate - agent should collapse to 1 unique row."""
    logger.info("=== TEST: All-duplicate rows ===")
    base_row = {
        "age": [35.0], "income": [75000.0], "tenure_years": [5.0],
        "credit_score": [720.0], "employment_type": ["full-time"],
        "application_date": ["2024-01-15"], "approval_date": ["2024-01-20"],
        "target": [1],
    }
    df = pd.DataFrame({k: v * 100 for k, v in base_row.items()})
    result = run_self_correcting_cleaning(df)
    logger.info("Result: success=%s, attempts=%d", result.succeeded, result.attempts_used)

    if not result.succeeded:
        error_reporter.report_failure(
            test_name="All-duplicate rows",
            dataset_name="edge_case_all_dupes_100row",
            n_rows=len(df),
            attempts_used=result.attempts_used,
            max_attempts=5,
            failure_category=FailureCategory.AGENT_EXHAUSTION,
            final_error_type="Unknown",
            final_error_message=result.failure_reason or "Unknown",
        )
        return (
            _make_test_result(
                "all_duplicates", "FAIL", result.attempts_used,
                error_message=str(result.failure_reason or "Unknown"),
                error_category=FailureCategory.AGENT_EXHAUSTION.name,
            )
        )

    logger.info("After drop_duplicates, shape: %s (expected 1 row)", result.cleaned_df.shape)
    correctness_note = None
    if len(result.cleaned_df) != 1:
        logger.warning("FAILED: expected exactly 1 row after full deduplication, got %d",
                        len(result.cleaned_df))
        correctness_note = f"WARNING: expected 1 row after dedup, got {len(result.cleaned_df)}"
    return (
        _make_test_result(
            "all_duplicates", "PASS", result.attempts_used,
            notes=correctness_note or f"Cleaned shape: {result.cleaned_df.shape}",
        )
    )


def test_all_categorical() -> tuple[TestResult, list[str]]:
    """All columns are text - THE KEY TEST: agent must fully encode them."""
    logger.info("=== TEST: All-categorical (no numeric columns) ===")
    df = pd.DataFrame({
        "employment_type": ["full-time", "part-time", "self-employed", "unemployed"],
        "region": ["US", "EU", "APAC", "LATAM"],
        "status": ["active", "inactive", "pending", "active"],
        "target": [1, 0, 1, 0],
    })
    result = run_self_correcting_cleaning(df)
    logger.info("Result: success=%s, attempts=%d", result.succeeded, result.attempts_used)

    if not result.succeeded:
        error_reporter.report_failure(
            test_name="All-categorical",
            dataset_name="edge_case_all_categorical_4row",
            n_rows=len(df),
            attempts_used=result.attempts_used,
            max_attempts=5,
            failure_category=FailureCategory.AGENT_EXHAUSTION,
            final_error_type="Unknown",
            final_error_message=result.failure_reason or "Unknown",
        )
        return (
            _make_test_result(
                "all_categorical", "FAIL", result.attempts_used,
                error_message=str(result.failure_reason or "Unknown"),
                error_category=FailureCategory.AGENT_EXHAUSTION.name,
            )
        )

    logger.info("Cleaned dtypes:\n%s", result.cleaned_df.dtypes)
    encoding_ok = _assert_fully_numeric_except_target(result.cleaned_df)
    notes = None
    if not encoding_ok:
        logger.warning(
            "Agent reported success but LEFT TEXT COLUMNS UNENCODED. "
            "This means it 'succeeded' without actually doing its job."
        )
        notes = "WARNING: agent reported success but left text columns unencoded"
    return (
        _make_test_result(
            "all_categorical", "PASS" if encoding_ok else "FAIL", result.attempts_used,
            error_message=None if encoding_ok else "Text columns left unencoded despite reported success",
            error_category=None if encoding_ok else "SILENT_ENCODING_FAILURE",
            notes=notes,
        )
    )


def test_high_cardinality_category() -> tuple[TestResult, list[str]]:
    """Categorical column with >100 unique values."""
    logger.info("=== TEST: High-cardinality categorical ===")
    n = 500
    rng = np.random.default_rng(42)
    product_ids = [f"PROD-{i:04d}" for i in range(150)]
    df = pd.DataFrame({
        "age": rng.normal(40, 10, n),
        "income": rng.normal(65000, 15000, n),
        "product_id": rng.choice(product_ids, n),
        "target": rng.choice([0, 1], n),
    })
    logger.info("Unique product_ids: %d", df["product_id"].nunique())
    result = run_self_correcting_cleaning(df)
    logger.info("Result: success=%s, attempts=%d", result.succeeded, result.attempts_used)

    if not result.succeeded:
        error_reporter.report_failure(
            test_name="High-cardinality categorical",
            dataset_name="edge_case_high_card_500row",
            n_rows=len(df),
            attempts_used=result.attempts_used,
            max_attempts=5,
            failure_category=FailureCategory.AGENT_EXHAUSTION,
            final_error_type="Unknown",
            final_error_message=result.failure_reason or "Unknown",
            remediation_suggestions=[RemediationSuggestion.ADD_PREPROCESSING_RULE],
        )
        return (
            _make_test_result(
                "high_cardinality", "FAIL", result.attempts_used,
                error_message=str(result.failure_reason or "Unknown"),
                error_category=FailureCategory.AGENT_EXHAUSTION.name,
            )
        )

    logger.info("Cleaned shape: %s (watch column count - one-hot would add ~150)",
                 result.cleaned_df.shape)
    return (
        _make_test_result(
            "high_cardinality", "PASS", result.attempts_used,
            notes=f"Cleaned shape: {result.cleaned_df.shape}",
        )
    )


def test_date_parsing_failures() -> tuple[TestResult, list[str]]:
    """Date columns with inconsistent formats and unparseable values."""
    logger.info("=== TEST: Malformed dates ===")
    df = pd.DataFrame({
        "application_date": ["2024-01-15", "2024/02/20", "15-03-2024", "invalid", "2024-01-15"],
        "approval_date": ["2024-01-20", "2024/02/25", "20-03-2024", "2024-04-01", "2024-01-25"],
        "age": [30.0, 40.0, 35.0, 42.0, 38.0],
        "target": [1, 0, 1, 1, 0],
    })
    profile = profile_dataset(df, "target")
    logger.info("Date columns detected: %s", profile.date_column_candidates)
    result = run_self_correcting_cleaning(df)
    logger.info("Result: success=%s, attempts=%d", result.succeeded, result.attempts_used)

    if not result.succeeded:
        error_reporter.report_failure(
            test_name="Malformed dates",
            dataset_name="edge_case_malformed_dates_5row",
            n_rows=len(df),
            attempts_used=result.attempts_used,
            max_attempts=5,
            failure_category=FailureCategory.AMBIGUOUS_DATA,
            final_error_type="Unknown",
            final_error_message=result.failure_reason or "Unknown",
            remediation_suggestions=[
                RemediationSuggestion.LOWER_THRESHOLD,
                RemediationSuggestion.DOCUMENT_AS_KNOWN_LIMITATION,
            ]
        )
        return (
            _make_test_result(
                "malformed_dates", "FAIL", result.attempts_used,
                error_message=str(result.failure_reason or "Unknown"),
                error_category=FailureCategory.AMBIGUOUS_DATA.name,
                notes="Known limitation: 80% date parse rate is below the 90% threshold.",
            )
        )

    return (
        _make_test_result(
            "malformed_dates", "PASS", result.attempts_used,
            notes=f"Detected date columns: {profile.date_column_candidates}",
        )
    )


def test_mixed_null_types() -> tuple[TestResult, list[str]]:
    """NaN, NaT, None, and literal 'NA' strings all present."""
    logger.info("=== TEST: Mixed null representations ===")
    df = pd.DataFrame({
        "value_a": [1.0, np.nan, 3.0, None, 5.0],
        "value_b": ["A", "B", None, "D", "NA"],
        "value_c": [10, 20, 30, 40, 50],
        "target": [0, 1, 0, 1, 0],
    })
    logger.info("NaN per column:\n%s", df.isna().sum())
    result = run_self_correcting_cleaning(df)
    logger.info("Result: success=%s, attempts=%d", result.succeeded, result.attempts_used)

    if not result.succeeded:
        error_reporter.report_failure(
            test_name="Mixed null representations",
            dataset_name="edge_case_mixed_nulls_5row",
            n_rows=len(df),
            attempts_used=result.attempts_used,
            max_attempts=5,
            failure_category=FailureCategory.AGENT_EXHAUSTION,
            final_error_type="Unknown",
            final_error_message=result.failure_reason or "Unknown",
        )
        return (
            _make_test_result(
                "mixed_null_types", "FAIL", result.attempts_used,
                error_message=str(result.failure_reason or "Unknown"),
                error_category=FailureCategory.AGENT_EXHAUSTION.name,
            )
        )

    return (
        _make_test_result("mixed_null_types", "PASS", result.attempts_used)
    )


def test_full_pipeline_regression() -> tuple[Optional[TestResult], list[str]]:
    """Integration test: run the agent against the REAL Day 1 dataset."""
    logger.info("=== TEST: Full pipeline regression (real dataset) ===")
    raw_path = Path(__file__).resolve().parents[3] / "data" / "raw_messy_dataset.csv"
    if not raw_path.exists():
        logger.warning("Skipping: %s not found. Run baseline_pipeline.py first.", raw_path)
        return None, []  # skipped, not pass/fail - main() won't record this as a test

    df = pd.read_csv(raw_path)
    result = run_self_correcting_cleaning(df)
    logger.info("Result: success=%s, attempts=%d", result.succeeded, result.attempts_used)

    if not result.succeeded:
        error_reporter.report_failure(
            test_name="Full pipeline regression",
            dataset_name="production_raw_messy_1020row",
            n_rows=len(df),
            attempts_used=result.attempts_used,
            max_attempts=5,
            failure_category=FailureCategory.AGENT_EXHAUSTION,
            final_error_type="Unknown",
            final_error_message=result.failure_reason or "Unknown",
            remediation_suggestions=[RemediationSuggestion.ESCALATE_TO_ML_ENGINEER],
            human_notes="Production dataset failed to clean - requires investigation.",
        )
        return (
            _make_test_result(
                "full_pipeline_regression", "FAIL", result.attempts_used,
                error_message=str(result.failure_reason or "Unknown"),
                error_category=FailureCategory.AGENT_EXHAUSTION.name,
            )
        )

    logger.info("Cleaned shape: %s (from raw shape %s)", result.cleaned_df.shape, df.shape)
    encoding_ok = _assert_fully_numeric_except_target(result.cleaned_df)
    no_nulls = result.cleaned_df.drop(columns=["target"]).isna().sum().sum() == 0
    logger.info("No remaining NaNs: %s", no_nulls)

    if not (encoding_ok and no_nulls):
        logger.warning("Full pipeline regression test FAILED correctness checks.")
        error_reporter.report_failure(
            test_name="Full pipeline regression (correctness)",
            dataset_name="production_raw_messy_1020row",
            n_rows=len(result.cleaned_df),
            attempts_used=result.attempts_used,
            max_attempts=5,
            failure_category=FailureCategory.AGENT_EXHAUSTION,
            final_error_type="CorrectnessCheck",
            final_error_message="Data failed post-cleaning validation (NaNs or non-numeric columns remain)",
        )
        return (
            _make_test_result(
                "full_pipeline_regression", "FAIL", result.attempts_used,
                error_message="Post-cleaning validation failed (NaNs or non-numeric columns remain)",
                error_category="CORRECTNESS_CHECK",
                notes=f"Cleaned shape: {result.cleaned_df.shape} (from raw {df.shape})",
            )
        )

    return (
        _make_test_result(
            "full_pipeline_regression", "PASS", result.attempts_used,
            notes=f"Cleaned shape: {result.cleaned_df.shape} (from raw {df.shape})",
        )
    )


def main():
    """Run all edge case tests and export error reports."""
    logger.info("Starting edge case test suite...")

    test_functions = [
        test_all_nulls_column,
        test_single_row,
        test_all_duplicates,
        test_all_categorical,
        test_high_cardinality_category,
        test_date_parsing_failures,
        test_mixed_null_types,
        test_full_pipeline_regression,
    ]

    results: list[TestResult] = []
    for test_fn in test_functions:
        start = time.time()
        test_result, _ = test_fn()   # discard attempt logs
        duration = time.time() - start

        if test_result is None:
            continue  # skipped test (raw CSV missing), not a pass/fail

        test_result.duration_seconds = duration
        results.append(test_result)

    logger.info("Edge case test suite complete.")
    logger.info(error_reporter.summary())
    error_reporter.export_all()

    json_path, md_path = io_manager.write_report(
        phase="phase2",
        results=results,
        extra={"ollama_model": "qwen2.5-coder:7b"},
    )
    logger.info("Report written to: %s", md_path)

    failed = sum(1 for r in results if r.status == "FAIL")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
