"""
real_world_stress_test.py

Validation experiment: run the Phase 2 agent against a REAL, external CSV
with ZERO code changes made in advance to accommodate it. This is not
a unit test with a known-good expected outcome -- it's an honest probe
to find out what actually happens, so "I believe it would struggle with
real-world data" can become "I tested it and here's exactly what broke."

Usage:
    uv run python -m agentic_data_pipeline.real_world_stress_test <path_to_csv> <target_column>

Example:
    uv run python -m agentic_data_pipeline.real_world_stress_test data\\external\\titanic.csv Survived

What this script does NOT do:
    - It does not modify the CSV before feeding it in.
    - It does not add new prompt rules or profiler logic on the fly.
    - It does not retry with a different strategy if the first attempt fails.
It runs the pipeline exactly as it exists today, once, and reports what
happened -- including prompt size, timing, and the literal failure mode
if one occurs.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from agentic_data_pipeline.phase2.agent import run_self_correcting_cleaning
from agentic_data_pipeline.phase2.profiler import profile_dataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("real_world_stress_test")

REPORT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "real_world_stress_test"


def _char_count_to_rough_tokens(char_count: int) -> int:
    """Very rough english-text heuristic (~4 chars/token) -- good enough to
    flag 'this prompt is probably too big', not precise token accounting."""
    return char_count // 4


def run_stress_test(csv_path: Path, target_column: str) -> dict:
    """Run the unmodified pipeline against a real external CSV once, and
    capture everything worth knowing about what happened -- success or not.
    """
    result_record: dict = {
        "csv_path": str(csv_path),
        "target_column": target_column,
    }

    # --- Step 1: can we even read it? (encoding / parsing issues surface here) ---
    logger.info("=== STEP 1: Reading CSV ===")
    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError as exc:
        logger.error("FAILED at CSV read: encoding issue. %s", exc)
        result_record["stage_failed"] = "csv_read_encoding"
        result_record["error"] = str(exc)
        logger.info(
            "Retry hint: try pd.read_csv(path, encoding='latin-1') or "
            "encoding='cp1252' -- this is a real, common failure mode this "
            "pipeline does not currently handle automatically."
        )
        return result_record
    except Exception as exc:
        logger.error("FAILED at CSV read: %s: %s", type(exc).__name__, exc)
        result_record["stage_failed"] = "csv_read_other"
        result_record["error"] = f"{type(exc).__name__}: {exc}"
        return result_record

    result_record["n_rows"] = len(df)
    result_record["n_columns"] = len(df.columns)
    result_record["columns"] = df.columns.tolist()
    logger.info("Read OK: %d rows, %d columns.", len(df), len(df.columns))

    if target_column not in df.columns:
        logger.error(
            "Target column '%s' not found. Available columns: %s",
            target_column, df.columns.tolist(),
        )
        result_record["stage_failed"] = "target_column_missing"
        return result_record

    # --- Step 2: profile it, and measure prompt size (the untested scaling risk) ---
    logger.info("=== STEP 2: Profiling ===")
    try:
        profile = profile_dataset(df, target_column=target_column)
        profile_text = profile.to_prompt_text()
    except Exception as exc:
        logger.error("FAILED at profiling: %s: %s", type(exc).__name__, exc)
        result_record["stage_failed"] = "profiling"
        result_record["error"] = f"{type(exc).__name__}: {exc}"
        return result_record

    profile_char_count = len(profile_text)
    result_record["profile_char_count"] = profile_char_count
    result_record["profile_rough_token_estimate"] = _char_count_to_rough_tokens(profile_char_count)
    logger.info(
        "Profile generated: %d chars (~%d tokens, rough estimate). "
        "For reference, the synthetic Phase 1 dataset (8 columns) profiles "
        "to a few hundred chars -- compare against that baseline.",
        profile_char_count, result_record["profile_rough_token_estimate"],
    )
    if len(df.columns) > 15:
        logger.warning(
            "%d columns is well beyond the ~8 columns this pipeline was "
            "tuned against. Watch for degraded generation quality, not "
            "just outright crashes -- the model may silently ignore some "
            "columns rather than raising a clean error.",
            len(df.columns),
        )

    # --- Step 3: run the actual self-correcting agent, unmodified ---
    logger.info("=== STEP 3: Running self-correcting agent (up to 5 attempts) ===")
    start = time.time()
    try:
        agent_result = run_self_correcting_cleaning(df, target_column=target_column)
    except Exception as exc:
        # A genuinely UNHANDLED exception escaping the agent loop itself
        # would be a real, notable finding -- the agent is designed to
        # never raise, so this branch firing is itself a bug report.
        elapsed = time.time() - start
        logger.error(
            "UNEXPECTED: agent raised an unhandled exception (it's designed "
            "not to). %s: %s", type(exc).__name__, exc,
        )
        result_record["stage_failed"] = "agent_unhandled_exception"
        result_record["error"] = f"{type(exc).__name__}: {exc}"
        result_record["elapsed_seconds"] = round(elapsed, 1)
        return result_record

    elapsed = time.time() - start
    result_record["elapsed_seconds"] = round(elapsed, 1)
    result_record["attempts_used"] = agent_result.attempts_used
    result_record["succeeded"] = agent_result.succeeded

    if not agent_result.succeeded:
        logger.warning(
            "Agent did NOT succeed after %d attempts (%.1fs). Failure reason:\n%s",
            agent_result.attempts_used, elapsed, agent_result.failure_reason,
        )
        result_record["stage_failed"] = "agent_exhausted"
        result_record["failure_reason"] = agent_result.failure_reason
        return result_record

    # --- Step 4: even on "success", sanity-check the result actually looks right ---
    logger.info("=== STEP 4: Post-hoc sanity check on 'successful' result ===")
    cleaned = agent_result.cleaned_df
    non_target_cols = [c for c in cleaned.columns if c != target_column]
    non_numeric = [c for c in non_target_cols if not pd.api.types.is_numeric_dtype(cleaned[c])]
    remaining_nans = int(cleaned[non_target_cols].isna().sum().sum())
    row_retention_pct = round(100 * len(cleaned) / len(df), 1) if len(df) else 0.0

    result_record["cleaned_shape"] = list(cleaned.shape)
    result_record["non_numeric_columns_remaining"] = non_numeric
    result_record["nans_remaining"] = remaining_nans
    result_record["row_retention_pct"] = row_retention_pct

    logger.info(
        "SUCCEEDED in %d attempt(s), %.1fs. Cleaned shape: %s. "
        "Row retention: %.1f%% (%d -> %d rows). Non-numeric cols remaining: %s. "
        "NaNs remaining: %d.",
        agent_result.attempts_used, elapsed, cleaned.shape,
        row_retention_pct, len(df), len(cleaned), non_numeric, remaining_nans,
    )

    if row_retention_pct < 90.0:
        logger.warning(
            "Row retention dropped below 90%% -- worth manually checking "
            "WHY so many rows were dropped. This can be legitimate (heavy "
            "real missingness) or a sign the agent over-aggressively "
            "dropped rows instead of imputing."
        )

    return result_record


def _print_summary(record: dict) -> None:
    print("\n--- REAL-WORLD STRESS TEST SUMMARY ---")
    for key, value in record.items():
        if key == "columns" and isinstance(value, list) and len(value) > 20:
            print(f"{key}: [{len(value)} columns, truncated] {value[:10]} ...")
        else:
            print(f"{key}: {value}")

    if record.get("stage_failed"):
        print(f"\nRESULT: FAILED at stage '{record['stage_failed']}'.")
        print("This is a genuine finding, not a bad outcome -- it tells you")
        print("exactly which real-world gap to prioritize fixing next.")
    else:
        print("\nRESULT: SUCCEEDED end-to-end on this real dataset, unmodified.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the unmodified Phase 2 pipeline against a real external CSV."
    )
    parser.add_argument("csv_path", type=str, help="Path to the real-world CSV to test.")
    parser.add_argument("target_column", type=str, help="Name of the target/label column.")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"ERROR: file not found: {csv_path}")
        sys.exit(1)

    record = run_stress_test(csv_path, args.target_column)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    import json
    from datetime import datetime
    out_path = REPORT_DIR / f"stress_test_{csv_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)
    logger.info("Full result record saved to %s", out_path)

    _print_summary(record)
    sys.exit(0 if not record.get("stage_failed") else 1)


if __name__ == "__main__":
    main()
