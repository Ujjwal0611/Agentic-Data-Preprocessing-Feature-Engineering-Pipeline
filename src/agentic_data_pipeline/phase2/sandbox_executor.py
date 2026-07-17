"""
sandbox_executor.py

Executes LLM-generated pandas code against a COPY of the working DataFrame,
inside a restricted namespace with a minimal __builtins__ allow-list.
Captures success/failure as a typed result rather than letting exceptions
propagate and crash the orchestrating agent loop.

Defense-in-depth philosophy: this module does NOT trust the LLM's own
assert statements (or lack thereof) as the sole correctness gate. After
execution, it independently re-checks the resulting DataFrame for the
class of "success=True but actually wrong" failures observed in testing
(unencoded columns, leftover NaNs, leftover duplicates, target tampering).
This is deliberately NOT a substitute for semantic/model-quality validation
(e.g. "was one-hot encoding grouped sensibly?") -- that requires the
F1-vs-baseline comparison built in Day 3, since there's no ground truth
for "correctly cleaned" available at this layer.
"""

from __future__ import annotations

import contextlib
import io
import logging
import traceback
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from agentic_data_pipeline.phase2.code_safety import validate_code_safety

logger = logging.getLogger("sandbox_executor")

# datetime is harmless stdlib (no filesystem/network/process access) and the
# LLM reaches for it out of habit when handling date-order violations --
# blocking it just burns a retry for no safety benefit. pandas/numpy are
# allowed for the same "redundant import" habit reason.
_ALLOWED_REIMPORTS: frozenset[str] = frozenset({"pandas", "numpy", "datetime"})


def _guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
    """Allow redundant `import pandas`/`import numpy`/`import datetime`
    (common LLM habits) while still refusing everything else at the
    execution layer, even if code_safety.py's static check somehow missed it."""
    if name not in _ALLOWED_REIMPORTS:
        raise ImportError(f"Import of '{name}' is not permitted in the sandbox.")
    return __import__(name, *args, **kwargs)


SAFE_BUILTINS: dict[str, Any] = {
    "len": len,
    "range": range,
    "print": print,
    "float": float,
    "int": int,
    "str": str,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "round": round,
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "bool": bool,
    "sorted": sorted,
    "isinstance": isinstance,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "next": next,
    "iter": iter,
    "any": any,
    "all": all,
    "True": True,
    "False": False,
    "None": None,
    "__import__": _guarded_import,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "AssertionError": AssertionError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "Exception": Exception,
}


@dataclass(frozen=True)
class ExecutionResult:
    """Typed outcome of a sandboxed code execution attempt."""

    success: bool
    resulting_df: Optional[pd.DataFrame]
    stdout: str
    error_traceback: Optional[str]


def _validate_structural_correctness(
    original_df: pd.DataFrame, result_df: pd.DataFrame, target_column: str
) -> list[str]:
    """Cheap, ground-truth-free structural checks that catch 'reported
    success but actually wrong' failures the dtype check alone misses.

    NOT a substitute for the F1-based semantic validation in Day 3 -- this
    only catches things provably wrong regardless of what 'correct'
    cleaning looks like for this specific dataset (e.g. it can prove NaNs
    remain, but it can't prove a one-hot grouping was the *right* grouping).

    Args:
        original_df: the pre-cleaning DataFrame, for target-integrity comparison.
        result_df: the DataFrame the generated code produced.
        target_column: name of the label column, which must never be altered.

    Returns:
        A list of human-readable issue descriptions. Empty if none found.
    """
    issues: list[str] = []

    non_target_cols = [c for c in result_df.columns if c != target_column]

    if non_target_cols:
        remaining_nans = result_df[non_target_cols].isna().sum()
        bad_nan_cols = remaining_nans[remaining_nans > 0]
        if not bad_nan_cols.empty:
            issues.append(
                f"NaNs still present after cleaning claimed success: "
                f"{bad_nan_cols.to_dict()}. Every missing value must be "
                f"imputed, or the row/column dropped."
            )

    n_dupes = int(result_df.duplicated().sum())
    if n_dupes > 0:
        issues.append(
            f"{n_dupes} duplicate rows still present after cleaning claimed "
            f"success. Call df.drop_duplicates() and reassign to df."
        )

    if len(original_df) > 0 and len(result_df) == 0:
        issues.append(
            "Cleaning code dropped every single row, leaving an empty "
            "DataFrame. This is never correct -- if NaNs or invalid values "
            "can't be fixed by imputation or targeted row/column drops, "
            "something upstream is wrong. Never resolve NaNs with a blanket "
            "df.dropna() across the whole DataFrame."
        )

    if target_column not in result_df.columns:
        issues.append(f"Target column '{target_column}' was dropped or renamed.")
    elif len(result_df) > 0 and target_column in original_df.columns:
        original_target_values = set(original_df[target_column].dropna().unique())
        result_target_values = set(result_df[target_column].dropna().unique())
        if not result_target_values.issubset(original_target_values):
            issues.append(
                f"Target column values changed: original had "
                f"{sorted(original_target_values)}, result has "
                f"{sorted(result_target_values)}. The target must never be "
                f"transformed, recoded, or scaled."
            )

    return issues


def execute_cleaning_code(
    code: str, df: pd.DataFrame, target_column: str = "target"
) -> ExecutionResult:
    """Statically validate, then execute, LLM-generated code against a copy
    of df. The code must operate on a variable named `df` and leave a valid,
    fully-numeric (except target), defect-free pandas DataFrame in that
    variable when it finishes.

    Args:
        code: Python source generated by the LLM.
        df: the DataFrame to clean. NEVER mutated -- a copy is passed in.
        target_column: name of the label column, excluded from the
            numeric-dtype requirement and checked for tampering.

    Returns:
        An ExecutionResult capturing success/failure, the resulting
        DataFrame (if successful), captured stdout, and a full traceback
        string (if it failed) suitable for feeding back into the next
        LLM prompt as self-correction context. Structural-correctness
        failures are surfaced as a ValueError traceback here too, so they
        flow through the exact same retry path as a runtime crash.
    """
    validate_code_safety(code)

    # Single merged namespace used as BOTH globals and locals for exec().
    # Using two separate dicts breaks closures (e.g. a lambda inside
    # df['col'].apply(lambda x: ...)) because the closure binds to
    # __globals__, not the locals dict -- variables assigned earlier in the
    # same exec() become invisible to it. See Bug #8.
    namespace: dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        "df": df.copy(deep=True),
        "pd": pd,
        "np": np,
    }
    stdout_capture = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(  # noqa: S102 -- deliberate, gated by validate_code_safety() above
                compile(code, "<agent_generated_code>", "exec"),
                namespace,
            )

        result_df = namespace.get("df")
        if not isinstance(result_df, pd.DataFrame):
            raise TypeError(
                "Generated code did not leave a valid pandas DataFrame in "
                "variable 'df' after execution."
            )

        # --- Defense-in-depth check #1: full numeric encoding ---
        # Do NOT trust the LLM's own assert (or lack of one). An agent that
        # reports success while leaving text columns behind is worse than
        # one that fails loudly, because nothing downstream will catch it.
        non_numeric = [
            c for c in result_df.columns
            if c != target_column and not pd.api.types.is_numeric_dtype(result_df[c])
        ]
        if non_numeric:
            raise ValueError(
                f"Cleaning code reported success but left non-numeric "
                f"column(s) unencoded: {non_numeric}. Every non-target "
                f"column (including date columns) must be numeric before "
                f"finishing -- discover ALL categorical/date columns "
                f"programmatically and convert every one of them, not just "
                f"the ones mentioned in warnings above."
            )

        # --- Defense-in-depth check #2: structural correctness ---
        structural_issues = _validate_structural_correctness(df, result_df, target_column)
        if structural_issues:
            raise ValueError(
                "Cleaning code reported success but failed structural "
                "correctness checks: " + "; ".join(structural_issues)
            )

        logger.info("Sandbox execution succeeded (%d rows, %d cols).", *result_df.shape)
        return ExecutionResult(
            success=True,
            resulting_df=result_df,
            stdout=stdout_capture.getvalue(),
            error_traceback=None,
        )
    except Exception:
        tb = traceback.format_exc()
        logger.warning("Sandbox execution failed:\n%s", tb)
        return ExecutionResult(
            success=False,
            resulting_df=None,
            stdout=stdout_capture.getvalue(),
            error_traceback=tb,
        )