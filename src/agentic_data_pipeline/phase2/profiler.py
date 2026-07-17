"""
profiler.py

Converts a raw pandas DataFrame into a structured, LLM-readable metadata
summary. This is the ONLY representation of the dataset that ever gets sent
to the local Ollama model -- raw row data never leaves this process.

Detects: missing values, numeric outliers (IQR), near-duplicate/fuzzy-spelled
categorical values, exact duplicate rows, and cross-column date-order violations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

logger = logging.getLogger("profiler")

IQR_OUTLIER_MULTIPLIER: float = 1.5
MAX_SAMPLE_VALUES: int = 8
FUZZY_SIMILARITY_THRESHOLD: float = 80.0
DATE_MIN_PARSE_RATE: float = 0.9


@dataclass(frozen=True)
class ColumnProfile:
    """Structured, typed summary of a single DataFrame column."""

    name: str
    dtype: str
    nan_count: int
    nan_percentage: float
    n_unique: int
    is_categorical_text: bool
    sample_values: list[Any] = field(default_factory=list)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    outlier_count: Optional[int] = None
    fuzzy_duplicate_clusters: list[list[str]] = field(default_factory=list)


@dataclass(frozen=True)
class DatasetProfile:
    """Structured, typed summary of the full dataset."""

    n_rows: int
    n_columns: int
    target_column: str
    columns: list[ColumnProfile]
    n_duplicate_rows: int = 0
    date_column_candidates: list[str] = field(default_factory=list)
    date_order_violations: list[dict[str, Any]] = field(default_factory=list)

    def to_prompt_text(self) -> str:
        """Render this profile as plain English the LLM can reason over."""
        lines = [
            f"Dataset shape: {self.n_rows} rows, {self.n_columns} columns.",
            f"Target column: '{self.target_column}' (this column must never "
            f"be modified, dropped, or used to derive other features).",
        ]

        if self.n_duplicate_rows > 0:
            lines.append(
                f"WARNING: {self.n_duplicate_rows} fully duplicate rows detected. "
                f"Drop them with df.drop_duplicates(), keeping the first occurrence."
            )

        if self.date_order_violations:
            for v in self.date_order_violations:
                lines.append(
                    f"WARNING: columns '{v['col_a']}' and '{v['col_b']}' look like "
                    f"dates. In {v['a_after_b']} rows, '{v['col_a']}' is after "
                    f"'{v['col_b']}'; in {v['b_after_a']} rows it's the reverse. "
                    f"Based on the column names, determine which ordering is "
                    f"logically correct and either flag or correct the violating rows."
                )

        lines.append("")
        lines.append("Columns to clean:")
        for col in self.columns:
            if col.name == self.target_column:
                continue
            line = (
                f"- '{col.name}' (dtype={col.dtype}): "
                f"{col.nan_count} missing values ({col.nan_percentage}%), "
                f"{col.n_unique} unique values"
            )
            if col.is_categorical_text:
                line += f", sample values: {col.sample_values}"
            else:
                line += (
                    f", range=[{col.min_value}, {col.max_value}], "
                    f"mean={col.mean_value}"
                )
                if col.outlier_count:
                    line += f", {col.outlier_count} potential outliers (IQR method)"
            lines.append(line)

            if col.fuzzy_duplicate_clusters:
                line2 = (
                    f"  NOTE: '{col.name}' has near-duplicate spellings that likely "
                    f"represent the same category. Normalize these via "
                    f".str.strip().str.lower() plus a manual mapping dict before "
                    f"encoding. Suspected groups: {col.fuzzy_duplicate_clusters}"
                )
                lines.append(line2)

        return "\n".join(lines)


def _cluster_similar_values(
    values: list[str], threshold: float = FUZZY_SIMILARITY_THRESHOLD
) -> list[list[str]]:
    """Group near-duplicate strings using rapidfuzz similarity ratios.

    Args:
        values: unique string values from a categorical column.
        threshold: minimum similarity score (0-100) to consider strings
            as the same underlying category.

    Returns:
        A list of clusters, each containing 2+ original values judged to be
        the same category under different spelling/casing/whitespace.
    """
    normalized = [(v, str(v).strip().lower()) for v in values]
    assigned: set[str] = set()
    clusters: list[list[str]] = []

    for i, (orig_i, norm_i) in enumerate(normalized):
        if orig_i in assigned:
            continue
        cluster = [orig_i]
        assigned.add(orig_i)
        for orig_j, norm_j in normalized[i + 1 :]:
            if orig_j in assigned:
                continue
            if fuzz.ratio(norm_i, norm_j) >= threshold:
                cluster.append(orig_j)
                assigned.add(orig_j)
        clusters.append(cluster)

    return [c for c in clusters if len(c) > 1]


def _detect_date_columns(df: pd.DataFrame, min_parse_rate: float = DATE_MIN_PARSE_RATE) -> list[str]:
    """Identify text columns whose values are overwhelmingly parseable as dates."""
    candidates = []
    for col in df.select_dtypes(include=["object", "str"]).columns:
        parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
        if parsed.notna().mean() >= min_parse_rate:
            candidates.append(col)
    return candidates


def _detect_date_order_violations(
    df: pd.DataFrame, date_columns: list[str]
) -> list[dict[str, Any]]:
    """For every pair of detected date columns, report rows with
    illogical orderings so the LLM can judge correctness from column names."""
    violations = []
    for col_a, col_b in combinations(date_columns, 2):
        a = pd.to_datetime(df[col_a], errors="coerce", format="mixed")
        b = pd.to_datetime(df[col_b], errors="coerce", format="mixed")
        valid_mask = a.notna() & b.notna()
        if valid_mask.sum() == 0:
            continue
        a_after_b = int((a[valid_mask] > b[valid_mask]).sum())
        b_after_a = int((b[valid_mask] > a[valid_mask]).sum())
        if a_after_b > 0 and b_after_a > 0:
            violations.append(
                {
                    "col_a": col_a,
                    "col_b": col_b,
                    "a_after_b": a_after_b,
                    "b_after_a": b_after_a,
                }
            )
    return violations


def profile_dataset(df: pd.DataFrame, target_column: str = "target") -> DatasetProfile:
    """Build a structured profile detecting missing values, outliers,
    fuzzy-duplicate categories, exact duplicate rows, and date-order issues.

    Args:
        df: the raw messy DataFrame to profile.
        target_column: name of the label column, excluded from cleaning instructions.

    Returns:
        A DatasetProfile summarizing every column plus dataset-level issues.

    Raises:
        ValueError: if target_column is not present in df.
    """
    if target_column not in df.columns:
        raise ValueError(
            f"target_column '{target_column}' not found in DataFrame columns: "
            f"{list(df.columns)}"
        )

    n_duplicate_rows = int(df.duplicated().sum())
    date_columns = _detect_date_columns(df)
    date_violations = _detect_date_order_violations(df, date_columns)

    column_profiles: list[ColumnProfile] = []
    for col_name in df.columns:
        series = df[col_name]
        nan_count = int(series.isna().sum())
        nan_pct = round(100 * nan_count / len(df), 2) if len(df) else 0.0
        is_text = series.dtype == object or str(series.dtype).startswith("str")

        kwargs: dict[str, Any] = dict(
            name=col_name,
            dtype=str(series.dtype),
            nan_count=nan_count,
            nan_percentage=nan_pct,
            n_unique=int(series.nunique(dropna=True)),
            is_categorical_text=bool(is_text),
        )

        if is_text:
            unique_values = series.dropna().unique().tolist()
            kwargs["sample_values"] = unique_values[:MAX_SAMPLE_VALUES]
            if col_name not in date_columns:
                kwargs["fuzzy_duplicate_clusters"] = _cluster_similar_values(unique_values)
        else:
            numeric_series = pd.to_numeric(series, errors="coerce").dropna()
            if len(numeric_series) > 0:
                q1, q3 = numeric_series.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound = q1 - IQR_OUTLIER_MULTIPLIER * iqr
                upper_bound = q3 + IQR_OUTLIER_MULTIPLIER * iqr
                outlier_count = int(
                    ((numeric_series < lower_bound) | (numeric_series > upper_bound)).sum()
                )
                kwargs.update(
                    min_value=round(float(numeric_series.min()), 2),
                    max_value=round(float(numeric_series.max()), 2),
                    mean_value=round(float(numeric_series.mean()), 2),
                    outlier_count=outlier_count,
                )

        column_profiles.append(ColumnProfile(**kwargs))

    profile = DatasetProfile(
        n_rows=len(df),
        n_columns=len(df.columns),
        target_column=target_column,
        columns=column_profiles,
        n_duplicate_rows=n_duplicate_rows,
        date_column_candidates=date_columns,
        date_order_violations=date_violations,
    )
    logger.info(
        "Profiled dataset: %d rows, %d columns, %d duplicate rows, "
        "%d date columns, %d date-order violation pairs.",
        profile.n_rows,
        profile.n_columns,
        n_duplicate_rows,
        len(date_columns),
        len(date_violations),
    )
    return profile