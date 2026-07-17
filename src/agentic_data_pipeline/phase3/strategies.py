"""
strategies.py

Defines named cleaning-strategy configurations that the Phase 3 orchestrator
cycles through when a cleaning attempt fails to improve F1 over baseline (or
over the best score seen so far). A "strategy" is NOT a different algorithm --
it's a different set of policy parameters injected into the LLM prompt, plus
a different generation temperature, so retries produce genuinely different
code rather than near-identical restatements of the same attempt.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CleaningStrategy:
    """One named configuration of cleaning policy + generation parameters."""

    name: str
    description: str
    temperature: float
    numeric_impute_policy: str          # e.g. "median", "mean"
    iqr_multiplier: float               # outlier clipping strictness
    high_cardinality_threshold: int     # unique-value count that triggers grouping
    high_cardinality_top_k: int         # how many top categories to keep before "other"
    extra_prompt_directives: str        # appended verbatim to the LLM prompt

    def to_prompt_directive_block(self) -> str:
        """Render this strategy's parameters as an instruction block the LLM
        can follow verbatim, on top of the fixed SYSTEM_PROMPT rules."""
        lines = [
            f"STRATEGY '{self.name}' PARAMETERS (follow these exactly for this attempt):",
            f"- Impute missing numeric values using the column {self.numeric_impute_policy}, "
            f"not any other statistic.",
            f"- When clipping outliers with the IQR method, use a multiplier of "
            f"{self.iqr_multiplier} (Q1 - {self.iqr_multiplier}*IQR to Q3 + "
            f"{self.iqr_multiplier}*IQR), not the textbook default unless it matches.",
            f"- For any categorical column with more than {self.high_cardinality_threshold} "
            f"unique values, keep only the top {self.high_cardinality_top_k} most frequent "
            f"categories and group everything else into a single 'other' category before "
            f"one-hot encoding.",
        ]
        if self.extra_prompt_directives:
            lines.append(self.extra_prompt_directives)
        return "\n".join(lines)


STRATEGY_LIBRARY: list[CleaningStrategy] = [
    CleaningStrategy(
        name="conservative_median",
        description=(
            "Baseline-safe strategy: median imputation, standard 1.5x IQR clipping, "
            "tight top-10 grouping for high-cardinality columns. Low generation "
            "temperature for deterministic, repeatable code."
        ),
        temperature=0.1,
        numeric_impute_policy="median",
        iqr_multiplier=1.5,
        high_cardinality_threshold=50,
        high_cardinality_top_k=10,
        extra_prompt_directives="",
    ),
    CleaningStrategy(
        name="mean_wide_tolerance",
        description=(
            "Mean imputation instead of median (sensitive to the outliers we WANT "
            "preserved, e.g. legitimate high earners), looser 3.0x IQR clipping so "
            "fewer legitimate extreme values get clipped, wider top-20 grouping."
        ),
        temperature=0.3,
        numeric_impute_policy="mean",
        iqr_multiplier=3.0,
        high_cardinality_threshold=50,
        high_cardinality_top_k=20,
        extra_prompt_directives=(
            "- Do not clip outliers unless they are a clear data-entry error "
            "pattern (e.g. impossible values); prefer preserving extreme-but-"
            "plausible values over aggressive clipping."
        ),
    ),
    CleaningStrategy(
        name="tight_grouping_high_temp",
        description=(
            "Median imputation, standard clipping, but aggressive top-5 grouping "
            "for high-cardinality columns to minimize one-hot dimensionality, "
            "with a higher temperature to encourage a structurally different "
            "code path than earlier attempts (helps escape a repeated bad pattern)."
        ),
        temperature=0.6,
        numeric_impute_policy="median",
        iqr_multiplier=1.5,
        high_cardinality_threshold=30,
        high_cardinality_top_k=5,
        extra_prompt_directives=(
            "- Prioritize minimizing the number of resulting columns after "
            "encoding; prefer frequency-based grouping over one-hot explosion "
            "wherever a column has more than a handful of unique values."
        ),
    ),
]


def get_strategy(name: str) -> CleaningStrategy:
    """Look up a strategy by name. Raises ValueError if not found -- fail
    loud rather than silently falling back to a default the caller didn't ask for."""
    for strategy in STRATEGY_LIBRARY:
        if strategy.name == name:
            return strategy
    valid_names = [s.name for s in STRATEGY_LIBRARY]
    raise ValueError(f"Unknown strategy '{name}'. Valid options: {valid_names}")