"""
error_reporter.py

Structured error reporting for agent failures that require human intervention.
Captures failures, categorizes them, suggests remediation, and exports both
human-readable and machine-parseable reports for audit trail and debugging.

This is the observability layer that turns "the agent failed 5 times" into
actionable intelligence for a human operator.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger("error_reporter")


class FailureCategory(Enum):
    """Taxonomies of failures requiring human intervention."""

    AGENT_EXHAUSTION = "agent_exhaustion"  # Hit max retries without succeeding
    AMBIGUOUS_DATA = "ambiguous_data"  # Data is genuinely ambiguous (e.g., 80% parseable dates)
    PROFILE_INCOMPLETE = "profile_incomplete"  # Profile missed a data-quality issue
    SANDBOX_RESTRICTION = "sandbox_restriction"  # Legit code blocked by sandbox (shouldn't happen)
    LLM_CAPABILITY_LIMIT = "llm_capability_limit"  # Model can't solve this problem (too complex)
    UNKNOWN = "unknown"  # Couldn't determine root cause


class RemediationSuggestion(Enum):
    """Actionable remedies a human can take."""

    LOWER_THRESHOLD = "lower_parse_threshold"  # Relax date detection threshold
    ADD_PREPROCESSING_RULE = "add_preprocessing_rule"  # Extend system prompt
    INSPECT_DATA = "inspect_data"  # Manual data inspection needed
    ADJUST_SANDBOX = "adjust_sandbox"  # Relax sandbox restrictions
    ESCALATE_TO_ML_ENGINEER = "escalate_to_ml_engineer"  # Beyond agent scope
    RETRY_WITH_DIFFERENT_MODEL = "retry_with_different_model"  # Try qwen vs deepseek
    DOCUMENT_AS_KNOWN_LIMITATION = "document_as_known_limitation"  # Accept and document


@dataclass(frozen=True)
class FailureReport:
    """A single failure captured for human review."""

    timestamp: str
    test_name: str
    dataset_name: str
    n_rows: int
    attempts_used: int
    max_attempts: int
    failure_category: str
    final_error_type: str
    final_error_message: str
    last_traceback: Optional[str] = None
    remediation_suggestions: list[str] = field(default_factory=list)
    human_notes: str = ""

    def to_dict(self) -> dict:
        """Export as JSON-serializable dict."""
        return asdict(self)


@dataclass
class ErrorReporter:
    """Collects and exports error reports from a test run or agent loop."""

    run_name: str
    output_dir: Path = Path(__file__).resolve().parents[3] / "data" / "error_reports"
    failures: list[FailureReport] = field(default_factory=list)

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def report_failure(
        self,
        test_name: str,
        dataset_name: str,
        n_rows: int,
        attempts_used: int,
        max_attempts: int,
        failure_category: FailureCategory,
        final_error_type: str,
        final_error_message: str,
        last_traceback: Optional[str] = None,
        remediation_suggestions: Optional[list[RemediationSuggestion]] = None,
        human_notes: str = "",
    ) -> None:
        """Record a failure. Call this when agent exhausts retries without success.

        Args:
            test_name: name of the test case (e.g., "Malformed dates").
            dataset_name: a descriptive name for the dataset (e.g., "edge_case_malformed_5row").
            n_rows: number of rows in the dataset.
            attempts_used: how many retries the agent burned.
            max_attempts: maximum allowed retries.
            failure_category: FailureCategory enum.
            final_error_type: exception class name (e.g., "ValueError").
            final_error_message: the exception message.
            last_traceback: full traceback string from the last attempt.
            remediation_suggestions: list of RemediationSuggestion enum values.
            human_notes: free-text notes for human reviewer (e.g., "this is expected — data is ambiguous").
        """
        if remediation_suggestions is None:
            remediation_suggestions = []

        report = FailureReport(
            timestamp=datetime.utcnow().isoformat(),
            test_name=test_name,
            dataset_name=dataset_name,
            n_rows=n_rows,
            attempts_used=attempts_used,
            max_attempts=max_attempts,
            failure_category=failure_category.value,
            final_error_type=final_error_type,
            final_error_message=final_error_message,
            last_traceback=last_traceback,
            remediation_suggestions=[s.value for s in remediation_suggestions],
            human_notes=human_notes,
        )
        self.failures.append(report)
        logger.warning(
            "Failure recorded: %s (test: %s, category: %s, attempts: %d/%d)",
            dataset_name,
            test_name,
            failure_category.value,
            attempts_used,
            max_attempts,
        )

    def export_json(self) -> Path:
        """Export all failures as a single JSON file (machine-parseable)."""
        data = {
            "run_name": self.run_name,
            "timestamp": datetime.utcnow().isoformat(),
            "total_failures": len(self.failures),
            "failures": [f.to_dict() for f in self.failures],
        }
        path = self.output_dir / f"{self.run_name}_failures.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        logger.info("Exported JSON error report to %s", path)
        return path

    def export_markdown(self) -> Path:
        """Export as human-readable Markdown report."""
        lines = [
            f"# Error Report: {self.run_name}",
            f"\n**Generated:** {datetime.utcnow().isoformat()}",
            f"**Total Failures:** {len(self.failures)}\n",
        ]

        if not self.failures:
            lines.append("✓ No failures recorded.")
        else:
            # Group by category
            by_category = {}
            for f in self.failures:
                cat = f.failure_category
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(f)

            for category in sorted(by_category.keys()):
                failures = by_category[category]
                lines.append(f"\n## {category.replace('_', ' ').title()}")
                lines.append(f"**Count:** {len(failures)}\n")

                for f in failures:
                    lines.append(f"### {f.test_name} — {f.dataset_name}")
                    lines.append(f"- **Rows:** {f.n_rows}")
                    lines.append(f"- **Attempts:** {f.attempts_used}/{f.max_attempts}")
                    lines.append(f"- **Error:** `{f.final_error_type}: {f.final_error_message}`")

                    if f.remediation_suggestions:
                        suggestions = ", ".join([s.replace("_", " ").title() for s in f.remediation_suggestions])
                        lines.append(f"- **Remediation:** {suggestions}")

                    if f.human_notes:
                        lines.append(f"- **Notes:** {f.human_notes}")

                    if f.last_traceback:
                        lines.append("\n<details><summary>Traceback</summary>\n")
                        lines.append("```")
                        lines.append(f.last_traceback)
                        lines.append("```")
                        lines.append("\n</details>\n")

                    lines.append("")

        path = self.output_dir / f"{self.run_name}_failures.md"
        with path.open("w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        logger.info("Exported Markdown error report to %s", path)
        return path

    def export_all(self) -> dict[str, Path]:
        """Export both JSON and Markdown reports.

        Returns:
            A dict mapping format name ('json', 'markdown') to output Path.
        """
        results = {
            "json": self.export_json(),
            "markdown": self.export_markdown(),
        }
        logger.info("Error reports exported to %s", self.output_dir)
        return results

    def summary(self) -> str:
        """Return a one-line summary for logging."""
        if not self.failures:
            return "✓ No failures recorded."
        by_cat = {}
        for f in self.failures:
            by_cat[f.failure_category] = by_cat.get(f.failure_category, 0) + 1
        parts = [f"{count} {cat}" for cat, count in sorted(by_cat.items())]
        return f"Failures: {', '.join(parts)}"