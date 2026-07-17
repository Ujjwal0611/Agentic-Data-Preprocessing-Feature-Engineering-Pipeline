"""
reporter.py

Generates the Phase 3 executive summary: F1-vs-baseline comparison across
every cleaning strategy attempted, framed around the zero-marginal-cost
story of a fully local LLM pipeline (no per-token API billing, no data
egress). Writes both a machine-parseable comparison.json and a human-
readable executive.md.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("reporter")


@dataclass(frozen=True)
class StrategyRunResult:
    """Outcome of running ONE cleaning strategy end-to-end through Phase 3:
    clean train/test independently, align columns, retrain, score."""

    strategy_name: str
    succeeded: bool
    f1_score: Optional[float]
    accuracy: Optional[float]
    beat_baseline: Optional[bool]
    n_train_rows: Optional[int]
    n_test_rows: Optional[int]
    n_features: Optional[int]
    agent_attempts_used: Optional[int]
    failure_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def build_executive_report(
    baseline_f1: float,
    strategy_results: list[StrategyRunResult],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write comparison.json + executive.md to output_dir (caller passes the
    already-created reports/ directory from DataIOManager.create_test_run_dirs).

    Returns:
        (comparison_json_path, executive_md_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    successful = [r for r in strategy_results if r.succeeded and r.f1_score is not None]
    best = max(successful, key=lambda r: r.f1_score) if successful else None

    comparison = {
        "timestamp": datetime.now().isoformat(),
        "baseline_f1": baseline_f1,
        "strategies_attempted": len(strategy_results),
        "strategies_succeeded": len(successful),
        "best_strategy": best.strategy_name if best else None,
        "best_f1": best.f1_score if best else None,
        "improvement_over_baseline": (round(best.f1_score - baseline_f1, 4) if best else None),
        "results": [r.to_dict() for r in strategy_results],
    }

    comparison_path = output_dir / "comparison.json"
    with comparison_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    md_lines = [
        "# Phase 3 Executive Summary: Agentic Data Cleaning vs. Baseline",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        "",
        "## Cost Model",
        f"This entire cleaning + retraining cycle -- across {len(strategy_results)} "
        "independent strategies, each with its own self-correcting retry budget -- "
        "ran entirely on a local LLM (qwen2.5-coder:7b via Ollama). Zero per-token "
        "API cost, zero data egress.",
        "",
        "## Baseline",
        f"- **Naive baseline F1** (string columns dropped, no real cleaning): `{baseline_f1:.4f}`",
        "",
        "## Strategy Comparison",
        "",
        "| Strategy | Status | F1 | Accuracy | vs. Baseline | Agent Attempts (train+test) |",
        "|---|---|---|---|---|---|",
    ]

    for r in strategy_results:
        if r.succeeded and r.f1_score is not None:
            delta = round(r.f1_score - baseline_f1, 4)
            delta_str = f"{'+' if delta >= 0 else ''}{delta:.4f}"
            status = "PASS" if delta >= 0 else "RAN (below baseline)"
            f1_str = f"{r.f1_score:.4f}"
            acc_str = f"{r.accuracy:.4f}" if r.accuracy is not None else "N/A"
        else:
            status = "FAIL"
            f1_str = acc_str = delta_str = "N/A"
        md_lines.append(
            f"| {r.strategy_name} | {status} | {f1_str} | {acc_str} | {delta_str} "
            f"| {r.agent_attempts_used if r.agent_attempts_used is not None else 'N/A'} |"
        )

    md_lines.append("")
    if best:
        delta = round(best.f1_score - baseline_f1, 4)
        md_lines.append(
            f"## Winner: `{best.strategy_name}`\n\n"
            f"F1 improved from baseline `{baseline_f1:.4f}` to `{best.f1_score:.4f}` "
            f"({'+' if delta >= 0 else ''}{delta:.4f})."
        )
    else:
        md_lines.append(
            "## No strategy succeeded\n\n"
            "Every cleaning strategy either exhausted its retry budget or produced "
            "a result that failed structural validation. See `data/error_reports/` "
            "for per-attempt failure detail."
        )

    md_lines.append("")
    md_lines.append("## Failed Strategies (detail)")
    failed = [r for r in strategy_results if not r.succeeded]
    if not failed:
        md_lines.append("None -- all strategies completed successfully.")
    else:
        for r in failed:
            md_lines.append(f"- **{r.strategy_name}**: {r.failure_reason or 'Unknown failure'}")

    executive_path = output_dir / "executive.md"
    with executive_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    logger.info("Executive report written to %s", executive_path)
    return comparison_path, executive_path