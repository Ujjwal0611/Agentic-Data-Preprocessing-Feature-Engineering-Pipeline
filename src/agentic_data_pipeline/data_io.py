"""
data_io.py

Minimal I/O utility: one JSON report + one Markdown report per run,
flat into data/{phase}_outputs/. No nested subdirectories, no per-test files.

Structure produced:
    data/
    ├── phase2_outputs/
    │   └── report_YYYYMMDD_HHMMSS.json  (+ .md)
    └── phase3_outputs/
        ├── report_YYYYMMDD_HHMMSS.json  (+ .md)
        ├── runs.jsonl
        └── strategies/{strategy_name}/train_cleaned.csv, test_cleaned.csv, results.json
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TestResult:
    """Result of a single test execution."""
    test_id: str
    status: str               # "PASS" or "FAIL"
    attempts: int
    duration_seconds: float
    error_message: Optional[str] = None
    error_category: Optional[str] = None
    notes: Optional[str] = None


class DataIOManager:
    """Writes one flat report (JSON + MD) per run into data/{phase}_outputs/."""

    def __init__(self, data_root: Path = None):
        if data_root is None:
            # data_io.py lives at src/agentic_data_pipeline/data_io.py
            # parents[2] = project root (contains src/ and data/)
            data_root = Path(__file__).resolve().parents[2] / "data"
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self._run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def get_run_id(self) -> str:
        return self._run_id

    def get_output_dir(self, phase: str) -> Path:
        """Return (and create) the single flat output dir for this phase."""
        d = self.data_root / f"{phase}_outputs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_report(
        self,
        phase: str,
        results: List[TestResult],
        extra: Dict[str, Any] = None,
    ) -> tuple[Path, Path]:
        """Write one JSON + one Markdown report for this run.

        Args:
            phase: "phase2" or "phase3"
            results: list of TestResult from the run
            extra: optional extra fields in JSON (e.g. ollama_model)

        Returns:
            (json_path, md_path)
        """
        out_dir = self.get_output_dir(phase)
        passed = sum(1 for r in results if r.status == "PASS")
        failed = len(results) - passed

        payload = {
            "run_id": self._run_id,
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "total_tests": len(results),
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / len(results), 4) if results else 0,
            **(extra or {}),
            "results": [asdict(r) for r in results],
        }

        json_path = out_dir / f"report_{self._run_id}.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        md_lines = [
            f"# {phase.upper()} Run Report",
            f"**Run ID:** {self._run_id}",
            f"**Timestamp:** {payload['timestamp']}",
            "",
            "## Summary",
            "| Total | Passed | Failed | Pass Rate |",
            "|---|---|---|---|",
            f"| {len(results)} | {passed} | {failed} | {100*payload['pass_rate']:.1f}% |",
            "",
            "## Results",
            "| Test | Status | Attempts | Duration (s) | Error |",
            "|---|---|---|---|---|",
        ]
        for r in results:
            error = (r.error_message or "")[:60]
            md_lines.append(
                f"| {r.test_id} | {r.status} | {r.attempts} "
                f"| {r.duration_seconds:.1f} | {error} |"
            )

        if extra:
            md_lines.append("")
            md_lines.append("## Run Parameters")
            for k, v in extra.items():
                md_lines.append(f"- **{k}:** {v}")

        md_path = out_dir / f"report_{self._run_id}.md"
        with md_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        return json_path, md_path

    def append_run_history(self, phase: str, entry: Dict[str, Any]) -> None:
        """Append one line to a JSONL run history file (used by Phase 3)."""
        history_path = self.get_output_dir(phase) / "runs.jsonl"
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")