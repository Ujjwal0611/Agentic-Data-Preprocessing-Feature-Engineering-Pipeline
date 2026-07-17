"""
agent.py

Runs ONE full generate -> validate -> execute -> self-correct cycle against
a messy dataset, with a bounded retry count. The LLM sees its own full
recent attempt history (code + error) on every retry, not just the single
most recent one -- this lets it recognize when it's repeating a mistake
instead of re-deriving the same broken fix from a single error message.

This is deliberately a single-pass demonstration of the self-correction
loop -- Phase 3's orchestrator wraps this in the outer F1-improvement state
machine across multiple cleaning strategies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_data_pipeline.phase3.strategies import CleaningStrategy

from agentic_data_pipeline.phase2.code_safety import UnsafeCodeError
from agentic_data_pipeline.phase2.llm_client import (
    AttemptRecord,
    DEFAULT_MODEL_NAME,
    LLMGenerationError,
    generate_cleaning_code,
)
from agentic_data_pipeline.phase2.profiler import profile_dataset
from agentic_data_pipeline.phase2.sandbox_executor import execute_cleaning_code

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("agent")

MAX_RETRIES: int = 5

# parents[0]=phase2, [1]=agentic_data_pipeline, [2]=src, [3]=project root
DATA_DIR = Path(__file__).resolve().parents[3] / "data"
RAW_CSV_PATH = DATA_DIR / "raw_messy_dataset.csv"
CLEANED_CSV_PATH = DATA_DIR / "agent_cleaned_dataset.csv"
ATTEMPT_LOG_DIR = DATA_DIR / "attempt_logs"


@dataclass(frozen=True)
class AgentRunResult:
    """Outcome of one full self-correction cycle."""

    succeeded: bool
    attempts_used: int
    final_code: Optional[str]
    cleaned_df: Optional[pd.DataFrame]
    failure_reason: Optional[str]


def _dump_attempt_history_to_disk(history: list[AttemptRecord]) -> Optional[Path]:
    """Write the full failed-attempt history to disk for human post-mortem
    review. Only called when all retries are exhausted -- successful runs
    don't need this, and error_reporter.py already logs the summary.

    This is deliberately NOT read back by the LLM (that's what attempt_history
    in-memory is for during the loop itself) -- this file exists purely for
    a human to open afterward and see the full "what did it try, in order"
    story in one place.
    """
    if not history:
        return None

    ATTEMPT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = ATTEMPT_LOG_DIR / f"exhausted_{timestamp}.txt"

    with path.open("w", encoding="utf-8") as fh:
        for record in history:
            fh.write(
                f"=== Attempt {record.attempt_number} ===\n"
                f"CODE:\n{record.code}\n\n"
                f"ERROR:\n{record.error}\n\n"
            )

    logger.info("Full attempt history dumped to %s for post-mortem review.", path)
    return path


def run_self_correcting_cleaning(
    df: pd.DataFrame,
    target_column: str = "target",
    model_name: str = DEFAULT_MODEL_NAME,
    max_retries: int = MAX_RETRIES,
    strategy: "CleaningStrategy | None" = None,
    ) -> AgentRunResult:
    """Run the generate -> validate -> execute -> self-correct loop.

    Each retry's prompt includes the model's own recent attempt history
    (code + exact error, capped at the most recent few -- see
    llm_client.MAX_HISTORY_ATTEMPTS_IN_PROMPT), so it can recognize a
    repeated mistake instead of just reacting to the latest traceback
    in isolation.
    """
    profile = profile_dataset(df, target_column=target_column)
    profile_text = profile.to_prompt_text()

    attempt_history: list[AttemptRecord] = []

    for attempt in range(1, max_retries + 1):
        logger.info("Attempt %d/%d: requesting cleaning code from '%s'.", attempt, max_retries, model_name)
        try:
            code = generate_cleaning_code(
                dataset_profile_text=profile_text,
                model_name=model_name,
                attempt_history=attempt_history,
                strategy=strategy,
            )
        except LLMGenerationError as exc:
            logger.error("LLM generation failed on attempt %d: %s", attempt, exc)
            return AgentRunResult(False, attempt, None, None, str(exc))

        try:
            result = execute_cleaning_code(code, df, target_column=target_column)
        except UnsafeCodeError as exc:
            logger.warning("Attempt %d rejected by safety validator: %s", attempt, exc)
            attempt_history.append(AttemptRecord(attempt, code, f"UnsafeCodeError: {exc}"))
            continue

        if result.success:
            logger.info("Attempt %d succeeded after %d total attempt(s).", attempt, attempt)
            return AgentRunResult(True, attempt, code, result.resulting_df, None)

        logger.warning("Attempt %d failed at runtime; feeding traceback back for retry.", attempt)
        attempt_history.append(AttemptRecord(attempt, code, result.error_traceback))

    last_error = attempt_history[-1].error if attempt_history else "Unknown"
    last_code = attempt_history[-1].code if attempt_history else None
    _dump_attempt_history_to_disk(attempt_history)

    return AgentRunResult(
        succeeded=False,
        attempts_used=max_retries,
        final_code=last_code,
        cleaned_df=None,
        failure_reason=f"Exhausted {max_retries} attempts. Last error:\n{last_error}",
    )


def main() -> None:
    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(
            f"{RAW_CSV_PATH} not found. Run Phase 1's baseline_pipeline first: "
            f"uv run python -m agentic_data_pipeline.phase1.baseline_pipeline"
        )

    df = pd.read_csv(RAW_CSV_PATH)
    result = run_self_correcting_cleaning(df)

    if result.succeeded:
        result.cleaned_df.to_csv(CLEANED_CSV_PATH, index=False)
        print(f"\n--- AGENT SUCCEEDED after {result.attempts_used} attempt(s) ---")
        print(f"Cleaned dataset saved to {CLEANED_CSV_PATH}")
        print("\nFinal generated code:\n")
        print(result.final_code)
    else:
        print(f"\n--- AGENT FAILED: {result.failure_reason} ---")


if __name__ == "__main__":
    main()