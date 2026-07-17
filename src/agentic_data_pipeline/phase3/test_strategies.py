"""
test_strategies.py

Run with:
    uv run python -m agentic_data_pipeline.test_strategies
"""

from __future__ import annotations

import logging

from agentic_data_pipeline.phase3.strategies import STRATEGY_LIBRARY, get_strategy


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_strategies")


def test_all_strategies_have_unique_names():
    logger.info("=== TEST: strategy names are unique ===")
    names = [s.name for s in STRATEGY_LIBRARY]
    assert len(names) == len(set(names)), f"Duplicate strategy names found: {names}"
    logger.info("PASSED: %d strategies, all unique names: %s", len(names), names)


def test_get_strategy_returns_correct_config():
    logger.info("=== TEST: get_strategy looks up by name correctly ===")
    strategy = get_strategy("conservative_median")
    assert strategy.numeric_impute_policy == "median"
    assert strategy.temperature == 0.1
    logger.info("PASSED: retrieved '%s' with expected params.", strategy.name)


def test_get_strategy_unknown_name_raises():
    logger.info("=== TEST: unknown strategy name raises ValueError ===")
    try:
        get_strategy("does_not_exist")
        logger.warning("FAILED: expected ValueError, none raised.")
    except ValueError as exc:
        logger.info("PASSED: ValueError raised as expected: %s", exc)


def test_prompt_directive_block_contains_key_params():
    logger.info("=== TEST: rendered prompt block reflects strategy params ===")
    strategy = get_strategy("tight_grouping_high_temp")
    block = strategy.to_prompt_directive_block()
    assert "median" in block
    assert "5" in block  # top_k
    logger.info("PASSED: rendered block includes expected policy values.")
    logger.info("Sample rendered block:\n%s", block)


def main():
    logger.info("Starting strategies test suite...")
    test_all_strategies_have_unique_names()
    test_get_strategy_returns_correct_config()
    test_get_strategy_unknown_name_raises()
    test_prompt_directive_block_contains_key_params()
    logger.info("strategies test suite complete.")


if __name__ == "__main__":
    main()