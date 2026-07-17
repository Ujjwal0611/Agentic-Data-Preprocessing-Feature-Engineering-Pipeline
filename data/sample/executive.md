# Phase 3 Executive Summary: Agentic Data Cleaning vs. Baseline

**Generated:** 2026-07-17T23:30:30.660804

## Cost Model
This entire cleaning + retraining cycle -- across 3 independent strategies, each with its own self-correcting retry budget -- ran entirely on a local LLM (qwen2.5-coder:7b via Ollama). Zero per-token API cost, zero data egress.

## Baseline
- **Naive baseline F1** (string columns dropped, no real cleaning): `0.5784`

## Strategy Comparison

| Strategy | Status | F1 | Accuracy | vs. Baseline | Agent Attempts (train+test) |
|---|---|---|---|---|---|
| conservative_median | RAN (below baseline) | 0.5417 | 0.5600 | -0.0367 | 4 |
| mean_wide_tolerance | RAN (below baseline) | 0.5257 | 0.5677 | -0.0527 | 4 |
| tight_grouping_high_temp | PASS | 0.5816 | 0.5729 | +0.0032 | 2 |

## Winner: `tight_grouping_high_temp`

F1 improved from baseline `0.5784` to `0.5816` (+0.0032).

## Failed Strategies (detail)
None -- all strategies completed successfully.