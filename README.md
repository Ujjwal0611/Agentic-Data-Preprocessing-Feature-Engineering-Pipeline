# Agentic Data Preprocessing & Feature Engineering Pipeline

A local, autonomous data-cleaning system where an LLM agent writes its own pandas cleaning code, tests it in a sandbox, and self-corrects on failure — with **zero paid API cost** and **zero data egress**. The system runs three independent cleaning strategies against a leakage-safe train/test split and reports whether agent-driven cleaning actually beats a naive baseline, with real F1 numbers, not just "the code runs."

Built entirely on a local LLM (`qwen2.5-coder:7b` via Ollama) — no OpenAI/Anthropic API calls anywhere in the pipeline.

---

## Why this exists

Most "AI agent" demos stop at "the agent ran without crashing." That's necessary but not sufficient — an agent that reports success while silently leaving text columns unencoded, or one that's evaluated on data its own cleaning step already leaked into, produces numbers you can't trust.

This project is built around three problems that are usually glossed over:

1. **Small local models don't reason abstractly — they pattern-complete.** A 7B model shown a bare traceback often repeats the exact same mistake on every retry. The system prompt here is built from real observed failures (18+ documented bugs), each fixed with an explicit correct/incorrect code pair, not a prose warning.
2. **"Success" reported by generated code can't be trusted at face value.** Every cleaning attempt is checked *twice* — once by the LLM's own in-code assertions, and independently again by a defense-in-depth structural validator that doesn't trust the code's own claim of success.
3. **Data leakage is silent and easy to introduce by accident.** Cleaning the full dataset before splitting is the single most common way ML projects produce misleadingly good numbers. Here, splitting happens first, is structurally the only way data can flow into cleaning, and is verified at runtime — not just assumed.

---

## Architecture

```
Phase 1: Baseline           Phase 2: Self-Correcting Agent      Phase 3: Orchestration
─────────────────           ──────────────────────────────      ───────────────────────
Generate messy synthetic    Profile dataset → LLM generates      Split BEFORE cleaning (leakage-safe)
loan-approval dataset       pandas cleaning code → sandbox       → run 3 cleaning strategies,
(NaNs, outliers, fuzzy      executes it → structural checks      each independently on train/test
categories, dup rows,       (defense-in-depth, doesn't trust     → align divergent one-hot columns
bad date ordering)          the code's own success claim) →      → retrain XGBoost per strategy
                             on failure, feed traceback + full    → compare F1 against baseline
Train naive XGBoost          attempt history back to LLM,        → generate executive report
baseline (drop string        retry up to 5x
columns) → save F1 as
the number to beat
```

**Data flow guarantee:** raw row-level data is never sent to the LLM. Only a structured, aggregated profile (column types, missing-value rates, ranges, sample values, detected issues) is included in any prompt.

### Module map

| Module | Responsibility |
|---|---|
| `phase1/baseline_pipeline.py` | Generates the messy synthetic dataset; trains the naive "before" baseline |
| `phase2/profiler.py` | Converts a raw DataFrame into an LLM-readable structured summary — the *only* thing about the data the model ever sees |
| `phase2/llm_client.py` | Prompts `qwen2.5-coder:7b` via Ollama for cleaning code; renders cumulative attempt history so retries don't repeat mistakes |
| `phase2/code_safety.py` | AST-based static validator — blocks dangerous imports/calls before anything executes |
| `phase2/sandbox_executor.py` | Executes generated code against a copy of the data in a restricted namespace; independently re-verifies structural correctness after execution |
| `phase2/agent.py` | The generate → validate → execute → self-correct retry loop (max 5 attempts) |
| `phase2/error_reporter.py` | Structured, categorized failure reports (JSON + Markdown) for human review |
| `phase3/leakage_guard.py` | Enforces train/test split before any cleaning; runtime-verifies split integrity; reconciles one-hot column divergence after independent cleaning |
| `phase3/strategies.py` | Named cleaning-policy configurations (imputation method, outlier tolerance, grouping aggressiveness, LLM temperature) |
| `phase3/main.py` | Orchestrator: loops all strategies, retrains, compares F1 vs. baseline |
| `phase3/reporter.py` | Generates the executive Markdown + JSON comparison report |

---

## Results

See [`data/sample/executive.md`](data/sample/executive.md) for a real run's output — actual F1 scores, per-strategy comparison, and attempt counts, not illustrative numbers.

**Test status:**
- Phase 2 edge-case suite: 7/8 passing (`malformed_dates` is a documented, deliberate known limitation — see below)
- Phase 3 integration suite (`test_orchestrator.py`): 6/6 passing, including one real end-to-end Ollama-backed run
- Full Phase 3 orchestration (all 3 strategies, real ~1000-row dataset): confirmed passing end-to-end

**Known limitations (documented, not blocking):**
- `malformed_dates`: a date column with an 80% parse rate falls just below the profiler's 90% detection threshold and gets classified as categorical instead of a date. This is treated as an accepted edge case rather than a bug fix, since deciding "is this column really a date" at 80% parseability is a genuine judgment call, not an obvious defect.
- `single_row`: an intermittent failure tied to local-model non-determinism, not yet fully root-caused. Flagged for follow-up, does not block the rest of the pipeline.

---

## Design decisions worth asking about

- **Agent is deliberately task-agnostic.** It fixes true data defects — NaNs, duplicates, fuzzy category spellings, illogical date ordering, unencoded categoricals — and nothing else. Class imbalance and outlier magnitude are treated as properties of the data, not bugs, and are explicitly left for downstream modeling decisions.
- **A statistics-based fuzzy-column detector (GMM/BIC) was built and removed.** It required graduate-level statistical assumptions I couldn't confidently defend under interview questioning, so it was cut in favor of a simpler, fully explainable approach — a deliberate scope decision, not a missed feature.
- **Structural correctness ≠ semantic correctness.** The sandbox can prove "every column is numeric" or "zero NaNs remain" without any domain knowledge. It cannot prove a particular one-hot grouping was the *right* grouping. That question is only answerable by Phase 3's F1-vs-baseline comparison — the sandbox intentionally does not try to fake an answer to a question it can't actually check.

Full rationale, module-by-module theory, and an exhaustive interview Q&A archive are maintained in [`docs/AUTOMATION_LEDGER.md`](docs/AUTOMATION_LEDGER.md).

---

## Running it locally

Windows only (PowerShell). Requires [Ollama](https://ollama.com) running locally with `qwen2.5-coder:7b` pulled.

```powershell
# Setup
uv sync
ollama pull qwen2.5-coder:7b
ollama serve   # separate terminal, must stay running

# 1. Generate messy dataset + naive baseline
uv run python -m agentic_data_pipeline.phase1.baseline_pipeline

# 2. Verify the self-correcting agent against edge cases (~3-5 min)
uv run python -m agentic_data_pipeline.phase2.test_edge_cases_reporting

# 3. Run the Phase 3 integration test suite (~2 min)
uv run python -m agentic_data_pipeline.phase3.test_orchestrator

# 4. Full orchestration run: all strategies, real dataset (~10-20 min)
uv run python -m agentic_data_pipeline.phase3.main

# 5. Read the results
Get-Content data\phase3_outputs\reports\executive.md
```

---

## Stack

`uv` · Python 3.11 · pandas 3.0 · XGBoost · scikit-learn · `rapidfuzz` · Ollama (`qwen2.5-coder:7b`, local GPU) · pytest-style module test scripts

---

## Project status

Phases 1–3 complete and passing. See [`docs/AUTOMATION_LEDGER.md`](docs/AUTOMATION_LEDGER.md) for the full development log, every bug found and fixed, and design-decision rationale.
