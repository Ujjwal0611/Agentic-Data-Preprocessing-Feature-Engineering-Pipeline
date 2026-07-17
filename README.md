# Agentic Data Preprocessing & Feature Engineering Pipeline

A local, autonomous data-cleaning system where an LLM agent writes its own pandas cleaning code, tests it in a sandbox, and self-corrects on failure — with **zero paid API cost** and **zero data egress**. Three independent cleaning strategies run against a leakage-safe train/test split, and the system reports whether agent-driven cleaning actually beats a naive baseline, with real F1 numbers — not just "the code ran."

Built entirely on a local LLM (`qwen2.5-coder:7b` via Ollama). No OpenAI/Anthropic API calls anywhere in the pipeline.

This is a proof of concept for a **specific, bounded problem area** — see [Scope & Limitations](#scope--limitations) below for exactly where that boundary sits, and [Real-World Validation](#real-world-validation-uci-adultcensus-income) for evidence of what happens when the pipeline meets data it wasn't built around.

---

## Why this exists

Most "AI agent" demos stop at "the agent ran without crashing." That's necessary but not sufficient — an agent that reports success while silently leaving text columns unencoded, or one evaluated on data its own cleaning step already leaked into, produces numbers you can't trust.

This project is built around three problems that are usually glossed over:

1. **Small local models don't reason abstractly — they pattern-complete.** A 7B model shown a bare traceback often repeats the exact same mistake on every retry. The system prompt here is built from real observed failures (22 documented bugs), each fixed with an explicit correct/incorrect code pair, not a prose warning.
2. **"Success" reported by generated code can't be trusted at face value.** Every cleaning attempt is checked twice — once by the LLM's own in-code assertions, and independently again by a defense-in-depth structural validator that doesn't trust the code's own claim of success.
3. **Data leakage is silent and easy to introduce by accident.** Cleaning the full dataset before splitting is the single most common way ML projects produce misleadingly good numbers. Here, splitting happens first, is structurally the only way data can flow into cleaning, and is verified at runtime — not just assumed.

---

## Module flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 1 — Baseline                                                  │
│                                                                       │
│   Generate messy dataset  ──────▶  Train naive XGBoost baseline      │
│   (NaNs, outliers, dupes,          (drop string columns,             │
│    fuzzy categories, bad           save F1 to metrics.json —         │
│    date ordering)                  the number every strategy         │
│                                     in Phase 3 must beat)             │
└──────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 2 — Self-correcting cleaning agent           (loop, ≤5×)      │
│                                                              │        │
│   profiler.py                                               │        │
│   Raw DataFrame → structured text summary.                  │        │
│   No row-level data ever leaves this step.                  │        │
│         │                                                    │        │
│         ▼                                                    │        │
│   llm_client.py                                              │        │
│   qwen2.5-coder:7b (via Ollama) generates cleaning code       │        │
│   from the profile + prior failed attempts, if any.          │        │
│         │                                                    │        │
│         ▼                                                    │        │
│   code_safety.py                                              │       │
│   AST static check — blocks dangerous imports/calls           │       │
│   BEFORE anything executes.                                   │       │
│         │                                                    │        │
│         ▼                                                    │        │
│   sandbox_executor.py                                         │       │
│   Executes code in a restricted namespace against a COPY      │       │
│   of the data. Independently RE-VERIFIES the result —         │       │
│   never trusts the code's own claim of success.               │       │
│         │                                                    │        │
│         ├── fail ──────────────────────────────────────────▶─┘        │
│         │   (append attempt to history, retry with           (up to  │
│         │    full context of what already failed)            5x)    │
│         ▼                                                             │
│   success → cleaned DataFrame                                         │
│   exhausted → attempt log dumped to disk for human review             │
└──────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 3 — Multi-strategy orchestration                               │
│                                                                        │
│   leakage_guard.split_before_cleaning()                                │
│   ONLY sanctioned split point. Verified at runtime — not just          │
│   assumed. (Row counts reconcile, no duplicate row crosses the         │
│   train/test boundary.)                                                │
│         │                                                              │
│         ▼                                                              │
│   for each of 3 strategies in STRATEGY_LIBRARY:                        │
│         │                                                              │
│         ├── clean TRAIN split ──┐   (via Phase 2 agent,                │
│         ├── clean TEST split ───┤    completely independently —        │
│         │                       │    this is the leakage boundary)     │
│         ▼                       ▼                                     │
│   align_train_test_columns()                                           │
│   Zero-fill one-hot column gaps caused by independent cleaning         │
│   (not leakage — pure bookkeeping after cleaning is already done)      │
│         │                                                              │
│         ▼                                                              │
│   Retrain XGBoost, score F1 ── compare against Phase 1 baseline        │
│         │                                                              │
│         ▼                                                              │
│   reporter.py → executive.md + comparison.json                        │
│   (repeats for every strategy, even after one already wins)           │
└─────────────────────────────────────────────────────────────────────┘
```

### Module map

| Module | Responsibility |
|---|---|
| `phase1/baseline_pipeline.py` | Generates the messy synthetic dataset; trains the naive "before" baseline |
| `phase2/profiler.py` | Converts a raw DataFrame into an LLM-readable structured summary — the only thing about the data the model ever sees |
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

See [`data/sample/executive.md`](data/sample/executive.md) for a real Phase 3 run's output on the synthetic dataset — actual F1 scores, per-strategy comparison, and attempt counts.

**Test status:**
- Phase 2 edge-case suite: 7/8 passing (`malformed_dates` is a documented, deliberate known limitation)
- Phase 3 integration suite (`test_orchestrator.py`): 6/6 passing, including one real end-to-end Ollama-backed run
- Full Phase 3 orchestration (all 3 strategies, real ~1,000-row synthetic dataset): confirmed passing end-to-end

---

## Real-world validation: UCI Adult/Census Income

Rather than claim the pipeline generalizes beyond the synthetic dataset it was tuned against, it was tested — unmodified, zero code changes made in advance — against a real, well-known messy dataset: **UCI's Adult/Census Income dataset** (32,561 rows, 15 columns, real missing values encoded as literal `"?"` strings, mixed numeric/categorical columns).

**First run (unmodified pipeline): failed.**
The agent exhausted all 5 retries (86.7s). Diagnosis found two distinct issues:
1. Attempt 1 was a genuine, unremarkable first-try miss — the `native_country` column was left almost entirely unimputed (32,537 of 32,561 rows still NaN) and 38 duplicate rows remained. This is exactly the kind of mistake the self-correction loop exists to fix.
2. It never got the chance. Attempts 2 through 5 were all blocked by a **newly discovered infrastructure bug** (documented as Bug #22): generated code tried to `raise ValueError(...)`, but `ValueError` wasn't exposed in the sandbox's restricted builtins. The resulting `NameError: name 'ValueError' is not defined` gave the model no actionable signal, so it repeated the identical broken pattern on every remaining retry.

**Fix applied:** added `ValueError`, `TypeError`, `KeyError`, `AssertionError`, `IndexError`, `AttributeError`, and `Exception` to the sandbox's builtin allow-list — generated validation code legitimately needs to raise these.

**Second run (same dataset, one fix, no other changes): succeeded on attempt 1/5, in 17.4s** (down from 86.7s — roughly 5x faster).

| | Before fix | After fix |
|---|---|---|
| Result | Failed (exhausted 5/5 attempts) | Succeeded (1/5 attempts) |
| Time | 86.7s | 17.4s |
| Cleaned shape | — | `[32537, 77]` |
| Non-numeric columns remaining | — | 0 |
| NaNs remaining | — | 0 |
| Row retention | — | 99.9% (32,561 → 32,537) |

The row loss (24 rows) exactly matches the profiler's independently reported duplicate-row count for this dataset — the agent deduplicated correctly rather than over-aggressively dropping rows to dodge the missing-value problem, and every one of the 15 original columns (a mix of numeric and categorical) was fully encoded into 77 final numeric columns via real one-hot encoding, not a passthrough.

**What this demonstrates:** the underlying cleaning logic wasn't broken by real-world scale or messiness — it was blocked by one identifiable, fixable infrastructure gap. That's a materially different (and more useful) finding than either "it just works" or "it doesn't generalize" — it's evidence-backed, and it's exactly the kind of gap a stress test against real data is supposed to surface.

---

## Scope & limitations

This project is a well-engineered proof of concept for a **bounded problem class**: tabular CSV data exhibiting the specific defect types it was deliberately built and iteratively hardened against — missing values, exact duplicates, fuzzy-spelled categories, illogical date ordering, unencoded categorical/date columns. It is not yet validated as a general-purpose solution for arbitrary real-world data. Specific, known gaps:

- **Prompt size scales with column count.** The Adult dataset's 15-column profile was already ~3,461 characters (~865 tokens) versus a few hundred characters for the synthetic 8-column dataset. Datasets with 50-200+ columns haven't been tested and may exceed what a 7B local model attends to reliably — likely producing silently degraded generation rather than a clean error.
- **High-cardinality handling is tuned to one specific edge case** (a 147-unique-value column). Real-world extremes — free-text fields, unique IDs, tens of thousands of unique values — would technically trigger the same top-K-plus-"other" grouping mechanism, but whether the result is semantically meaningful for prediction is unverified.
- **File-parsing-level messiness is untested.** Mixed encodings, BOM markers, locale-specific number/date formats, and embedded commas/newlines in quoted fields are common in real CSVs and absent from every dataset tested so far — every documented bug to date is about cleaning logic, not ingestion.
- **Relational/logical defects beyond date-pairs aren't detected.** The profiler's date-order-violation check is the only cross-column logical-consistency check that exists; domain-specific rules (e.g., "balance can't be negative for this account type") have no general detection mechanism.
- **Scale, both rows and compute cost.** The sandbox deep-copies the full DataFrame on every execution attempt; behavior at 100K+ rows or very wide datasets hasn't been systematically measured for memory or wall-clock cost.
- **Combinatorial edge cases.** All 22 documented bugs were found and fixed largely in isolation. Real data frequently has several defect types compounding in the same column simultaneously — this hasn't been systematically tested.
- **Single-run non-determinism.** `qwen2.5-coder:7b` is confirmed non-deterministic (the same edge case has passed reliably in one run and failed identically across all retries in another with zero code changes). Individual run results, including the real-world validation above, should be read as evidence, not a permanent guarantee — repeated runs would strengthen the claim further.

Stating these explicitly is a deliberate choice: a scope boundary stated up front is more defensible than an unqualified claim of generality discovered to be false later.

---

## Future expansions

**Scaling the LLM itself.** The entire zero-cost, zero-egress design constraint — a single local 7B model on one machine — is what forces so much of this system's engineering (extremely prescriptive prompting, defense-in-depth structural checks, cumulative attempt history) to compensate for a small model's specific weaknesses. That constraint doesn't have to hold in a production/enterprise context. A natural next step: swap in a larger open-weight model (e.g., in the 30B-70B+ class) served across a multi-GPU or multi-node cluster internal to a company, which would let many users run this pipeline concurrently against much larger and more varied datasets, with a model that reasons more reliably from abstract error tracebacks and needs less hand-holding in the prompt. The architecture doesn't need to change to support this — `llm_client.py`'s `model_name` parameter and the Ollama-based call pattern are already decoupled from any specific model size; a production deployment would swap the backend (e.g., a self-hosted vLLM or TGI cluster serving a larger open-weight model) behind the same interface, and could relax some of the more defensive prompt engineering that exists specifically to accommodate a 7B model's pattern-completion tendency.

**Handling more complex ambiguity.** Several of the scope limitations above point at the same underlying theme — the system currently handles unambiguous, structurally-detectable defects well (a NaN is a NaN, an exact duplicate is exact) but has no mechanism for genuinely ambiguous judgment calls. Concrete directions:
- A confidence-scored profiler output (e.g., "this column is 80% parseable as a date — flag for human review" instead of a hard threshold cutoff) rather than the current binary classification that produces the `malformed_dates` limitation.
- A general-purpose relational/business-rule checker that goes beyond the current date-pair-only detection — potentially LLM-assisted itself, where the model proposes plausible cross-column constraints from the profile and a human confirms or rejects them before they're enforced.
- Ensemble or multi-sample generation (asking the model for N candidate cleaning approaches per attempt instead of one, then picking via a secondary check) to reduce reliance on any single generation's correctness — particularly valuable given the confirmed non-determinism.
- Extending the profiler to detect encoding and locale issues (mixed character sets, ambiguous date formats, thousands-separator conventions) as a distinct, explicitly-flagged category, rather than letting them surface only as opaque parsing failures downstream.
- A human-in-the-loop review queue for cases the structural validator can prove are *safe* but can't prove are *correct* (per the structural-vs-semantic-correctness distinction throughout this project) — surfacing exactly the cases where the sandbox's guarantees run out, rather than silently accepting or blindly rejecting them.

---

## Design decisions worth asking about

- **Agent is deliberately task-agnostic.** It fixes true data defects and nothing else — class imbalance and outlier magnitude are treated as properties of the data, not bugs, and are left for downstream modeling decisions.
- **A statistics-based fuzzy-column detector (GMM/BIC) was built and removed.** It required graduate-level statistical assumptions that weren't confidently defensible under interview questioning, so it was cut in favor of a simpler, fully explainable approach.
- **Structural correctness ≠ semantic correctness.** The sandbox can prove "every column is numeric" without any domain knowledge. It cannot prove a particular one-hot grouping was the *right* grouping — that question is only answerable by Phase 3's F1-vs-baseline comparison.

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

# Optional: stress-test against a real external CSV, unmodified
uv run python -m agentic_data_pipeline.real_world_stress_test data\external\your_dataset.csv target_column_name
```

---

## Stack

`uv` · Python 3.11 · pandas 3.0 · XGBoost · scikit-learn · `rapidfuzz` · Ollama (`qwen2.5-coder:7b`, local GPU) · pytest-style module test scripts

---
