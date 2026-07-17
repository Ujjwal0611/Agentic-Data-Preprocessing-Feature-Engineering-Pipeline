

from __future__ import annotations

import logging
import re

from ollama import chat

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported for type hints -- avoids a real runtime dependency from
    # phase2 -> phase3, which would be backwards (phase3 depends on phase2,
    # not the other way around).
    from agentic_data_pipeline.phase3.strategies import CleaningStrategy

logger = logging.getLogger("llm_client")

DEFAULT_MODEL_NAME: str = "qwen2.5-coder:7b"

SYSTEM_PROMPT: str = (
    "You are a senior data engineer writing pandas code to clean a messy "
    "tabular dataset. You will be given a structured profile of the "
    "dataset's columns, their missing-value rates, ranges, unique values, "
    "and any flagged data-quality issues. ONLY reference columns that are "
    "explicitly listed in the profile -- never assume a column exists "
    "(e.g. do not assume a 'date' column exists unless it is listed).\n\n"
    "Rules you MUST follow:\n"
    "1. Output ONLY a single Python code block, nothing else -- no prose, "
    "no explanation, no markdown outside the code fence.\n"
    "2. The code must operate on a pre-existing pandas DataFrame variable "
    "named `df` and must reassign the cleaned result back to `df`.\n"
    "3. Never modify, drop, or reference the target column by name in a "
    "way that would leak it into other features.\n"
    "4. Never import any module beyond pandas (`pd`), numpy (`np`), and "
    "datetime, which are already available in scope.\n"
    "5. NEVER use `inplace=True` on any pandas operation -- pandas' "
    "Copy-on-Write behavior means inplace chained assignment silently does "
    "NOT modify the DataFrame. Always reassign explicitly: "
    "`df['col'] = df['col'].fillna(x)`.\n"
    "6. The `.str` accessor (`.str.strip()`, `.str.lower()`, etc.) ONLY "
    "works on a single column (a Series), NEVER on the whole DataFrame:\n"
    "   WRONG: df.str.strip()\n"
    "   WRONG: df['col'].strip()\n"
    "   RIGHT: df['col'] = df['col'].str.strip()\n"
    "7. `DataFrame.applymap()` has been REMOVED in pandas 3.0+ and will "
    "raise AttributeError. Never use it. For per-element transforms on a "
    "single column, use: df['col'] = df['col'].apply(func)\n"
    "8. NEVER call `.quantile()`, `.mean()`, `.std()`, or `.clip()` on the "
    "ENTIRE DataFrame if it contains text/date columns, or with "
    "multi-column Series bounds -- both raise errors (TypeError on string "
    "subtraction, or ValueError on axis ambiguity). ALWAYS process numeric "
    "columns ONE AT A TIME in a loop:\n"
    "   numeric_cols = df.select_dtypes(include='number').columns\n"
    "   for col in numeric_cols:\n"
    "       Q1 = df[col].quantile(0.25)\n"
    "       Q3 = df[col].quantile(0.75)\n"
    "       IQR = Q3 - Q1\n"
    "       df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)\n"
    "9. EVERY column except the target must be numeric before the code "
    "finishes -- this includes date columns, and includes columns with "
    "only 1 unique value (drop or encode them, never leave as text). Do "
    "NOT hardcode individual column names one at a time -- discover and "
    "encode ALL of them programmatically in one step:\n"
    "   categorical_cols = [c for c in df.select_dtypes(include=['object', 'str']).columns "
    "if c != 'target']\n"
    "   For cardinality < 10 per column: df = pd.get_dummies(df, columns=categorical_cols, "
    "drop_first=True)\n"
    "   For any column with cardinality > 50: first reduce it by grouping rare values "
    "into 'other' (see rule 18) BEFORE including it in categorical_cols.\n"
    " After this step, verify no text columns remain. NEVER use a bare "
    "`assert` here -- if it fails, a bare assert gives NO information about "
    "which column is still wrong, and you will make the same mistake on "
    "every retry. ALWAYS include an f-string message naming the offending "
    "columns:\n"
    "remaining_text_cols = df.select_dtypes(include=['object','str']).columns.difference(['target']).tolist()\n"
    "assert not remaining_text_cols, f'Still non-numeric: {remaining_text_cols}'\n"
    "10. Date columns must ALSO end up numeric (they are not exempt from "
    "rule 9). Convert each date column with EXACTLY this pattern -- do not "
    "improvise:\n"
    "   EPOCH = pd.Timestamp('1970-01-01')\n"
    "   for col in [<the date columns from the profile>]:\n"
    "       parsed = pd.to_datetime(df[col], errors='coerce', format='mixed')\n"
    "       df[col] = (parsed - EPOCH).dt.total_seconds()\n"
    "       df[col] = df[col].fillna(df[col].median())\n"
    "   NEVER use .astype('int64') on a datetime column -- unparseable "
    "dates (NaT) silently become the garbage sentinel value -9223372037 "
    "instead of NaN. The subtraction pattern above turns them into real "
    "NaN so they can be imputed correctly.\n"
    "11. NEVER assign encoded columns back into a subset of the frame. "
    "`df[['a','b']] = pd.get_dummies(...)` and `df['col'] = "
    "pd.get_dummies(df['col'])` BOTH raise 'ValueError: Columns must be "
    "same length as key', because get_dummies returns a different number "
    "of columns than you assigned to. The ONLY correct form reassigns the "
    "WHOLE frame:\n"
    "   df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)\n"
    "12. After ANY row-dropping operation (`drop_duplicates()`, boolean-"
    "mask filtering, `dropna()`), immediately call "
    "`df = df.reset_index(drop=True)` before using `.loc[]`/`.iloc[]` "
    "with positional-looking labels -- stale index labels cause KeyError.\n"
    "13. Some 'missing' values arrive as literal strings, not NaN: 'NA', "
    "'N/A', 'null', 'None', '-'. pandas does NOT auto-convert these when "
    "the DataFrame is built directly in code (only read_csv does this). "
    "Before any NaN-handling step, run:\n"
    "   df = df.replace(['NA', 'N/A', 'null', 'None', '-'], np.nan)\n"
    "14. NEVER resolve missing values with a blanket `df.dropna()` across "
    "the whole DataFrame -- this can silently delete every row if NaNs "
    "are spread across many columns. Handle missing values per-column: "
    "impute (`fillna` with mean/median/mode as appropriate) or drop only "
    "the specific column/rows that are unsalvageable.\n"
    "15. If the profile reports duplicate rows, drop them with "
    "df.drop_duplicates(keep='first'), then reset_index (see rule 12).\n"
    "16. If the profile reports near-duplicate category spellings, "
    "normalize them (lowercase, strip whitespace, map remaining variants "
    "to one canonical value) before encoding.\n"
    "17. If the profile reports date-order violations between two "
    "columns, use the column names to judge the logically correct order "
    "and DROP the violating rows entirely, e.g. df = df[~violating_mask], "
    "then reset_index (see rule 12). Do NOT try to set individual "
    "date/string values to NaN/null in place -- pandas' strict string "
    "dtype will raise TypeError for non-string assignments. Dropping the "
    "18. For text columns with more than 50 unique values, do NOT "
    "enumerate every category. Instead, keep only the top 20 most "
    "frequent values, group the rest into 'other', THEN encode "
    "immediately in the same block -- never leave a grouped column as "
    "text waiting for a later step:\n"
    "top_vals = df['col'].value_counts().nlargest(20).index\n"
    "df['col'] = df['col'].where(df['col'].isin(top_vals), 'other')\n"
    "df = pd.get_dummies(df, columns=['col'], drop_first=True)\n"
    "19. NEVER use `df.loc[:, col] = ...` to change a column's dtype. In "
    "pandas 3.0 `.loc` assigns IN PLACE and preserves the existing dtype, "
    "so writing numbers into a datetime or string column raises TypeError. "
    "To replace a column and its dtype, always use plain bracket "
    "assignment: `df[col] = <new values>`.\n"
    "20. NEVER call `pd.to_datetime(df)` on the whole DataFrame -- pandas "
    "will try to assemble a date from year/month/day COLUMNS and raise "
    "'to assemble mappings requires...'. Only ever call it on a single "
    "column: `pd.to_datetime(df[col], errors='coerce', format='mixed')`.\n"
    "21. NEVER use `.astype(int)` or `.astype('int64')` to make a TEXT "
    "column numeric -- it raises 'invalid literal for int()'. Text columns "
    "become numeric ONLY via pd.get_dummies (rule 9). astype(int) is for "
    "columns that already hold numeric values.\n"
    "22. Before running the general NaN-imputation step, check for columns "
    "that are 100% missing (every value is NaN) and DROP them entirely -- "
    "do NOT try to impute them:\n"
    "   all_nan_cols = df.columns[df.isna().all()].tolist()\n"
    "   df = df.drop(columns=all_nan_cols)\n"
    "   fillna(df[col].median()) is a NO-OP on a fully-empty column, since "
    "the median of zero valid values is itself NaN -- it will silently "
    "leave the column unresolved no matter how many times you retry it.\n"
    "23. Process each column EXACTLY ONCE, in this fixed order, and never "
    "re-touch a column after converting it: (a) drop 100%-empty columns "
    "(rule 22), (b) normalize fuzzy spellings ONLY on remaining TEXT "
    "columns (rule 16) -- never on date columns, (c) convert date columns "
    "to numeric (rule 10), (d) one-hot encode remaining text columns "
    "(rule 9). NEVER call `.str` accessor methods on a column after it has "
    "already been converted to numeric by an earlier step in your own "
    "code -- if you get 'Can only use .str accessor with string values, "
    "not floating', you are re-processing an already-converted column.\n"
)
# ADD near the top, after the SYSTEM_PROMPT constant
from dataclasses import dataclass
MAX_HISTORY_ATTEMPTS_IN_PROMPT: int = 3  # cap it -- see note below
@dataclass(frozen=True)
class AttemptRecord:
    """One failed attempt: what the LLM wrote, and exactly what broke."""
    attempt_number: int
    code: str
    error: str


def _render_attempt_history(history: list["AttemptRecord"]) -> str:
    """Render the most recent failed attempts so the LLM can see the full
    pattern of what it's already tried, not just the last error."""
    if not history:
        return ""
    recent = history[-MAX_HISTORY_ATTEMPTS_IN_PROMPT:]
    blocks = [
        f"--- Attempt {r.attempt_number} (FAILED) ---\n"
        f"Code:\n```python\n{r.code}\n```\n"
        f"Error:\n{r.error}\n"
        for r in recent
    ]
    return (
        f"You have already failed {len(history)} time(s) on this dataset. "
        f"Below are your most recent failed attempts. Do NOT repeat the same "
        f"code or the same mistake twice -- if two attempts share a root "
        f"cause, try a genuinely different approach:\n\n" + "\n".join(blocks)
    )

CODE_BLOCK_PATTERN = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


class LLMGenerationError(Exception):
    """Raised when the LLM response cannot be parsed into executable code."""


def _extract_code_block(raw_response_text: str) -> str:
    """Pull the Python code out of a markdown-fenced LLM response.

    Falls back to treating the entire response as code if no fence is
    found, since small local models don't always fence consistently.
    """
    match = CODE_BLOCK_PATTERN.search(raw_response_text)
    if match:
        return match.group(1).strip()
    stripped = raw_response_text.strip()
    if not stripped:
        raise LLMGenerationError("LLM returned an empty response.")
    return stripped
    
def generate_cleaning_code(
    dataset_profile_text: str,
    model_name: str = DEFAULT_MODEL_NAME,
    attempt_history: list[AttemptRecord] | None = None,
    strategy: "CleaningStrategy | None" = None,
) -> str:
    history_block = _render_attempt_history(attempt_history or [])
    strategy_block = strategy.to_prompt_directive_block() if strategy is not None else ""

    prompt_parts = [f"Dataset profile:\n{dataset_profile_text}"]
    if strategy_block:
        prompt_parts.append(strategy_block)
    if history_block:
        prompt_parts.append(history_block)
        prompt_parts.append("Write a corrected, complete version of the cleaning code now.")
    else:
        prompt_parts.append("Write the cleaning code now.")
    user_prompt = "\n\n".join(prompt_parts)

    chat_kwargs: dict = dict(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    if strategy is not None:
        # Different strategies use different generation temperatures on
        # purpose (see strategies.py) -- higher temp on later retries helps
        # escape a repeated bad pattern instead of regenerating near-
        # identical broken code.
        chat_kwargs["options"] = {"temperature": strategy.temperature}

    try:
        response = chat(**chat_kwargs)
    except Exception as exc:
        raise LLMGenerationError(
            f"Ollama call failed for model '{model_name}'. Is `ollama serve` "
            f"running and has `ollama pull {model_name}` been run? "
            f"Original error: {exc}"
        ) from exc

    raw_text = response.message.content or ""
    code = _extract_code_block(raw_text)
    logger.info("LLM generated %d chars of code using model '%s'.", len(code), model_name)
    return code