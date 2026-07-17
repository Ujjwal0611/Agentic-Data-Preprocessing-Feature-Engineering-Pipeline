"""
code_safety.py

AST-based static validator for LLM-generated pandas code. Runs BEFORE any
generated code is executed. This is a defense-in-depth allow/block-list
mechanism, not a full sandbox -- see sandbox_executor.py for the runtime
containment layer that runs alongside this.
"""

from __future__ import annotations

import ast
import logging

logger = logging.getLogger("code_safety")

BLOCKED_MODULE_NAMES: frozenset[str] = frozenset(
    {
        "os", "sys", "subprocess", "socket", "shutil", "pathlib",
        "requests", "urllib", "http", "ftplib", "smtplib", "ctypes",
        "multiprocessing", "threading", "importlib",
    }
)

BLOCKED_CALL_NAMES: frozenset[str] = frozenset(
    {"eval", "exec", "compile", "open", "__import__", "input", "vars", "globals", "locals"}
)

MAX_CODE_LENGTH_CHARS: int = 8_000


class UnsafeCodeError(Exception):
    """Raised when generated code fails static safety validation."""


def validate_code_safety(code: str) -> None:
    """Statically validate that generated code contains no dangerous
    imports or built-in calls, and is syntactically valid Python."""
    if len(code) > MAX_CODE_LENGTH_CHARS:
        raise UnsafeCodeError(
            f"Generated code is suspiciously long ({len(code)} chars, "
            f"max allowed {MAX_CODE_LENGTH_CHARS}). Refusing to execute."
        )

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise UnsafeCodeError(f"Generated code has a syntax error: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
            blocked = BLOCKED_MODULE_NAMES.intersection(names)
            if blocked:
                raise UnsafeCodeError(f"Blocked import(s) detected: {sorted(blocked)}")

        elif isinstance(node, ast.ImportFrom):
            root_module = (node.module or "").split(".")[0]
            if root_module in BLOCKED_MODULE_NAMES:
                raise UnsafeCodeError(f"Blocked import detected: {root_module}")

        elif isinstance(node, ast.Call):
            func_name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if func_name in BLOCKED_CALL_NAMES:
                raise UnsafeCodeError(f"Blocked function call detected: {func_name}()")

    logger.debug("Code passed static safety validation (%d chars).", len(code))