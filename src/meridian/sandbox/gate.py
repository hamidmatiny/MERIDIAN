"""Reject remediation source before it is allowed to run."""

import ast

from meridian.config import REGRESSION_MODULES, REPAIR_MODULES

_BANNED_MODULES = frozenset({"os", "sys", "subprocess", "socket", "pathlib", "importlib"})
_BANNED_NAMES = frozenset({"open", "exec", "eval", "compile", "__import__", "__builtins__"})


def check_source(source: str, *, allow_pandera: bool = False) -> list[str]:
    """Return rejection reasons. An empty list means the source may run.

    The repair module may import the modules in ``REPAIR_MODULES``. A regression
    test may import ``REGRESSION_MODULES``, which adds pandera. Anything else,
    including dunder access, is rejected.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error: {exc.msg}"]
    allowed = set(REGRESSION_MODULES if allow_pandera else REPAIR_MODULES)
    reasons: list[str] = []
    for node in ast.walk(tree):
        reason = _reason(node, allowed)
        if reason is not None and reason not in reasons:
            reasons.append(reason)
    return reasons


def _reason(node: ast.AST, allowed: set[str]) -> str | None:
    if isinstance(node, ast.Import):
        for alias in node.names:
            return _module(alias.name, allowed)
        return None
    if isinstance(node, ast.ImportFrom):
        if node.level != 0 or node.module is None:
            return "relative imports are not allowed"
        return _module(node.module, allowed)
    if isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
        return f"name {node.id} is not allowed"
    if isinstance(node, ast.Attribute):
        if node.attr.startswith("__"):
            return f"attribute {node.attr} is not allowed"
        if isinstance(node.value, ast.Name) and node.value.id == "__builtins__":
            return "access to __builtins__ is not allowed"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in _BANNED_NAMES:
            return f"call to {node.func.id} is not allowed"
        if node.func.id == "getattr" and _getattr_unsafe(node):
            return "getattr on a dunder is not allowed"
    return None


def _module(name: str, allowed: set[str]) -> str | None:
    root = name.split(".", maxsplit=1)[0]
    if root in _BANNED_MODULES or root not in allowed:
        return f"import {name} is not allowed"
    return None


def _getattr_unsafe(node: ast.Call) -> bool:
    if len(node.args) < 2 or not isinstance(node.args[1], ast.Constant):
        return True
    value = node.args[1].value
    if not isinstance(value, str):
        return True
    return value.startswith("__")
