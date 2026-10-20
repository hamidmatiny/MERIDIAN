"""The import surface stays off Slack, MCP, and the critic provider.

Phase 2 imports Ray, LangGraph, and langchain-anthropic.
"""

import ast
from pathlib import Path

import meridian

ROOT = Path(__file__).resolve().parents[1] / "src" / "meridian"
BANNED = {
    "slack_bolt",
    "slack_sdk",
    "langchain_xai",
    "anthropic",
    "mcp",
}


def test_version_is_exposed() -> None:
    assert meridian.__version__ == "0.1.0"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            found.add(node.module.split(".", maxsplit=1)[0])
    return found


def test_sources_do_not_import_agent_runtimes() -> None:
    paths = sorted(ROOT.rglob("*.py"))
    assert paths, f"no Python sources found under {ROOT}"
    offenders: list[str] = []
    for path in paths:
        banned = _imported_modules(path) & BANNED
        if banned:
            offenders.append(f"{path.relative_to(ROOT.parent.parent)}: {sorted(banned)}")
    assert offenders == []
