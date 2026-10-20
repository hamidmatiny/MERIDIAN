"""The import surface stays off Slack and the critic provider.

MCP and the MCP adapter are imported only by the tool servers.
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
}
MCP_MODULES = {"mcp", "langchain_mcp_adapters"}


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


def test_mcp_imports_stay_in_the_servers() -> None:
    offenders: list[str] = []
    for path in sorted(ROOT.rglob("*.py")):
        if "mcp_servers" in path.parts:
            continue
        imported = _imported_modules(path) & MCP_MODULES
        if imported:
            offenders.append(f"{path.relative_to(ROOT.parent.parent)}: {sorted(imported)}")
    assert offenders == []
