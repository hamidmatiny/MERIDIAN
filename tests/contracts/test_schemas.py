"""Committed JSON Schemas must match a fresh export, including untracked files."""

import subprocess
from pathlib import Path

from meridian.schema_export import CONTRACT_SCHEMAS, export_schemas

ROOT = Path(__file__).resolve().parents[2]
COMMITTED = ROOT / "schemas"


def _texts(directory: Path) -> dict[str, str]:
    return {
        path.name: path.read_text(encoding="utf-8") for path in sorted(directory.glob("*.json"))
    }


def test_export_writes_one_file_per_contract(tmp_path: Path) -> None:
    written = export_schemas(tmp_path)
    assert [path.name for path in written] == [f"{stem}.json" for stem, _source in CONTRACT_SCHEMAS]
    for path in written:
        text = path.read_text(encoding="utf-8")
        assert text.endswith("\n")
        assert len(text) > 2


def test_export_is_stable_and_drops_stale_files(tmp_path: Path) -> None:
    stale = tmp_path / "obsolete.json"
    stale.write_text("{}\n", encoding="utf-8")
    first = _texts(export_schemas(tmp_path)[0].parent)
    assert "obsolete.json" not in first
    export_schemas(tmp_path)
    assert _texts(tmp_path) == first


def test_schema_tree_has_no_untracked_or_modified_files() -> None:
    export_schemas(COMMITTED)
    status = subprocess.run(
        ["git", "status", "--porcelain", "schemas/"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert status.stdout == ""
