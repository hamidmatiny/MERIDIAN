"""Write contract JSON Schemas into ``schemas/``."""

from pathlib import Path

from meridian.schema_export import export_schemas


def main() -> int:
    """Export every contract schema next to this repository's ``schemas`` directory."""
    destination = Path(__file__).resolve().parents[1] / "schemas"
    for path in export_schemas(destination):
        print(path.relative_to(destination.parent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
