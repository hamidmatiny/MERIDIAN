"""Print paid spend. ``python -m meridian.spend``."""

from meridian.config import get_settings
from meridian.llm.budget import SpendStore, format_report


def main() -> int:
    """Print today's paid spend and the all-time paid spend, per provider."""
    from pathlib import Path

    settings = get_settings()
    store = SpendStore(Path(settings.spend_path))
    try:
        print(format_report(store))
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
