"""The committed env template holds placeholders, never a live credential."""

import re
from pathlib import Path

_FORBIDDEN = (
    re.compile("sk-" + "ant-"),
    re.compile("xai-"),
    re.compile("xoxb-" + r"\d"),
    re.compile("xapp-" + r"\d"),
)


def test_env_example_values_are_placeholders() -> None:
    text = Path(".env.example").read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#") or "=" not in stripped:
            continue
        _name, value = stripped.split("=", 1)
        for pattern in _FORBIDDEN:
            assert pattern.search(value) is None, stripped
