"""Combined size of the local models a run is configured to load."""

import subprocess

from meridian.config import Settings

_UNITS = {"MB": 1 / 1024, "GB": 1.0, "TB": 1024.0}


def configured_models(settings: Settings) -> list[str]:
    """Unique Ollama model tags for the four nodes. A shared tag is counted once."""
    names: list[str] = []
    for name in (
        settings.triage_model,
        settings.rootcause_model,
        settings.fix_model,
        settings.critic_model,
    ):
        if name and name not in names:
            names.append(name)
    return names


def parse_ollama_list(text: str) -> dict[str, float]:
    """NAME ID SIZE MODIFIED, with SIZE as a number and a unit token."""
    sizes: dict[str, float] = {}
    for line in text.splitlines():
        parts = line.split()
        unit_at = next((index for index, token in enumerate(parts) if token in _UNITS), None)
        if unit_at is None or unit_at == 0 or len(parts) < 3:
            continue
        try:
            amount = float(parts[unit_at - 1])
        except ValueError:
            continue
        sizes[parts[0]] = amount * _UNITS[parts[unit_at]]
    return sizes


def read_ollama_sizes() -> dict[str, float]:
    """Sizes from ``ollama list``. An empty map means the command did not answer."""
    try:
        completed = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return {}
    if completed.returncode != 0:
        return {}
    return parse_ollama_list(completed.stdout)


def size_report(settings: Settings, sizes: dict[str, float]) -> str:
    """One line naming each configured model and the combined size."""
    names = configured_models(settings)
    missing = [name for name in names if name not in sizes]
    if missing:
        return "cannot read model size for " + ", ".join(missing)
    total = sum(sizes[name] for name in names)
    listed = "; ".join(f"{name} {sizes[name]:.1f} GB" for name in names)
    return f"combined local models {total:.1f} GB (cap {settings.max_model_gb:.0f} GB): {listed}"


def budget_problem(settings: Settings, sizes: dict[str, float]) -> str | None:
    """Refuse a live start when the configured models are larger than the cap."""
    report = size_report(settings, sizes)
    if report.startswith("cannot read"):
        return report
    names = configured_models(settings)
    total = sum(sizes[name] for name in names)
    if total <= settings.max_model_gb:
        return None
    largest = max(names, key=lambda name: sizes[name])
    return f"{report}. Downsize {largest}."
