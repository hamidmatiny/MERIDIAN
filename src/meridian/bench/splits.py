"""Frozen bench splits. The test seeds never overlap the dev seeds."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from meridian.contracts.fault_type import FaultType

_FAULTS = frozenset(item.value for item in FaultType)


@dataclass(frozen=True)
class RunSpec:
    """One graph execution."""

    split: str
    dataset: str
    seed: int
    fault: str | None
    intensity: float | None

    @property
    def key(self) -> str:
        if self.fault is None:
            return f"{self.dataset}__clean__s{self.seed}"
        return f"{self.dataset}__{self.fault}__i{self.intensity}__s{self.seed}"


def splits_path() -> Path:
    """``bench/splits.yaml`` at the repository root."""
    return Path(__file__).resolve().parents[3] / "bench" / "splits.yaml"


def load_splits(path: Path | None = None) -> dict[str, list[RunSpec]]:
    """Expand the yaml. Raise when dev and test seeds overlap or a fault is unknown."""
    source = splits_path() if path is None else path
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("splits.yaml must be a mapping")
    dev = _injected("dev", loaded.get("dev"))
    test = _injected("test", loaded.get("test"))
    hard = _hard(loaded.get("hard"))
    overlap = {item.seed for item in dev} & {item.seed for item in test}
    if overlap:
        names = ", ".join(str(seed) for seed in sorted(overlap))
        raise ValueError(f"dev and test seeds overlap: {names}")
    return {"dev": dev, "test": test, "hard": hard}


def _injected(split: str, raw: object) -> list[RunSpec]:
    block = _block(split, raw)
    faults = _strings(block, "faults")
    unknown = [name for name in faults if name not in _FAULTS]
    if unknown:
        raise ValueError(f"unknown fault in {split}: {', '.join(unknown)}")
    datasets = _strings(block, "datasets")
    seeds = _ints(block, "seeds")
    intensities = _floats(block, "intensities")
    specs: list[RunSpec] = []
    for fault in faults:
        for dataset in datasets:
            for intensity in intensities:
                for seed in seeds:
                    specs.append(
                        RunSpec(
                            split=split,
                            dataset=dataset,
                            seed=seed,
                            fault=fault,
                            intensity=intensity,
                        )
                    )
    return specs


def _hard(raw: object) -> list[RunSpec]:
    block = _block("hard", raw)
    specs: list[RunSpec] = []
    for dataset in _strings(block, "datasets"):
        for seed in _ints(block, "seeds"):
            specs.append(
                RunSpec(split="hard", dataset=dataset, seed=seed, fault=None, intensity=None)
            )
    return specs


def _block(name: str, raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError(f"{name} split must be a mapping")
    return raw


def _strings(block: dict[str, object], key: str) -> list[str]:
    raw = block.get(key)
    if not isinstance(raw, list) or not raw or not all(isinstance(item, str) for item in raw):
        raise ValueError(f"{key} must be a non-empty list of strings")
    return [str(item) for item in raw]


def _ints(block: dict[str, object], key: str) -> list[int]:
    raw = block.get(key)
    if not isinstance(raw, list) or not raw or not all(isinstance(item, int) for item in raw):
        raise ValueError(f"{key} must be a non-empty list of ints")
    return [int(item) for item in raw]


def _floats(block: dict[str, object], key: str) -> list[float]:
    raw = block.get(key)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{key} must be a non-empty list of numbers")
    numbers: list[float] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise ValueError(f"{key} must be a non-empty list of numbers")
        numbers.append(float(item))
    return numbers
