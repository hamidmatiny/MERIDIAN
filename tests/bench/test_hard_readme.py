"""Hard-mode FP in the README comes from the clean hard split."""

import json
from pathlib import Path

from meridian.bench.report import render_limitations, render_readme_results

_ROOT = Path("bench/results")


def test_readme_hard_fp_comes_from_hard_clean() -> None:
    test = json.loads((_ROOT / "test-fix-v2-clean" / "summary.json").read_text(encoding="utf-8"))
    hard = json.loads((_ROOT / "hard-clean" / "summary.json").read_text(encoding="utf-8"))
    text = render_readme_results(test, hard=hard)
    assert "| hard-mode FP rate | 0/30 |" in text
    assert "| machine time p50 | 288 s |" in text
    assert "| machine time p95 | 499 s |" in text
    assert f"Recorded at commit `{test['meta']['commit']}`." in text
    assert "critic `gemma4:12b`" in text
    assert hard["meta"]["commit"] in text
    limits = render_limitations(test)
    assert test["evidence"]["exactly_one"]["text"] in limits
    assert f"{test['outcomes']['categories']['FIX_ATTEMPT_LIMIT']}/{test['runs']}" in limits
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "| hard-mode FP rate | 0/30 |" in readme
    assert "| machine time p50 | 288 s |" in readme
    assert "Results committed in `7e7f9f5`." in readme
    assert test["meta"]["commit"] in readme
    assert "critic `gemma4:12b`" in readme
    assert hard["meta"]["commit"] in readme
    assert "Safety comes from the deterministic verifier: 16 " in readme
