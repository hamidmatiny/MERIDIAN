"""Price refusals and the sqlite day ledger."""

from decimal import Decimal
from pathlib import Path

import pytest

from meridian.llm.budget import Budget, BudgetError, SpendStore, format_report
from meridian.llm.prices import ModelPrice, cost_usd


def test_missing_price_refuses_the_model(tmp_path: Path) -> None:
    budget = Budget(
        prices={},
        max_run=Decimal("1"),
        max_day=Decimal("1"),
        store=SpendStore(tmp_path / "spend.sqlite"),
        allow_paid=True,
    )
    with pytest.raises(BudgetError, match="no published price"):
        budget.preflight("anthropic", "claude-haiku", 10)


def test_estimate_blocks_a_call_over_the_run_cap(tmp_path: Path) -> None:
    price = ModelPrice(input_per_million=Decimal("3"), output_per_million=Decimal("15"))
    budget = Budget(
        prices={"claude-haiku": price},
        max_run=Decimal("0.25"),
        max_day=Decimal("1"),
        store=SpendStore(tmp_path / "spend.sqlite"),
        allow_paid=True,
    )
    with pytest.raises(BudgetError, match="MERIDIAN_MAX_USD_PER_RUN"):
        budget.preflight("anthropic", "claude-haiku", 1_000_000)


def test_real_usage_is_stored_for_the_day(tmp_path: Path) -> None:
    price = ModelPrice(input_per_million=Decimal("1"), output_per_million=Decimal("2"))
    store = SpendStore(tmp_path / "spend.sqlite")
    budget = Budget(
        prices={"claude-haiku": price},
        max_run=Decimal("1"),
        max_day=Decimal("1"),
        store=store,
        allow_paid=True,
    )
    budget.preflight("anthropic", "claude-haiku", 1000)
    spent = budget.commit("anthropic", "claude-haiku", 1000, 500)
    assert spent == cost_usd(price, 1000, 500)
    text = format_report(store)
    assert "today anthropic $0.00" in text or "today anthropic $0.01" in text
    assert "total anthropic" in text
    assert "today xai $0.00" in text
    store.close()


def test_free_commit_does_not_write_spend(tmp_path: Path) -> None:
    store = SpendStore(tmp_path / "spend.sqlite")
    budget = Budget(
        prices={},
        max_run=Decimal("1"),
        max_day=Decimal("1"),
        store=store,
        allow_paid=False,
    )
    assert budget.commit("ollama", "qwen3.5:4b", 100, 20) == Decimal(0)
    assert "today anthropic $0.00" in format_report(store)
    assert "total xai $0.00" in format_report(store)
    store.close()


def test_empty_ledger_prints_zero(tmp_path: Path) -> None:
    store = SpendStore(tmp_path / "spend.sqlite")
    text = format_report(store)
    assert text.count("$0.00") >= 4
    store.close()


def test_day_cap_and_a_disabled_paid_flag_block(tmp_path: Path) -> None:
    price = ModelPrice(input_per_million=Decimal("3"), output_per_million=Decimal("15"))
    store = SpendStore(tmp_path / "spend.sqlite")
    over_day = Budget(
        prices={"claude-haiku": price},
        max_run=Decimal("10"),
        max_day=Decimal("0"),
        store=store,
        allow_paid=True,
    )
    with pytest.raises(BudgetError, match="MERIDIAN_MAX_USD_PER_DAY"):
        over_day.preflight("anthropic", "claude-haiku", 1_000)
    closed = Budget(
        prices={"claude-haiku": price},
        max_run=Decimal("1"),
        max_day=Decimal("1"),
        store=store,
        allow_paid=False,
    )
    with pytest.raises(BudgetError, match="MERIDIAN_ALLOW_PAID"):
        closed.preflight("anthropic", "claude-haiku", 1)
    uncapped = Budget(
        prices={"claude-haiku": price},
        max_run=None,
        max_day=None,
        store=store,
        allow_paid=True,
    )
    with pytest.raises(BudgetError, match="not configured"):
        uncapped.preflight("anthropic", "claude-haiku", 1)
    store.close()
