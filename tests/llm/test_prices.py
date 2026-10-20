"""The price table refuses a missing or malformed file instead of guessing."""

from decimal import Decimal
from pathlib import Path

import pytest

from meridian.llm.prices import ModelPrice, cost_usd, load_prices


def test_cost_uses_both_directions() -> None:
    price = ModelPrice(input_per_million=Decimal("1"), output_per_million=Decimal("2"))
    assert cost_usd(price, 1_000_000, 500_000) == Decimal("2")


def test_empty_and_malformed_tables(tmp_path: Path) -> None:
    blank = tmp_path / "blank.yaml"
    blank.write_text("", encoding="utf-8")
    assert load_prices(blank) == {}
    listed = tmp_path / "list.yaml"
    listed.write_text("- 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_prices(listed)
    bad_models = tmp_path / "models.yaml"
    bad_models.write_text("models: [1]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="models"):
        load_prices(bad_models)
    bad_row = tmp_path / "row.yaml"
    bad_row.write_text("models:\n  claude: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="price for"):
        load_prices(bad_row)
