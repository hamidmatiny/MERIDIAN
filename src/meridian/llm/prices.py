"""Published USD prices for paid models. Missing rows are refused, never guessed."""

from decimal import Decimal
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

_PATH = Path(__file__).with_name("prices.yaml")


class ModelPrice(BaseModel):
    """USD per one million tokens. Both directions are required."""

    model_config = ConfigDict(extra="forbid")

    input_per_million: Decimal = Field(ge=0)
    output_per_million: Decimal = Field(ge=0)


def load_prices(path: Path | None = None) -> dict[str, ModelPrice]:
    """Read the price table. An empty table means every paid model is refused."""
    source = _PATH if path is None else path
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("price table must be a mapping")
    models = loaded.get("models", {})
    if not isinstance(models, dict):
        raise ValueError("price table models must be a mapping")
    prices: dict[str, ModelPrice] = {}
    for name, raw in models.items():
        if not isinstance(name, str) or not isinstance(raw, dict):
            raise ValueError(f"price for {name} must be a mapping")
        prices[name] = ModelPrice.model_validate(raw)
    return prices


def cost_usd(price: ModelPrice, input_tokens: int, output_tokens: int) -> Decimal:
    """USD for one call. Tokens are the provider's counts, not an estimate."""
    million = Decimal(1_000_000)
    incoming = Decimal(input_tokens) / million * price.input_per_million
    outgoing = Decimal(output_tokens) / million * price.output_per_million
    return incoming + outgoing
