"""Run and day budgets for paid providers. Free providers cost nothing."""

import sqlite3
from datetime import date
from decimal import Decimal
from pathlib import Path

from meridian.llm.prices import ModelPrice, cost_usd, load_prices

PAID = frozenset({"anthropic", "xai"})


class BudgetError(Exception):
    """A call is refused because the price is missing or a cap would be crossed."""


class SpendStore:
    """Day spend in ``data/spend.sqlite``. Amounts are integer micros."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS spend (
                id INTEGER PRIMARY KEY,
                day TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                usd_micros INTEGER NOT NULL
            )
            """
        )
        self._conn.commit()

    def add(
        self,
        *,
        day: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        usd: Decimal,
    ) -> None:
        micros = int((usd * Decimal(1_000_000)).quantize(Decimal("1")))
        self._conn.execute(
            """
            INSERT INTO spend (day, provider, model, input_tokens, output_tokens, usd_micros)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (day, provider, model, input_tokens, output_tokens, micros),
        )
        self._conn.commit()

    def micros(self, *, day: str | None, provider: str | None) -> int:
        clauses: list[str] = []
        params: list[str] = []
        if day is not None:
            clauses.append("day = ?")
            params.append(day)
        if provider is not None:
            clauses.append("provider = ?")
            params.append(provider)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        row = self._conn.execute(
            f"SELECT COALESCE(SUM(usd_micros), 0) FROM spend{where}",
            params,
        ).fetchone()
        if row is None:
            return 0
        return int(row[0])

    def providers(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT provider FROM spend ORDER BY provider"
        ).fetchall()
        return [str(row[0]) for row in rows]

    def close(self) -> None:
        self._conn.close()


class Budget:
    """Pre-call estimate against the remaining run and day caps, then the real cost."""

    def __init__(
        self,
        *,
        prices: dict[str, ModelPrice],
        max_run: Decimal | None,
        max_day: Decimal | None,
        store: SpendStore,
        allow_paid: bool,
    ) -> None:
        self.prices = prices
        self.max_run = max_run
        self.max_day = max_day
        self.store = store
        self.allow_paid = allow_paid
        self.run_spent = Decimal(0)

    def preflight(self, provider: str, model: str, estimated_input_tokens: int) -> None:
        """Block a paid call whose input estimate would cross either cap."""
        if provider not in PAID:
            return
        price = self._price(provider, model)
        estimate = Decimal(max(estimated_input_tokens, 0)) / Decimal(1_000_000)
        estimate *= price.input_per_million
        if self.max_run is None or self.run_spent + estimate > self.max_run:
            raise BudgetError("the call would exceed MERIDIAN_MAX_USD_PER_RUN")
        spent_today = Decimal(self.store.micros(day=_today(), provider=None)) / Decimal(1_000_000)
        if self.max_day is None or spent_today + estimate > self.max_day:
            raise BudgetError("the call would exceed MERIDIAN_MAX_USD_PER_DAY")

    def commit(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> Decimal:
        """Record the provider's real usage. Free providers stay at zero."""
        if provider not in PAID:
            return Decimal(0)
        price = self._price(provider, model)
        spent = cost_usd(price, input_tokens, output_tokens)
        self.run_spent += spent
        self.store.add(
            day=_today(),
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usd=spent,
        )
        return spent

    def _price(self, provider: str, model: str) -> ModelPrice:
        if not self.allow_paid:
            raise BudgetError("paid providers require MERIDIAN_ALLOW_PAID=1")
        if self.max_run is None or self.max_day is None:
            raise BudgetError("a USD budget cap is not configured")
        price = self.prices.get(model)
        if price is None:
            raise BudgetError(
                f"no published price for {provider} model {model}; refusing to run it"
            )
        return price


def open_budget(
    *,
    spend_path: str,
    allow_paid: bool,
    max_run: float | None,
    max_day: float | None,
    prices: dict[str, ModelPrice] | None = None,
) -> Budget:
    """Budget backed by the configured sqlite file and the committed price table."""
    return Budget(
        prices=load_prices() if prices is None else prices,
        max_run=None if max_run is None else Decimal(str(max_run)),
        max_day=None if max_day is None else Decimal(str(max_day)),
        store=SpendStore(Path(spend_path)),
        allow_paid=allow_paid,
    )


def format_report(store: SpendStore) -> str:
    """Today's paid spend and the all-time paid spend, per provider."""
    names = ["anthropic", "xai", *store.providers()]
    seen: list[str] = []
    for name in names:
        if name not in seen:
            seen.append(name)
    lines: list[str] = []
    for provider in seen:
        lines.append(f"today {provider} {_money(store, day=_today(), provider=provider)}")
    for provider in seen:
        lines.append(f"total {provider} {_money(store, day=None, provider=provider)}")
    return "\n".join(lines)


def _money(store: SpendStore, *, day: str | None, provider: str) -> str:
    amount = Decimal(store.micros(day=day, provider=provider)) / Decimal(1_000_000)
    return f"${amount.quantize(Decimal('0.01'))}"


def _today() -> str:
    return date.today().isoformat()
