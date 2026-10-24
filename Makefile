.PHONY: demo test bench-dev verify-audit

demo:
	uv run python -m meridian.demo

test:
	uv run pytest -q -m "not live"

bench-dev:
	uv run python -m meridian.bench --split dev

verify-audit:
	uv run python -m meridian.audit verify
