# ADR-006: uv + ruff + mypy strict toolchain

- Status: Accepted

## Context

The later phases add a graph, data contracts, Ray, Slack, and model clients. Those pieces need one install path, one formatter, and a type checker that fails on the contracts rather than warning about them. Drift between laptops and CI is a bad way to discover a broken foundation.

## Decision

- Python 3.11 or newer. The repository pins 3.11 in `.python-version` so CI and local runs share a minor version.
- `uv` installs the project. `uv.lock` is part of the repository.
- `ruff` lints and formats. Configuration lives in `pyproject.toml`.
- `mypy --strict` typechecks `src`. The Pydantic plugin is enabled so model fields are visible to the checker.
- `pytest` with `pytest-cov` is the test runner. Coverage on `src/meridian` must stay at or above 95 percent.
- CI on push and pull request runs the same commands a reviewer runs locally: `uv sync`, `ruff check`, `ruff format --check`, `mypy --strict src`, `pytest` with the coverage gate, and the schema export drift check.
- Settings use `pydantic-settings` (`BaseSettings`) with `SecretStr` for provider and Slack keys. `python-dotenv` still loads `.env` and does not override variables that are already set. The original dependency list did not name `pydantic-settings`; it is a direct dependency because that is the supported way to validate settings on load in Pydantic v2. Every key stays optional at import. Only `Settings.require_llm()` and `Settings.require_slack()` raise, and they name the missing variables.

Runtime libraries for later phases (`langgraph`, LangChain providers, `pandera`, `ray`, `slack-bolt`, `mcp`) are declared now so the lockfile is stable. Phase 0 does not import them.

## Consequences

- A change that fails ruff, mypy, the coverage gate, or the schema diff is not done.
- The lockfile will move when dependencies move. That churn is visible in review.
- Strict mode plus frozen models makes some convenient coercions unavailable in Python call sites. Tests and later nodes pass real types.
- `pydantic-settings` is a direct dependency even though the original library list did not name it. It is the settings implementation behind `python-dotenv` and `SecretStr`.

## Alternatives considered

- `pip` and a committed `requirements.txt`. Rejected because `uv.lock` resolves the full tree, including the later-phase libraries, in one command.
- Poetry. Rejected in favor of `uv` as the tool named in the project rules.
- Black plus a separate linter. Rejected because ruff already formats and lints, and one tool is easier to keep in CI.
- Mypy without `--strict`, or no type checker. Rejected because the contracts are the interface, and an untyped interface will rot as nodes are added.
- A hand-rolled settings object that only calls `python-dotenv` and reads `os.environ`. Rejected because `pydantic-settings` already validates `SecretStr`, ignores unrelated environment variables, and treats blank values as unset. `python-dotenv` remains the loader for `.env`.
