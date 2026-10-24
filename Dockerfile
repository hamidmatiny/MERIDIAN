FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /uvx /bin/
WORKDIR /app
COPY . .
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH" \
    VIRTUAL_ENV=/app/.venv
CMD ["sh", "-c", "alembic upgrade head && python -m meridian.demo"]
