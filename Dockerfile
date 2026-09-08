FROM python:3.11-slim

# uv handles dependency resolution and the venv; ships as a static binary.
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /bin/

WORKDIR /app

# Copy dependency metadata first so dependency install is cached separately
# from source changes.
COPY pyproject.toml README.md ./
COPY src ./src

RUN uv sync --no-dev

COPY app.py ./

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

EXPOSE 8501

# Overridden per-service in docker-compose.yml (ingest vs. streamlit).
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.address=0.0.0.0"]
