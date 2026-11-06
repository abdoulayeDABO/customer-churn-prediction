FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --no-install-project --all-extras && uv sync --no-dev --all-extras

COPY src ./src
COPY scripts ./scripts
COPY config ./config

RUN uv sync --no-dev --all-extras

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 7860

# Default service = scoring app (Gradio). The model bundle must be provided
# by an image or by a volume: docker run -v ./models:/app/models ...
CMD ["churn-gradio", "--host", "0.0.0.0", "--port", "7860"]
