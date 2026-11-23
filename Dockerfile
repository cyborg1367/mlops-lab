
# Use the official uv image to obtain the uv binary
FROM ghcr.io/astral-sh/uv:0.12.5 AS uv

# Runtime environment
FROM python:3.12-slim

# Copy uv binaries
COPY --from=uv /uv /uvx /bin/

WORKDIR /app

# Python and uv configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# Copy dependency definitions
COPY pyproject.toml uv.lock ./

# Install production dependencies only
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source code
COPY src/ ./src/

# API port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
