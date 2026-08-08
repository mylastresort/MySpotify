FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps (install from lock file)
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen --no-dev

# Copy source
COPY . .

ENTRYPOINT ["python", "entrypoint/train.py"]
