FROM python:3.11-slim

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY src /app/src

RUN pip install --upgrade pip setuptools wheel \
    && pip install --index-url ${PYTORCH_INDEX_URL} torch \
    && pip install .[service,mcp]

EXPOSE 8000 8001

CMD ["timesfm-api"]
