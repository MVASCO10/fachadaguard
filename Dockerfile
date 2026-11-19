# Imagem única (CPU-only) para API, ingestor, worker, dashboard e simulador.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 fg
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY fachadaguard ./fachadaguard
COPY apps ./apps
COPY services ./services
COPY simulator ./simulator
COPY scripts ./scripts
RUN pip install ".[dashboard]"
RUN mkdir -p /app/data /app/artifacts && chown -R fg:fg /app
USER fg
ENV FG_DATA_DIR=/app/data FG_ARTIFACTS_DIR=/app/artifacts
EXPOSE 8000 8501
CMD ["uvicorn", "apps.api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
