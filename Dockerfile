# syntax=docker/dockerfile:1.7
# Job & Internship Intelligence Assistant - one image, two services (API and UI), CPU inference.
#   docker compose up --build            (see docker-compose.yml)

FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models JOBINTEL_DATA_DIR=/app/data JOBINTEL_DEVICE=cpu HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /app

FROM base AS builder
RUN python -m venv /venv
ENV PATH=/venv/bin:$PATH
# CPU-only PyTorch keeps the image ~2 GB smaller than the default CUDA wheels
RUN pip install --upgrade pip && pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[app]"

FROM base AS runtime
RUN useradd --create-home --uid 1000 app && mkdir -p /models /app/data && chown -R app /models /app
COPY --from=builder /venv /venv
ENV PATH=/venv/bin:$PATH
COPY --chown=app config ./config
COPY --chown=app eval ./eval
COPY --chown=app reports ./reports
COPY --chown=app .streamlit ./.streamlit
COPY --chown=app src ./src
USER app
EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/health'); sys.exit(0)" || exit 1
CMD ["jobintel", "api", "--host", "0.0.0.0", "--port", "8000"]
