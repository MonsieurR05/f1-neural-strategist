# Strategy API container. Build from the repository root (the data and model files are
# local build artifacts, not in git):
#
#   python -m strategist.data.clean && python -m strategist.data.tracks --circuit Sakhir
#   python -m strategist.sim.backtest          # trains lap_net, calibrates the simulator
#   docker build -f services/api/Dockerfile -t f1-strategist-api .
#   docker run -p 8080:8080 -e F1S_CORS_ORIGINS=http://localhost:3000 f1-strategist-api

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    F1S_API_CONFIG=configs/api.deploy.yaml \
    F1S_THREADS=2 \
    PORT=8080

WORKDIR /app

RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
COPY services/api/requirements.txt services/api/requirements.txt
RUN pip install -r services/api/requirements.txt

# The strategist package without its data-pipeline and training extras (FastF1, LightGBM, ...)
COPY packages/strategist packages/strategist
RUN pip install --no-deps ./packages/strategist

COPY services/api/app services/api/app
COPY configs configs
COPY data/processed data/processed
COPY data/raw data/raw
COPY artifacts/models artifacts/models

RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8080
CMD ["sh", "-c", "uvicorn app.main:app --app-dir services/api --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
