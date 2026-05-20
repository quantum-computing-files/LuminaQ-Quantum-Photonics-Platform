FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*
COPY packages/lumq-photonics  /app/packages/lumq-photonics
COPY packages/lumq-compiler   /app/packages/lumq-compiler
COPY packages/lumq-backends   /app/packages/lumq-backends
COPY packages/lumq-api        /app/packages/lumq-api
RUN pip install --upgrade pip \
    && pip install "jax[cpu]" numpy scipy matplotlib pydantic \
    && pip install fastapi "uvicorn[standard]" websockets pydantic-settings python-multipart anyio \
    && pip install -e /app/packages/lumq-photonics \
    && pip install -e /app/packages/lumq-compiler \
    && pip install -e /app/packages/lumq-backends \
    && pip install -e /app/packages/lumq-api
EXPOSE 8000
CMD ["uvicorn","lumq.api.app:app","--host","0.0.0.0","--port","8000","--workers","2"]
