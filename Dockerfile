# syntax=docker/dockerfile:1
FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend

ENV FLASK_APP=backend.app:create_app

EXPOSE 8000

CMD ["gunicorn", "--worker-class", "gthread", "--threads", "4", "--bind", "0.0.0.0:8000", "backend.wsgi:app"]
