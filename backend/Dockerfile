FROM python:3.12-slim

WORKDIR /app

COPY backend/pyproject.toml backend/
COPY backend/app backend/app
COPY ml ml

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e ./backend

ENV PYTHONPATH=/app
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
