FROM python:3.11-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml .
COPY uv.lock .

RUN uv sync --frozen --no-dev

COPY app/ ./app/
COPY frontend/ ./frontend/

EXPOSE 8000

CMD ["uv","run","uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]