FROM node:22-slim AS frontend-build

WORKDIR /frontend
COPY frontend/package.json frontend/vite.config.js ./
COPY frontend/src src
COPY frontend/index.html ./
RUN npm install && npm run build

FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .
COPY backend backend
COPY src src
COPY data/processed data/processed
COPY data/vector_index data/vector_index
COPY --from=frontend-build /frontend/dist frontend/dist

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]