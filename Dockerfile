FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DATABASE_URL=sqlite:////app/Data/medical_agent.db

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY backend backend
COPY frontend frontend
COPY scripts scripts
COPY README.md MODEL_CARD.md DATA_CARD.md ./

RUN mkdir -p Data KnowledgeBase/raw KnowledgeBase/processed

EXPOSE 8017

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8017/health', timeout=3).read()"

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8017"]
