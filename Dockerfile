FROM python:3.13-slim-trixie@sha256:9662417aace5ae7b8e2609cce472b72a8958e134ba372808abe9cc1a0c0125e6 AS builder

ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
ARG REQUIREMENTS_FILE=requirements.txt

RUN apt-get update \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

COPY requirements*.txt ./
RUN python -m pip install --upgrade \
        pip==26.2.1 \
        "setuptools>=78.1.1" \
        "wheel>=0.46.2" \
    && python -m pip install --prefix=/runtime -r "${REQUIREMENTS_FILE}"


FROM gcr.io/distroless/python3-debian13:nonroot@sha256:1c680cdb442a9e7a89f64fd1706367c62302ea1f9ab80fdebdb72ae9fcded46f

ARG SOURCE_REVISION=uncommitted
ARG BUILD_CREATED=unknown
LABEL org.opencontainers.image.title="NLCare synthetic-staging backend" \
      org.opencontainers.image.description="Restricted synthetic-only engineering runtime; not for clinical use" \
      org.opencontainers.image.source="https://github.com/KiyotakaShinichi/MedicalAgent" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.created="${BUILD_CREATED}" \
      ai.nlcare.deployment-scope="restricted-synthetic-staging-only" \
      ai.nlcare.clinical-validation="false"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/usr/local/lib/python3.13/site-packages
ENV DATABASE_URL=sqlite:////app/Data/medical_agent.db

WORKDIR /app

COPY --from=builder /runtime /usr/local
COPY --chown=nonroot:nonroot backend backend
COPY --chown=nonroot:nonroot frontend frontend
COPY --chown=nonroot:nonroot scripts scripts
COPY --chown=nonroot:nonroot config config
COPY --chown=nonroot:nonroot evals evals
COPY --chown=nonroot:nonroot alembic.ini alembic.ini
COPY --chown=nonroot:nonroot README.md MODEL_CARD.md DATA_CARD.md ./

USER nonroot:nonroot

EXPOSE 8017

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["/usr/bin/python3", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8017/health', timeout=3).read()"]

ENTRYPOINT ["/usr/bin/python3"]
CMD ["scripts/container_entrypoint.py"]
