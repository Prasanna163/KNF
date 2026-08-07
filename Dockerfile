FROM mambaorg/micromamba:1.5.8

USER root

ARG DEBIAN_FRONTEND=noninteractive
ARG MULTIWFN_URL=https://www.umsyar.com/multiwfn/misc/Multiwfn_3.8_dev_bin_Linux_noGUI.zip
ARG NCIFORGE_VERSION=1.0.9

LABEL org.opencontainers.image.title="NCIForge" \
      org.opencontainers.image.version="${NCIFORGE_VERSION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NCIFORGE_IN_DOCKER=1 \
    KUID_IN_DOCKER=1 \
    KNF_IN_DOCKER=1 \
    NCIFORGE_API_WORKERS=1 \
    NCIFORGE_DEFAULT_XTB_ENGINE=xtb \
    MULTIWFN_HOME=/opt/Multiwfn \
    NCIFORGE_MULTIWFN_PATH=/opt/Multiwfn/Multiwfn \
    KUID_MULTIWFN_PATH=/opt/Multiwfn/Multiwfn \
    KNF_MULTIWFN_PATH=/opt/Multiwfn/Multiwfn \
    MAMBA_ROOT_PREFIX=/opt/conda \
    XTBHOME=/opt/conda \
    MPLBACKEND=Agg \
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    PATH="/opt/conda/bin:/opt/conda/condabin:/opt/Multiwfn:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    unzip \
    tini \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN micromamba install -y -n base -c conda-forge \
    python=3.11 \
    numpy \
    scipy \
    matplotlib \
    rdkit \
    openbabel \
    xtb \
    && micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1

WORKDIR /opt
RUN wget "${MULTIWFN_URL}" -O Multiwfn.zip \
    && unzip Multiwfn.zip \
    && mv Multiwfn_3.8_dev_bin_Linux_noGUI Multiwfn \
    && rm Multiwfn.zip \
    && mv /opt/Multiwfn/Multiwfn_noGUI /opt/Multiwfn/Multiwfn \
    && chmod +x /opt/Multiwfn/Multiwfn \
    && printf "nthreads=4\nisilent=1\n" > /opt/Multiwfn/settings.ini

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir ".[api,plots]" \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

RUN sed -i 's/\r$//' /app/scripts/docker-entrypoint.sh \
    && chmod +x /app/scripts/docker-entrypoint.sh \
    && chown -R mambauser:mambauser /app

ENV LD_LIBRARY_PATH=/opt/conda/lib

USER mambauser

EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
  CMD bash -c "command -v nciforge && command -v knf && command -v geoinit && command -v nciforge-api && command -v xtb && command -v obabel && command -v Multiwfn && python -c 'import torch, matplotlib, fastapi, uvicorn'"

ENTRYPOINT ["/usr/bin/tini", "--", "/app/scripts/docker-entrypoint.sh"]
CMD ["--help"]

