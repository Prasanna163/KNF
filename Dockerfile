FROM mambaorg/micromamba:1.5.8

USER root

ARG DEBIAN_FRONTEND=noninteractive
ARG MULTIWFN_URL=http://sobereva.com/multiwfn/misc/Multiwfn_3.8_dev_bin_Linux_noGUI.zip

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KNF_IN_DOCKER=1 \
    MULTIWFN_HOME=/opt/Multiwfn \
    KNF_MULTIWFN_PATH=/opt/Multiwfn/Multiwfn \
    MAMBA_ROOT_PREFIX=/opt/conda \
    XTBHOME=/opt/conda \
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
    openbabel \
    xtb \
    && micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1

WORKDIR /opt
RUN wget "${MULTIWFN_URL}" -O Multiwfn.zip \
    && unzip Multiwfn.zip \
    && mv Multiwfn_3.8_dev_bin_Linux_noGUI Multiwfn \
    && rm Multiwfn.zip \
    && chmod +x /opt/Multiwfn/Multiwfn \
    && printf "nthreads=4\nisilent=1\n" > /opt/Multiwfn/settings.ini

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir ".[torch-nci]"

RUN sed -i 's/\r$//' /app/scripts/docker-entrypoint.sh \
    && chmod +x /app/scripts/docker-entrypoint.sh \
    && chown -R mambauser:mambauser /app

USER mambauser

HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
  CMD bash -lc "command -v knf && command -v xtb && command -v obabel && command -v Multiwfn"

ENTRYPOINT ["/usr/bin/tini", "--", "/app/scripts/docker-entrypoint.sh"]
CMD ["--help"]
