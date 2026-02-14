# Base image with Mamba/Conda
FROM mambaorg/micromamba:1.5.8

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MULTIWFN_HOME=/opt/Multiwfn
ENV PATH="${MULTIWFN_HOME}:${PATH}"

USER root

# Install system dependencies (wget, unzip for Multiwfn)
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Conda dependencies
# KNF-Core needs: python, numpy, scipy, openbabel
# xTB is also needed.
RUN micromamba install -y -n base -c conda-forge \
    python=3.11 \
    numpy \
    scipy \
    openbabel \
    xtb \
    && micromamba clean --all --yes

# Activate conda environment settings for interactions
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Install Multiwfn (Linux noGUI version)
# Using 3.8 dev version as utilized in the Windows setup reference
WORKDIR /opt
RUN wget http://sobereva.com/multiwfn/misc/Multiwfn_3.8_dev_bin_Linux_noGUI.zip -O Multiwfn.zip \
    && unzip Multiwfn.zip \
    && mv Multiwfn_3.8_dev_bin_Linux_noGUI Multiwfn \
    && rm Multiwfn.zip \
    && chmod +x /opt/Multiwfn/Multiwfn \
    # Setup settings.ini (copy generic one if needed or let it use default)
    && echo "nthreads=4" > /opt/Multiwfn/settings.ini

# Set working directory for app
WORKDIR /app

# Copy project files
COPY . .

# Install KNF-Core as a package
RUN pip install .

# Environment variable for KNF to know it's in Docker (optional, but good practice)
ENV KNF_IN_DOCKER=1

# Default command
ENTRYPOINT ["knf"]
