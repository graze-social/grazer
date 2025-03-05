ARG PYTHON_VERSION=3.11
ARG DISTRO=slim
ARG USE_UV=true

FROM python:${PYTHON_VERSION}-${DISTRO}

# Optional Cache buster
ENV UPDATED_AT=03-03-2025

# Set working directory
WORKDIR /grazer

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cmake \
    ninja-build \
    git \
    pkg-config \
    libre2-dev \
    tmux \
    nano \
    rsync \
    unzip \
    zip \
    htop \
    curl \
    && apt-get clean

# Install dependencies
RUN apt-get install -y libre2-dev cmake ninja-build

# Install UV
# TODO: Pin the uv image SHA
COPY --from=ghcr.io/astral-sh/uv:0.6.3 /uv /uvx /bin/

# Copy src directory
COPY pyproject.toml .
COPY pdm.lock .
COPY main.py .
COPY app/ .



# Install PDM
ENV PDM_VERSION=2.22.3
RUN pip install -U pdm==${PDM_VERSION}

# Disable pdm update check
ENV PDM_CHECK_UPDATE=false

# Use `uv` for build speed
RUN pdm config use_uv ${USE_UV}
RUN pdm sync --prod \
    --no-self \
    --no-editable \
    --fail-fast

# Please see list of scripts in pyproject.toml
# CMD ["pdm", "start_all_workers"]
