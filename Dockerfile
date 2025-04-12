ARG PYTHON_VERSION=3.11.11
ARG DISTRO=slim
FROM python:${PYTHON_VERSION}-${DISTRO} AS builder

ARG USE_UV=true

# Optional Cache buster
ENV UPDATED_AT=04-08-2025:00:00:02

# Set working directory
WORKDIR /grazer

# Install OS dependencies
# NOTE: Might need to keep libre2-dev
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

# Install UV
# TODO: Pin the uv image SHA
COPY --from=ghcr.io/astral-sh/uv:0.6.3 /uv /uvx /bin/

# Install PDM
ENV PDM_VERSION=2.23.1
RUN pip install -U pdm==${PDM_VERSION}

# Disable pdm update check
ENV PDM_CHECK_UPDATE=false

# Use `uv` for build speed
RUN pdm config use_uv ${USE_UV}

# Options:
# 1. pdm.lock
# 2. py311-linux.lock
ARG LOCKFILE=py311-linux.lock

# Copy dependency specification files first to leverage caching
COPY pyproject.toml ${LOCKFILE} ./

# manually install for pyre2
RUN pip install pybind11[global]
RUN pip install cython>=3.0.12 ninja>=1.11.1.3 setuptools>=75.8.2

RUN pdm sync \
    -G cluster \
    -L ${LOCKFILE} \
    --no-isolation \
    --no-editable \
    --no-self \
    --fail-fast

FROM python:${PYTHON_VERSION}-${DISTRO} AS grazer

WORKDIR /grazer

# Reinstall PDM
# TODO: can we copy this to I wonder
ENV PDM_VERSION=2.23.1
RUN pip install -U pdm==${PDM_VERSION}

COPY --from=builder /grazer/.venv .
COPY --from=builder /grazer/pyproject.toml .
ENV PATH="/grazer/.venv/bin:$PATH"

# Now copy the rest of your application code
COPY app/ app
COPY startup_ray.sh .
COPY run_ray_cache.py .
COPY run_ray_cpu_worker.py .
COPY run_ray_gpu_worker.py .
COPY run_ray_network_worker.py .
COPY run_runpod_worker.py .
COPY test.sh .

# Please see list of scripts in pyproject.toml
# CMD ["pdm", "start_all_workers"]
