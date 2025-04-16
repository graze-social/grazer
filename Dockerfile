ARG PYTHON_VERSION=3.11
ARG DISTRO=slim
ARG USE_UV=true

FROM python:${PYTHON_VERSION}-${DISTRO}

# Optional Cache buster
ENV UPDATED_AT=04-08-2025:00:00:01

# Set working directory
WORKDIR /grazer

# Install OS dependencies
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
ENV PDM_VERSION=2.22.3
RUN pip install -U pdm==${PDM_VERSION}

# Disable pdm update check
ENV PDM_CHECK_UPDATE=false

# Use `uv` for build speed
RUN pdm config use_uv ${USE_UV}

# Copy dependency specification files first to leverage caching
COPY pyproject.toml pdm.lock ./

# Install Python dependencies
RUN pdm sync \
    --no-isolation \
    --no-editable \
    --no-self \
    --fail-fast

# Now copy the rest of your application code
COPY app/ app
COPY tests/ tests
COPY startup_ray.sh .
COPY run_ray_cache.py .
COPY run_ray_cpu_worker.py .
COPY run_ray_gpu_worker.py .
COPY run_ray_network_worker.py .
COPY run_runpod_worker.py .
COPY run_streamer.py .
COPY test.sh .

# Please see list of scripts in pyproject.toml
# CMD ["pdm", "start_all_workers"]