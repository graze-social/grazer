###############################
### STAGE 01 - Dependencies ###
###############################

ARG PYTHON_VERSION=3.11
ARG DISTRO=slim

FROM python:${PYTHON_VERSION}-${DISTRO} as builder

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
COPY . .

# Install PDM
ENV PDM_VERSION=2.22.3
RUN pip install -U pdm==${PDM_VERSION}

# Disable pdm update check
ENV PDM_CHECK_UPDATE=false

# Using `uv` for speed of build
RUN pdm config use_uv true
RUN pdm sync --prod \
    --no-self \
    --no-editable \
    --fail-fast

###############################
### STAGE 02 - App ###
###############################
FROM python:${PYTHON_VERSION}-${DISTRO} as app

COPY --from=builder /grazer/.venv /grazer/.venv
COPY --from=builder /grazer/main.py .
COPY --from=builder /grazer/app /grazer/app
COPY --from=builder /grazer/scripts /grazer/scripts
COPY --from=builder /grazer/pyproject.toml /grazer/pyproject.toml
COPY --from=builder /grazer/pdm.lock /grazer/pdm.lock


ENV PATH="/grazer/.venv/bin:$PATH"


# Please see list of scripts in pyproject.toml
# CMD ["pdm", "start_all_workers"]
