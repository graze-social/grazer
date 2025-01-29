# Use an official Python image as the base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements1.txt .
COPY requirements2.txt .
COPY requirements3.txt .
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
    && apt-get clean

# Install dependencies
RUN pip install -r requirements1.txt
RUN pip install -r requirements2.txt
RUN apt-get install -y libre2-dev cmake ninja-build
RUN pip install -r requirements3.txt
# Copy the application code
COPY . .
