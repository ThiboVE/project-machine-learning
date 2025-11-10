# Use an official Python image as base
FROM python:3.12-slim

WORKDIR /workspace

# Install system dependencies for RDKit
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libboost-all-dev \
    libeigen3-dev \
    libsqlite3-dev \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Install RDKit via pip
RUN pip install --no-cache-dir rdkit

# Install Git and other tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Install requirements if present
COPY .devcontainer/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt || true

CMD ["bash"]
