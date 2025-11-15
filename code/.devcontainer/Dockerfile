# Use an official Python image as base
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /workspace

# Install Git (and optionally some other useful tools)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Run requirements for LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1

# Install requirements if present
COPY .devcontainer/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt || true

# Install any extra tools you might want
# RUN apt-get update && apt-get install -y vim

# Default command when the container starts
CMD [ "bash" ]
