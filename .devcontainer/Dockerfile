FROM mcr.microsoft.com/vscode/devcontainers/python:3.11

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsqlite3-dev \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspaces/Summarization

