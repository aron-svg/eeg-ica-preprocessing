# Simple Dockerfile for Template OrigenesRD

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy and install dependencies
COPY pyproject.toml ./
RUN pip install -e .

# Copy application code
COPY src/ ./src/
COPY data/ ./data/

# Expose port
EXPOSE 8050

# Run the application
CMD ["python", "src/main.py"]