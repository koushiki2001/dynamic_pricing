ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE}

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
COPY server/requirements.txt server_requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r server_requirements.txt

# Copy project code
COPY . .

# HF Spaces expects port 7860
EXPOSE 7860

# Serve the OpenEnv FastAPI environment server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
