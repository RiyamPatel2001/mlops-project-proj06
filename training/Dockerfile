FROM python:3.11-slim

WORKDIR /app/training

# Install git first so MLflow can capture git SHA
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install dependencies first — this layer is cached as long as
# requirements.txt doesn't change, so rebuilds after code edits are fast.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all training code into /app/training (build context = training/)
COPY . .

# Build from the training/ directory:
#   docker build -t transaction-categorizer-training .
#
# Runtime overrides (all optional — defaults live in config.yaml):
#   MLFLOW_TRACKING_URI  — point at the Chameleon MLflow VM
#
# Example run (from training/ directory):
#   docker run --rm \
#     -v $(pwd)/../data:/app/data \
#     -v $(pwd)/models/layer1/artifacts:/app/training/models/layer1/artifacts \
#     -e MLFLOW_TRACKING_URI=http://<chameleon-ip>:8000 \
#     transaction-categorizer-training
#
# Windows (PowerShell):
#   docker run --rm \
#     -v ${PWD}\..\data:/app/data \
#     -v ${PWD}\models\layer1\artifacts:/app/training/models/layer1/artifacts \
#     -e MLFLOW_TRACKING_URI=http://<chameleon-ip>:8000 \
#     transaction-categorizer-training

ENTRYPOINT ["python", "train.py", "--config", "config.yaml"]
