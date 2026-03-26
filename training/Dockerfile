FROM python:3.11-slim

WORKDIR /app

# Install git first so MLflow can capture git SHA
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install dependencies first — this layer is cached as long as
# requirements.txt doesn't change, so rebuilds after code edits are fast.
COPY training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all training code from context root into /app
COPY . .

# Runtime overrides (all optional — defaults live in config.yaml):
#   MLFLOW_TRACKING_URI  — point at the Chameleon MLflow VM
#
# Example run:
#   docker run --rm \
#     -v %cd%\data:/app/data \
#     -v %cd%\models\artifacts:/app/models/artifacts \
#     -e MLFLOW_TRACKING_URI=http://<chameleon-ip>:8000 \
#     transaction-categorizer-training

ENTRYPOINT ["python", "training/train.py", "--config", "training/config.yaml"]
