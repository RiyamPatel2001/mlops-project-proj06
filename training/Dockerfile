FROM python:3.11-slim

WORKDIR /app

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
#     -e MLFLOW_TRACKING_URI=http://<chameleon-ip>:5000 \
#     transaction-categorizer-training

ENTRYPOINT ["python", "train.py", "--config", "config.yaml"]
