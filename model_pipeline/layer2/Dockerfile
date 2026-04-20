FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY model_pipeline/layer2/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model_pipeline/ model_pipeline/
COPY training/ training/
RUN rm -f model_pipeline/layer2/config.yaml

# Mount config.yaml at runtime:
#   -v $(pwd)/model_pipeline/layer2/config.yaml:/app/model_pipeline/layer2/config.yaml
CMD ["python", "-m", "model_pipeline.evaluate"]
