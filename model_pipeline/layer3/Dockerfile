FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY model_pipeline/layer2/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir sentence-transformers==2.7.*

COPY model_pipeline/ model_pipeline/
COPY training/ training/

CMD ["python", "-m", "model_pipeline.layer3.pipeline"]
