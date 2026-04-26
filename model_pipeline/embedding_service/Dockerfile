FROM python:3.11-slim

COPY model_pipeline/ /app/model_pipeline/
COPY model_pipeline/embedding_service/requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"

EXPOSE 8001

CMD ["uvicorn", "model_pipeline.embedding_service.app:app", "--host", "0.0.0.0", "--port", "8001"]
