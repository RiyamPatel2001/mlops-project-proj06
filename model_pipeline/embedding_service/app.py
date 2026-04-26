import sys
sys.path.insert(0, "/app")

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from model_pipeline.layer2.embedder import Embedder

_embedder: Embedder | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embedder
    _embedder = Embedder(model_name="sentence-transformers/all-mpnet-base-v2", max_length=128)
    yield


app = FastAPI(lifespan=lifespan)


class EmbedRequest(BaseModel):
    text: str


class EmbedResponse(BaseModel):
    embedding: list[float]


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest) -> EmbedResponse:
    return EmbedResponse(embedding=_embedder.embed(req.text).tolist())


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
