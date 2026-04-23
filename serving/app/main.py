import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import db, layer1
from app.routes import (
    auth,
    classify,
    custom_categories,
    feedback,
    monitoring,
    suggestion,
    tag_example,
)

app = FastAPI(title="ML Serving — ActualBudget", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_start_time: float = 0.0


def get_uptime() -> float:
    return time.time() - _start_time


@app.on_event("startup")
async def startup() -> None:
    global _start_time
    _start_time = time.time()
    await db.init_pool()
    await db.ensure_tables()
    layer1.load_model()


@app.on_event("shutdown")
async def shutdown() -> None:
    await db.close_pool()


app.include_router(auth.router)
app.include_router(classify.router)
app.include_router(feedback.router)
app.include_router(tag_example.router)
app.include_router(custom_categories.router)
app.include_router(suggestion.router)
app.include_router(monitoring.router)
