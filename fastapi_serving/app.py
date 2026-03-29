"""
FastAPI serving application for transaction classification.

Supports multiple model backends: distilbert, minilm (via PyTorch or ONNX),
tfidf_logreg (sklearn or ONNX), and fasttext (native).

Configure via environment variables:
    MODEL_TYPE: distilbert | minilm | tfidf_logreg | fasttext
    MODEL_BACKEND: pytorch | onnx | sklearn | native
    MODEL_PATH: path to model directory or file
"""

import os
import json
import logging

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_TYPE = os.getenv("MODEL_TYPE", "distilbert")
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "onnx")
MODEL_PATH = os.getenv("MODEL_PATH", f"models/{MODEL_TYPE}")
ONNX_PATH = os.getenv("ONNX_PATH", f"models/{MODEL_TYPE}_onnx/model.onnx")
LABEL_MAP_PATH = os.getenv("LABEL_MAP_PATH", f"models/{MODEL_TYPE}/label_map.json")

app = FastAPI(
    title="Transaction Classification API",
    description="Classify financial transactions into spending categories",
    version="1.0.0",
)


class TransactionRequest(BaseModel):
    payee: str
    amount: float
    day_of_week: str


class PredictionResponse(BaseModel):
    category: str


label_map = None
model = None
tokenizer = None
ort_session = None
ft_model = None
sklearn_pipeline = None


def load_label_map():
    global label_map
    if os.path.exists(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH) as f:
            label_map = json.load(f)
        logger.info(f"Loaded label map with {len(label_map)} classes")
    else:
        label_map = None
        logger.warning(f"No label map found at {LABEL_MAP_PATH}; using raw model output indices")


def get_label(idx):
    if label_map is not None:
        return label_map.get(str(idx), label_map.get(idx, f"class_{idx}"))
    return f"class_{idx}"


def load_transformer_pytorch():
    global model, tokenizer
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    logger.info(f"Loaded {MODEL_TYPE} PyTorch model from {MODEL_PATH}")


def load_transformer_onnx():
    global ort_session, tokenizer
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    ort_session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    logger.info(f"Loaded {MODEL_TYPE} ONNX model from {ONNX_PATH}")
    logger.info(f"Providers: {ort_session.get_providers()}")


def load_sklearn():
    global sklearn_pipeline
    import joblib

    for name in ["model.pkl", "model.joblib", "pipeline.pkl", "pipeline.joblib"]:
        p = os.path.join(MODEL_PATH, name)
        if os.path.exists(p):
            sklearn_pipeline = joblib.load(p)
            logger.info(f"Loaded sklearn pipeline from {p}")
            return

    mlflow_path = os.path.join(MODEL_PATH, "model", "model.pkl")
    if os.path.exists(mlflow_path):
        sklearn_pipeline = joblib.load(mlflow_path)
        logger.info(f"Loaded sklearn pipeline from {mlflow_path}")
        return

    raise FileNotFoundError(f"Cannot find sklearn model in {MODEL_PATH}")


def load_sklearn_onnx():
    global ort_session
    import onnxruntime as ort

    ort_session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    logger.info(f"Loaded sklearn ONNX model from {ONNX_PATH}")


def load_fasttext():
    global ft_model
    import fasttext

    for name in ["model.bin", "model.ftz", "fasttext_model.bin"]:
        p = os.path.join(MODEL_PATH, name)
        if os.path.exists(p):
            ft_model = fasttext.load_model(p)
            logger.info(f"Loaded FastText model from {p}")
            return

    for root, _, files in os.walk(MODEL_PATH):
        for f in files:
            if f.endswith((".bin", ".ftz")):
                ft_model = fasttext.load_model(os.path.join(root, f))
                logger.info(f"Loaded FastText model from {os.path.join(root, f)}")
                return

    raise FileNotFoundError(f"Cannot find FastText model in {MODEL_PATH}")


@app.on_event("startup")
def startup():
    load_label_map()

    if MODEL_TYPE in ("distilbert", "minilm"):
        if MODEL_BACKEND == "pytorch":
            load_transformer_pytorch()
        else:
            load_transformer_onnx()
    elif MODEL_TYPE == "tfidf_logreg":
        if MODEL_BACKEND == "onnx":
            load_sklearn_onnx()
        else:
            load_sklearn()
    elif MODEL_TYPE == "fasttext":
        load_fasttext()
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

    logger.info(f"Server ready: type={MODEL_TYPE}, backend={MODEL_BACKEND}")


def predict_transformer_pytorch(text: str):
    import torch

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0]
    probs = torch.softmax(logits, dim=0)
    idx = torch.argmax(probs).item()
    return get_label(idx), float(probs[idx])


def predict_transformer_onnx(text: str):
    encoded = tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=64)
    input_feed = {
        k: v for k, v in encoded.items()
        if k in [inp.name for inp in ort_session.get_inputs()]
    }
    outputs = ort_session.run(None, input_feed)
    logits = outputs[0][0]
    probs = np.exp(logits) / np.sum(np.exp(logits))
    idx = int(np.argmax(probs))
    return get_label(idx), float(probs[idx])


def predict_sklearn_native(text: str):
    probs = sklearn_pipeline.predict_proba([text])[0]
    idx = int(np.argmax(probs))
    classes = sklearn_pipeline.classes_
    return str(classes[idx]), float(probs[idx])


def predict_sklearn_onnx_fn(text: str):
    input_name = ort_session.get_inputs()[0].name
    input_data = np.array([text]).reshape(1, 1)
    outputs = ort_session.run(None, {input_name: input_data})
    label = outputs[0][0] if len(outputs) > 0 else "unknown"
    probs = outputs[1][0] if len(outputs) > 1 else {}
    if isinstance(probs, dict):
        confidence = max(probs.values()) if probs else 0.0
    elif hasattr(probs, '__len__'):
        confidence = float(max(probs))
    else:
        confidence = float(probs)
    return str(label), confidence


def predict_fasttext_fn(text: str):
    labels, probs = ft_model.predict(text.replace("\n", " "))
    label = labels[0].replace("__label__", "")
    return label, float(probs[0])


def _build_model_input(request: TransactionRequest) -> str:
    """Combine request fields into the text representation the model expects."""
    return f"{request.payee} {request.amount} {request.day_of_week}"


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TransactionRequest):
    text = _build_model_input(request)

    try:
        if MODEL_TYPE in ("distilbert", "minilm"):
            if MODEL_BACKEND == "pytorch":
                category, _ = predict_transformer_pytorch(text)
            else:
                category, _ = predict_transformer_onnx(text)
        elif MODEL_TYPE == "tfidf_logreg":
            if MODEL_BACKEND == "onnx":
                category, _ = predict_sklearn_onnx_fn(text)
            else:
                category, _ = predict_sklearn_native(text)
        elif MODEL_TYPE == "fasttext":
            category, _ = predict_fasttext_fn(text)
        else:
            raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return PredictionResponse(category=category)


@app.get("/health")
def health():
    return {"status": "healthy", "model_type": MODEL_TYPE, "backend": MODEL_BACKEND}
