# Project Overview: Serving Evaluation

This project compares multiple NLP classifiers for transaction categorization, then selects one deployment-ready serving option for production-style API testing.

## What was done

- Evaluated **4 models**: `distilbert`, `minilm`, `tfidf_logreg`, `fasttext`.
- Measured model-level performance across latency, throughput, model size, and hardware backends.
- Selected **FastText + FastAPI (native backend)** as the final serving path.
- Benchmarked FastAPI under worker and concurrency load, then finalized the results table.

## Fast, Good, Cheap choices

- **Fast:** `fasttext_native_cpu` (lowest latency and highest throughput in model-level tests).
- **Good (best balance):** `minilm_static_quant_conservative` (strong speed/size tradeoff on CPU).
- **Cheap:** `tfidf_logreg_sklearn_cpu` (smallest model and low-cost CPU serving).

## Why Triton was not used for FastText

Triton is most useful for ONNX/TensorRT-style serving, dynamic batching, and multi-instance GPU pipelines.  
The selected FastText path is a native CPU model served directly in FastAPI, so Triton-specific features were not required for the final deployment track.

## Hardware used for selected model

- Selected model: **FastText**
- Model-level benchmark hardware: **CPU**
- FastAPI serving benchmark hardware: **CPU** (on a GPU-capable node, but FastText inference itself remained CPU-bound)

## Concurrency tested

FastAPI benchmarking covered:
- Sequential: **100 requests**
- Concurrent stress test: **1000 requests** at **16 workers**
- Additional concurrency sweep: **1, 4, 8, 16, 32 workers** (500-request runs)

## Input and output schema

### Input (`POST /predict`)
- `transaction_id` (string)
- `payee` (string)
- `amount` (float)
- `date` (ISO date string, e.g. `2024-03-15`)

### Output
- `transaction_id` (string)
- `prediction_category` (string)
- `confidence` (float)
- `model_version` (string)

## Optimization summary

### 1) Model-level optimizations
- ONNX conversion and runtime optimization
- Dynamic quantization
- Static quantization (aggressive and conservative)
- Backend acceleration comparisons (CPU, CUDA, TensorRT, OpenVINO where applicable)

### 2) System-level optimizations
- FastAPI worker scaling (`UVICORN_WORKERS=1,2,4,8`)
- Concurrency tuning and stress benchmarking
- Selection of best serving worker configuration from measured latency/throughput

### 3) Infrastructure-level optimizations
- Chameleon bare-metal GPU-capable environment provisioning
- Dockerized, reproducible evaluation and serving environments
- Container-isolated benchmark runs for consistent results collection

### Combined optimizations used
- **Model + hardware combinations:** quantized/optimized models tested across CPU/GPU/OpenVINO backends.
- **System + serving combinations:** selected model deployed on FastAPI with worker and concurrency tuning.
- **Final selected combination:** FastText native CPU serving with tuned FastAPI worker configuration.
