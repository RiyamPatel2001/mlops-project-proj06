# training/

Layer 1 training pipeline for the transaction categorizer. Trains a text classifier on merchant payee strings and logs all runs to MLflow.

## Directory structure

```
training/
├── train.py                      # Main training entrypoint
├── retrain.py                    # Triggered retraining with new MinIO data
├── evaluate.py                   # Shared evaluation + MLflow metric logging
├── utils.py                      # Payee normalization + user-stratified split
├── config.yaml                   # Single source of truth for all training config
├── Dockerfile                    # CPU training image
├── Dockerfile.gpu                # GPU training image
├── Dockerfile.retrain            # Retraining image
├── models/
│   └── layer1/
│       ├── tfidf_logreg.py       # TF-IDF + Logistic Regression
│       ├── fasttext.py           # fastText n-gram classifier
│       ├── minilm_finetune.py    # Fine-tuned all-MiniLM-L6-v2
│       ├── distilbert_finetune.py# Fine-tuned DistilBERT
│       ├── mpnet_finetune.py     # Fine-tuned all-mpnet-base-v2
│       ├── transformer_base.py   # Shared HuggingFace training loop
│       └── artifacts/            # Saved model files (gitignored binaries)
├── mlruns/                       # Local MLflow tracking (fallback)
└── mlartifacts/                  # Local MLflow artifacts (fallback)
```

## Supported models

| Key | Architecture | Notes |
|-----|---|---|
| `tfidf_logreg` | TF-IDF + Logistic Regression | Fast baseline, sklearn joblib output |
| `fasttext` | fastText n-gram | Best speed/accuracy tradeoff, `.bin` output |
| `minilm` | all-MiniLM-L6-v2 fine-tune | Transformer, 384-dim embeddings |
| `distilbert` | DistilBERT fine-tune | Smaller BERT, 768-dim |
| `mpnet` | all-mpnet-base-v2 fine-tune | Strongest transformer baseline |

Switch candidates by changing the `model` key in `config.yaml`.

## Quick start

```bash
# Local run (from training/)
python train.py --config config.yaml
```

All Docker commands below assume you are in the `training/` directory.

---

### CPU training (single run)

```bash
# Build
docker build -t categorizer-training .

# Run — --network host lets the container reach K8s NodePort services (MinIO :30900, MLflow :30500)
docker run --rm \
  --network host \
  -e MLFLOW_TRACKING_URI=http://<chameleon-ip>:30500 \
  -e MINIO_ACCESS_KEY=minioadmin \
  -e MINIO_SECRET_KEY=minioadmin123 \
  -e GIT_SHA="$(git rev-parse HEAD)" \
  -v "$(pwd)/config.yaml:/app/training/config.yaml" \
  -v "$(pwd)/models/layer1/artifacts:/app/training/models/layer1/artifacts" \
  categorizer-training
```

---

### CPU sweep (fasttext + tfidf_logreg)

`sweep-cpu.py` runs a hyperparameter grid over CPU-only models — fasttext (`lr` ∈ [0.5, 1.0]) and tfidf_logreg (`C` ∈ [1.0, 10.0]). Pass `--model` to sweep a single model.

```bash
# Build (same image as single-run CPU)
docker build -t categorizer-training .

# Sweep a singe model (fasttext)
docker run --rm \
  --network host \
  --entrypoint python \
  -e MLFLOW_TRACKING_URI=http://129.114.25.143:30500 \
  -e MINIO_ENDPOINT_URL=http://129.114.25.143:30900 \
  -e MINIO_ACCESS_KEY=minioadmin \
  -e MINIO_SECRET_KEY=minioadmin123 \
  -e GIT_SHA="$(git rev-parse HEAD)" \
  -v "$(pwd)/train.py:/app/training/train.py" \
  -v "$(pwd)/config.yaml:/app/training/config.yaml" \
  -v "$(pwd)/sweep-cpu.py:/app/training/sweep-cpu.py" \
  ghcr.io/riyampatel2001/mlops-project-proj06/training-cpu:latest \
  /app/training/sweep-cpu.py --config /app/training/config.yaml --model fasttext

# Sweep a single model (tfidf_logreg)
docker run --rm \
  --network host \
  --entrypoint python \
  -e MLFLOW_TRACKING_URI=http://129.114.25.143:30500 \
  -e MINIO_ACCESS_KEY=minioadmin \
  -e MINIO_SECRET_KEY=minioadmin123 \
  -e GIT_SHA="$(git rev-parse HEAD)" \
  -v "$(pwd)/config.yaml:/app/training/config.yaml" \
  -v "$(pwd)/sweep-cpu.py:/app/training/sweep-cpu.py" \
  ghcr.io/riyampatel2001/mlops-project-proj06/training-cpu:latest \
  /app/training/sweep-cpu.py --config /app/training/config.yaml --model tfidf_logreg
```

---

### GPU training (single run)

```bash
# Build
docker build -f Dockerfile.gpu -t categorizer-training-gpu .

# Run — trains whichever model is set in config.yaml
docker run --rm --gpus all \
  --network host \
  -e MLFLOW_TRACKING_URI=http://129.114.25.143:30500 \
  -e MINIO_ACCESS_KEY=minioadmin \
  -e MINIO_SECRET_KEY=minioadmin123 \
  -e GIT_SHA="$(git rev-parse HEAD)" \
  -v "$(pwd)/config.yaml:/app/training/config.yaml" \
  -v "$(pwd)/models/layer1/artifacts:/app/training/models/layer1/artifacts" \
  categorizer-training-gpu
```

---

### GPU evaluation (eval_layer1.py)

`eval_layer1.py` evaluates a trained Layer 1 model (fasttext or transformer) on an arbitrary eval CSV from MinIO and logs results to MLflow. Pass `--no-quality-gate` for OOD datasets where lower scores are expected.

```bash
# Build (same image as GPU training)
docker build -f Dockerfile.gpu -t categorizer-training-gpu .

# Evaluate on eval_cex.csv
docker run --rm --gpus all \
  --network host \
  -e MLFLOW_TRACKING_URI=http://129.114.25.143:30500 \
  -e MINIO_ENDPOINT_URL=http://129.114.25.143:30900 \
  -e MINIO_ACCESS_KEY=minioadmin \
  -e MINIO_SECRET_KEY=minioadmin123 \
  -e GIT_SHA="$(git rev-parse HEAD)" \
  -v "$(pwd)/config.yaml:/app/training/config.yaml" \
  -v "$(pwd)/eval_layer1.py:/app/training/eval_layer1.py" \
  -v "$(pwd)/evaluate.py:/app/training/evaluate.py" \
  -v "$(pwd)/../data:/app/data" \
  --entrypoint python \
  categorizer-training-gpu \
  /app/training/eval_layer1.py \
    --run-id    <mlflow-run-id> \
    --model-type minilm \
    --eval-csv  processed/eval_cex.csv \
    --run-name  eval-minilm-cex

# Evaluate on eval_moneydata.csv (OOD — disable quality gate)
docker run --rm --gpus all \
  --network host \
  -e MLFLOW_TRACKING_URI=http://129.114.25.143:30500 \
  -e MINIO_ENDPOINT_URL=http://129.114.25.143:30900 \
  -e MINIO_ACCESS_KEY=minioadmin \
  -e MINIO_SECRET_KEY=minioadmin123 \
  -e GIT_SHA="$(git rev-parse HEAD)" \
  -v "$(pwd)/config.yaml:/app/training/config.yaml" \
  -v "$(pwd)/eval_layer1.py:/app/training/eval_layer1.py" \
  -v "$(pwd)/evaluate.py:/app/training/evaluate.py" \
  -v "$(pwd)/../data:/app/data" \
  --entrypoint python \
  categorizer-training-gpu \
  /app/training/eval_layer1.py \
    --run-id    <mlflow-run-id> \
    --model-type minilm \
    --eval-csv  processed/eval_moneydata.csv \
    --run-name  eval-minilm-moneydata \
    --no-quality-gate
```

`--model-type` accepts `fasttext`, `minilm`, `distilbert`, or `mpnet`. `--run-id` is the MLflow run ID from a previous training run.

### CPU evaluation (fasttext)

```bash
# Build (same image as CPU training)
docker build -t categorizer-training .

# Evaluate fasttext on eval_cex.csv
docker run --rm \
  --network host \
  -e MLFLOW_TRACKING_URI=http://129.114.25.143:30500 \
  -e MINIO_ENDPOINT_URL=http://129.114.25.143:30900 \
  -e MINIO_ACCESS_KEY=minioadmin \
  -e MINIO_SECRET_KEY=minioadmin123 \
  -e GIT_SHA="$(git rev-parse HEAD)" \
  -v "$(pwd)/config.yaml:/app/training/config.yaml" \
  -v "$(pwd)/eval_layer1.py:/app/training/eval_layer1.py" \
  -v "$(pwd)/evaluate.py:/app/training/evaluate.py" \
  -v "$(pwd)/../data:/app/data" \
  --entrypoint python \
  categorizer-training \
  /app/training/eval_layer1.py \
    --run-id    <mlflow-run-id> \
    --model-type fasttext \
    --eval-csv  processed/eval_cex.csv \
    --run-name  eval-fasttext-cex

# Evaluate fasttext on eval_moneydata.csv (OOD — disable quality gate)
docker run --rm \
  --network host \
  -e MLFLOW_TRACKING_URI=http://129.114.25.143:30500 \
  -e MINIO_ENDPOINT_URL=http://129.114.25.143:30900 \
  -e MINIO_ACCESS_KEY=minioadmin \
  -e MINIO_SECRET_KEY=minioadmin123 \
  -e GIT_SHA="$(git rev-parse HEAD)" \
  -v "$(pwd)/config.yaml:/app/training/config.yaml" \
  -v "$(pwd)/eval_layer1.py:/app/training/eval_layer1.py" \
  -v "$(pwd)/evaluate.py:/app/training/evaluate.py" \
  -v "$(pwd)/../data:/app/data" \
  --entrypoint python \
  categorizer-training \
  /app/training/eval_layer1.py \
    --run-id    <mlflow-run-id> \
    --model-type fasttext \
    --eval-csv  processed/eval_moneydata.csv \
    --run-name  eval-fasttext-moneydata \
    --no-quality-gate
```

---

### GPU sweep (minilm / distilbert / mpnet)

`sweep-gpu.py` runs a learning-rate grid over transformer models. Requires a GPU host and `--gpus all`.

```bash
# Build
docker build -f Dockerfile.gpu -t categorizer-training-gpu .

# Sweep a single transformer model (e.g. minilm)
docker run --rm --gpus all \
  --network host \
  -e MLFLOW_TRACKING_URI=http://129.114.25.143:30500 \
  -e MINIO_ACCESS_KEY=minioadmin \
  -e MINIO_SECRET_KEY=minioadmin123 \
  -e GIT_SHA="$(git rev-parse HEAD)" \
  -v "$(pwd)/config.yaml:/app/training/config.yaml" \
  -v "$(pwd)/sweep-gpu.py:/app/training/sweep-gpu.py" \
  -v "$(pwd)/models/layer1/artifacts:/app/training/models/layer1/artifacts" \
  --entrypoint python \
  categorizer-training-gpu \
  /app/training/sweep-gpu.py --model minilm

# Sweep all GPU models (omit --model)
docker run --rm --gpus all \
  --network host \
  -e MLFLOW_TRACKING_URI=http://129.114.25.143:30500 \
  -e MINIO_ACCESS_KEY=minioadmin \
  -e MINIO_SECRET_KEY=minioadmin123 \
  -e GIT_SHA="$(git rev-parse HEAD)" \
  -v "$(pwd)/config.yaml:/app/training/config.yaml" \
  -v "$(pwd)/sweep-gpu.py:/app/training/sweep-gpu.py" \
  -v "$(pwd)/models/layer1/artifacts:/app/training/models/layer1/artifacts" \
  --entrypoint python \
  categorizer-training-gpu \
  /app/training/sweep-gpu.py
```

---

## Retraining

`retrain.py` is the triggered-retraining entrypoint. It:
1. Downloads the latest versioned dataset from MinIO (`data/retraining/`)
2. Merges it with the base raw dataset (unless `--no-merge`)
3. Retrains and evaluates the model
4. Promotes the candidate only if it beats the current production model by at least `promotion.minimum_accuracy_delta`

```bash
# Local
python retrain.py --config config.yaml
python retrain.py --config config.yaml --force      # retrain even if dataset version is current
python retrain.py --config config.yaml --no-merge   # retrain on new data only
```

#### Docker (from `training/`)

```bash
# Build
docker build -f Dockerfile.retrain -t categorizer-retrain .

# Run (standard — merges new data with base dataset)
docker run --rm \
  -v "$(pwd):/app/training" \
  -v "$(pwd)/../data:/app/data" \
  -e MLFLOW_TRACKING_URI=http://129.114.26.157:30500 \
  -e MINIO_ENDPOINT_URL=http://10.43.4.193:9000 \
  -e MINIO_ACCESS_KEY=minioadmin \
  -e MINIO_SECRET_KEY=minioadmin123 \
  -e GIT_SHA="$(git rev-parse HEAD)" \
  categorizer-retrain

# Run with --no-merge (retrain on new data only, skip base dataset)
docker run --rm \
  -v "$(pwd):/app/training" \
  -v "$(pwd)/../data:/app/data" \
  -e MLFLOW_TRACKING_URI=http://129.114.26.157:30500 \
  -e MINIO_ENDPOINT_URL=http://10.43.4.193:9000 \
  -e MINIO_ACCESS_KEY=minioadmin \
  -e MINIO_SECRET_KEY=minioadmin123 \
  -e GIT_SHA="$(git rev-parse HEAD)" \
  categorizer-retrain \
  --no-merge
```

The retrain image mounts `training/` as a volume so you never need to rebuild after editing `retrain.py` or `config.yaml`. The active model pointer is written to `../serving/layer1_registry.json` (configurable via `MODEL_REGISTRY_PATH`).

## config.yaml reference

```yaml
model: fasttext              # which model to train

data:
  raw_path: ...              # MinIO URL or local path to raw CSV
  processed_dir: ../data/processed/
  retraining_prefix: data/retraining/

val_frac: 0.2                # fraction of users held out for validation
random_state: 42

model_output_dir: models/layer1/artifacts/

mlflow:
  tracking_uri: http://<host>:30500
  experiment_name: layer1-categorizer

promotion:
  target_tier: fast          # registry tier to promote into
  registry_path: ../serving/layer1_registry.json
  minimum_accuracy_delta: 0.01

minio:
  endpoint: http://...
  bucket: data
```

Model-specific hyperparameters live under their own key (`tfidf_logreg`, `fasttext`, `minilm`, `distilbert`, `mpnet`) — only the section matching `model` is used at runtime.

## Quality gate

`evaluate.py` enforces a quality gate on every run. Defaults:

| Metric | Minimum |
|---|---|
| Weighted F1 | 0.75 |
| Macro F1 | 0.55 |

Override via a `quality_gate` key in `config.yaml`. If the gate fails, the process exits with code 1 and the MLflow run is tagged `quality_gate: failed`.

## MLflow output

Each run logs:
- All config params (top-level scalars + active model hyperparameters)
- `weighted_f1`, `macro_f1`, `accuracy`, `training_time_seconds`
- Per-class F1 scores (`f1_<Category>`)
- `classification_report.json` and `classification_report.txt` artifacts
- The trained model artifact (`.bin`, `.joblib`, or HuggingFace directory)

## Environment variables

| Variable | Purpose |
|---|---|
| `MLFLOW_TRACKING_URI` | Overrides `mlflow.tracking_uri` in config |
| `DATA_RAW_PATH` | Overrides `data.raw_path` in config — use a local path when MinIO NodePort is unreachable |
| `MINIO_ENDPOINT_URL` | Overrides `minio.endpoint` in config |
| `MINIO_ACCESS_KEY` | MinIO credentials |
| `MINIO_SECRET_KEY` | MinIO credentials |
| `MODEL_REGISTRY_PATH` | Overrides `promotion.registry_path` |
| `GIT_SHA` | Git commit SHA tagged on the MLflow run |
