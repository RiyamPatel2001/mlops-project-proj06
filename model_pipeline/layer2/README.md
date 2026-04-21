# model_pipeline/layer2/

Layer 2 personalization layer. Uses per-user transaction history and semantic similarity to override Layer 1 predictions for users with sufficient history. Falls back to Layer 1 for new or cold-start users.

## How it works

```
Incoming transaction (payee, user_id)
        │
        ▼
┌─────────────────┐
│    Layer 1      │  fastText — population-level classifier
│  (always runs)  │
└────────┬────────┘
         │
         ▼
  Has user ≥ min_history transactions?
         │
    No ──┴── Yes
    │         │
    │         ▼
    │  ┌─────────────────┐
    │  │  Embed payee    │  all-mpnet-base-v2 → 768-dim unit vector
    │  └────────┬────────┘
    │           │
    │           ▼
    │  ┌─────────────────┐
    │  │  k-NN search    │  cosine similarity against user's store
    │  └────────┬────────┘
    │           │
    │           ▼
    │  max_similarity ≥ threshold?
    │           │
    │      No ──┴── Yes
    │      │         │
    │      │         ▼
    │      │   Layer 2 prediction (majority vote of top-k neighbors)
    │      │
    └──────┴──► Layer 1 prediction
                      │
                      ▼
              Update user store
```

## Files

| File | Role |
|---|---|
| `predictor.py` | Unified inference interface — combines Layer 1 + Layer 2 |
| `embedder.py` | Loads all-mpnet-base-v2, exposes `embed()` / `embed_batch()` |
| `matcher.py` | Pure cosine similarity, `get_top_k`, `majority_vote` |
| `user_store.py` | Runtime in-memory store with atomic-write persistence |
| `build_store.py` | Offline script — builds `user_store.pkl` from historical data |
| `evaluate_moneydata.py` | Cold-start demo — bootstraps store from moneydata 2015–2020, evaluates on 2021–2022 holdout |
| `../evaluate.py` | General-purpose pipeline evaluation — any eval CSV from MinIO, logs to MLflow |
| `config.yaml` | Layer 2 configuration |
| `Dockerfile` | Container image for build-store / evaluation jobs (build context: project root) |
| `requirements.txt` | Python dependencies |

## Quick start

### Run inference

```python
from model_pipeline.layer2.predictor import Predictor

predictor = Predictor(config_path="model_pipeline/layer2/config.yaml")

result = predictor.predict(
    transaction_id="txn_001",
    user_id="user_42",
    payee="Whole Foods Market",
    amount=54.20,
    date="2024-03-15",
)
# result keys: transaction_id, user_id, prediction_category,
#              confidence, model_version, source ("layer1" or "layer2")
```

### Build the user store (offline)

Run once before serving to pre-populate the store from historical data:

```bash
# From project root
python -m model_pipeline.layer2.build_store

# Docker
docker build -f model_pipeline/layer2/Dockerfile -t layer2 .
docker run --rm \
  -v $(pwd)/model_pipeline/layer2/config.yaml:/app/model_pipeline/layer2/config.yaml \
  layer2
```

The script uses the first 70% of each user's transactions (by date) and saves `artifacts/user_store.pkl`.

### Evaluate

There are two evaluation scripts with different purposes.

> **Note:** The Dockerfile copies `model_pipeline/` and `training/` from the project root, so all Docker builds below must be run from the **project root**, not `model_pipeline/layer2/`.

---

#### 1. Cold-start demo — `evaluate_moneydata.py`

Bootstraps the Layer 2 store from moneydata 2015–2020 user history and evaluates the combined pipeline on the 2021–2022 holdout. Demonstrates that Layer 2 bootstrapping solves the cold-start problem for out-of-distribution (UK) data where Layer 1 alone achieves weighted F1 ≈ 0.11. Does not require MLflow.

```bash
# Local (from project root)
python -m model_pipeline.layer2.evaluate_moneydata

# Docker (build and run from project root)
docker build -f model_pipeline/layer2/Dockerfile -t actualbudget-evaluate-moneydata .

docker run --rm \
  -v "$(pwd)/model_pipeline/layer2/config.yaml:/app/model_pipeline/layer2/config.yaml" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -e MINIO_ACCESS_KEY="${MINIO_ACCESS_KEY:-minioadmin}" \
  -e MINIO_SECRET_KEY="${MINIO_SECRET_KEY:-minioadmin123}" \
  actualbudget-evaluate-moneydata \
  python -m model_pipeline.layer2.evaluate_moneydata
```

---

#### 2. General-purpose pipeline evaluation — `model_pipeline.evaluate`

Batch-evaluates the full Layer 1 + Layer 2 pipeline and logs metrics to MLflow. Supports a pre-split eval CSV via `--eval-csv` (e.g. `processed/eval_moneydata.csv`, `processed/eval_cex.csv`) or defaults to a 70/30 temporal split on the train CSV. Logs weighted F1, macro F1, Layer 2 routing percentage, per-source F1, and a per-row results CSV as MLflow artifacts.

```bash
# Local (from project root)
python -m model_pipeline.evaluate \
  --config model_pipeline/layer2/config.yaml \
  --eval-csv processed/eval_moneydata.csv \
  --run-name fasttext-layer2-realworld \
  --experiment-suffix _moneydata

# Docker (build and run from project root)
docker build -f model_pipeline/layer2/Dockerfile -t categorizer-eval .

docker run --rm \
  -v "$(pwd)/model_pipeline/layer2/config.yaml:/app/model_pipeline/layer2/config.yaml" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -e MLFLOW_TRACKING_URI=http://129.114.26.157:30500 \
  -e MINIO_ACCESS_KEY=<your_access_key> \
  -e MINIO_SECRET_KEY=<your_secret_key> \
  -e GIT_SHA="$(git rev-parse HEAD)" \
  categorizer-eval \
  python -m model_pipeline.evaluate \
    --config model_pipeline/layer2/config.yaml \
    --eval-csv processed/eval_moneydata.csv \
    --run-name fasttext-layer2-realworld \
    --experiment-suffix _moneydata
```

| CLI flag | Purpose |
|---|---|
| `--config` | Path to `config.yaml` |
| `--eval-csv` | MinIO object path of a pre-split CSV; omit to use the default 70/30 split |
| `--run-name` | MLflow run name |
| `--experiment-suffix` | Appended to `experiment_name` from config (e.g. `_moneydata`) |

## config.yaml reference

```yaml
mlflow:
  tracking_uri: http://<host>:30500
  experiment_name: layer2-evaluation

minio:
  endpoint: http://...
  bucket: data
  object: processed/train.csv

layer1:
  model_uri: runs:/<run_id>/fasttext.bin   # MLflow artifact URI
  model_version: fasttext-v1

layer2:
  model_name: sentence-transformers/all-mpnet-base-v2
  max_length: 128          # tokenizer max length
  k: 5                     # number of nearest neighbors
  similarity_threshold: 0.85  # min cosine similarity to trust Layer 2
  min_history: 10          # min stored transactions before Layer 2 activates
  store_path: artifacts/user_store.pkl
```

## User store format

The store is a pickled `dict` keyed by `user_id`:

```python
{
    "user_42": {
        "embeddings": np.ndarray,  # (n, 768) float32, unit-normalized
        "labels":     list[str],   # category strings, parallel to embeddings
        "payees":     list[str],   # original payee strings (interpretability)
    },
    ...
}
```

All embeddings are L2-normalized so cosine similarity reduces to a dot product.

## Cold-start behavior

When a user has fewer than `min_history` stored transactions, `predictor.predict()` returns the Layer 1 result but still embeds and appends the transaction to the store so history accumulates. Once the threshold is crossed, Layer 2 activates automatically on the next call.

## Dependencies

See `requirements.txt`. Key packages:

| Package | Purpose |
|---|---|
| `torch` / `transformers` | all-mpnet-base-v2 embedding model |
| `fasttext-wheel` | Layer 1 model loading |
| `mlflow` | Artifact download + experiment tracking |
| `minio` | Object store access |
| `numpy` / `pandas` | Store manipulation and data loading |
