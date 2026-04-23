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
| `config.yaml` | Shared configuration for Layer 2, Layer 3, MLflow, MinIO, and Postgres |
| `Dockerfile` | Container image for all evaluation jobs (build context: project root) |
| `requirements.txt` | Python dependencies (covers Layer 2 + Layer 3) |
| `../layer3/cluster.py` | DBSCAN clustering on per-user embeddings |
| `../layer3/namer.py` | LLM cluster naming via Anthropic API (falls back to majority label) |
| `../layer3/pipeline.py` | Weekly production run — clusters users, names clusters, writes to Postgres |
| `../layer3/evaluate.py` | Layer 3 evaluation — cluster quality metrics + LLM naming accuracy |

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

Run once before serving to pre-populate the store from historical data. Uses the first 70% of each user's transactions (by date) and saves `user_store.pkl` to the path set in `config.yaml` (`layer2.store_path`).

```bash
# From project root
python -m model_pipeline.layer2.build_store

# Docker (build context: project root)
docker build -f model_pipeline/layer2/Dockerfile -t actualbudget-evaluate .

docker run --rm --network host \
  -v "$(pwd)/model_pipeline/layer2/config.yaml:/app/model_pipeline/layer2/config.yaml" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/training/models/layer1/artifacts/fasttext.bin:/app/training/models/layer1/artifacts/fasttext.bin" \
  -e MINIO_ENDPOINT_URL=http://129.114.25.143:30900 \
  -e MINIO_ACCESS_KEY=minioadmin \
  -e MINIO_SECRET_KEY=minioadmin123 \
  actualbudget-evaluate \
  python -m model_pipeline.layer2.build_store
```

> `--network host` is required when running outside the k8s cluster so the container can reach the MinIO NodePort. `MINIO_ENDPOINT_URL` overrides the internal cluster DNS in `config.yaml`. The fasttext.bin mount is needed if the MLflow artifact backend does not have the model file — mount the locally trained artifact directly.

### Evaluate

There are two evaluation scripts with different purposes.

> **Note:** The Dockerfile copies `model_pipeline/` and `training/` from the project root, so all Docker builds below must be run from the **project root**, not `model_pipeline/layer2/`.

---

#### 1. CEX evaluation — Layer 1 + Layer 2 on `eval_cex.csv`

Uses `train.csv` (2022 CEX) to build the per-user store, then evaluates the full pipeline on `eval_cex.csv` (2024 CEX). Logs weighted F1, macro F1, Layer 2 routing %, per-source F1, and a per-row results CSV to MLflow.

**Step 1 — Build user store** (must run before evaluate):

```bash
docker run --rm --network host \
  -v "$(pwd)/model_pipeline/layer2/config.yaml:/app/model_pipeline/layer2/config.yaml" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/training/models/layer1/artifacts/fasttext.bin:/app/training/models/layer1/artifacts/fasttext.bin" \
  -e MINIO_ENDPOINT_URL=http://129.114.25.143:30900 \
  -e MINIO_ACCESS_KEY=minioadmin \
  -e MINIO_SECRET_KEY=minioadmin123 \
  actualbudget-evaluate \
  python -m model_pipeline.layer2.build_store
```

**Step 2 — Evaluate Layer 1+2 on eval_cex.csv**:

```bash
docker run --rm --network host \
  -v "$(pwd)/model_pipeline/layer2/config.yaml:/app/model_pipeline/layer2/config.yaml" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/training/models/layer1/artifacts/fasttext.bin:/app/training/models/layer1/artifacts/fasttext.bin" \
  -e MINIO_ENDPOINT_URL=http://129.114.25.143:30900 \
  -e MINIO_ACCESS_KEY=minioadmin \
  -e MINIO_SECRET_KEY=minioadmin123 \
  actualbudget-evaluate \
  python -m model_pipeline.evaluate \
    --eval-csv processed/eval_cex.csv \
    --run-name fasttext-layer2-cex \
    --experiment-suffix _cex
```

Results logged to MLflow under experiment `layer2-evaluation_cex`.

---

#### 2. MoneyData sliding window evaluation

Simulates user onboarding from scratch over years 2015–2022. At each eval year, all prior years seed the Layer 2 store; the current year is the test set. Shows how pipeline F1 and Layer 2 routing improve as personal history accumulates. Logs per-year metrics as MLflow steps under experiment `layer1-layer2-evaluation`. No pre-built store needed — the store is built in-memory per year.

```bash
docker run --rm --network host \
  -v "$(pwd)/model_pipeline/layer2/config.yaml:/app/model_pipeline/layer2/config.yaml" \
  -v "$(pwd)/training/models/layer1/artifacts/fasttext.bin:/app/training/models/layer1/artifacts/fasttext.bin" \
  -e MINIO_ENDPOINT_URL=http://129.114.25.143:30900 \
  -e MINIO_ACCESS_KEY=minioadmin \
  -e MINIO_SECRET_KEY=minioadmin123 \
  actualbudget-evaluate \
  python -m model_pipeline.layer2.evaluate_moneydata_sliding
```

---

| CLI flag | Purpose |
|---|---|
| `--config` | Path to `config.yaml` (default: `model_pipeline/layer2/config.yaml`) |
| `--eval-csv` | MinIO object path of a pre-split CSV; omit to use the default 70/30 split |
| `--run-name` | MLflow run name |
| `--experiment-suffix` | Appended to `experiment_name` from config (e.g. `_cex`) |

---

#### 3. Layer 3 evaluation — cluster quality + LLM naming accuracy

Runs DBSCAN on every user's embeddings from the store and measures three things:
- **Cluster quality**: silhouette score, coverage (% non-noise), noise %, mean cluster size
- **Naming accuracy**: calls `namer.py` (Anthropic API) on pure clusters and checks if the suggestion matches the majority ground-truth label
- **Eps sensitivity**: silhouette + coverage swept at `eps × 0.5`, `eps`, and `eps × 2.0` — no LLM calls during the sweep

Logs to MLflow under experiment `layer3-evaluation`. Writes a per-user results CSV as an artifact.

**Prerequisite**: `user_store.pkl` must exist — built in CEX Step 1 above.

```bash
docker run --rm --network host \
  -v "$(pwd)/model_pipeline/layer2/config.yaml:/app/model_pipeline/layer2/config.yaml" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -e ANTHROPIC_API_KEY=<your-key> \
  actualbudget-evaluate \
  python -m model_pipeline.layer3.evaluate
```

> `ANTHROPIC_API_KEY` is required for naming accuracy. Without it the API call returns 401, `namer.py` falls back to the majority label, and naming accuracy will show as 0.

---

#### 4. Layer 3 pipeline — production run (writes suggestions to Postgres)

Clusters all users in the store, names each cluster via the Anthropic API, and inserts pending suggestions into the `layer3_suggestions` table. Intended to run weekly.

**Prerequisite**: `user_store.pkl` must exist. Rebuild the image after `requirements.txt` was updated to add `psycopg2-binary`:

```bash
docker build -f model_pipeline/layer2/Dockerfile -t actualbudget-evaluate .
```

```bash
docker run --rm --network host \
  -v "$(pwd)/model_pipeline/layer2/config.yaml:/app/model_pipeline/layer2/config.yaml" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -e ANTHROPIC_API_KEY=<your-key> \
  -e POSTGRES_DSN="postgresql://mlops_user:mlops_pass@129.114.25.143:<pg-nodeport>/mlops" \
  actualbudget-evaluate \
  python -m model_pipeline.layer3.pipeline
```

`POSTGRES_DSN` overrides the internal k8s DSN (`postgres.mlops.svc.cluster.local`) in `config.yaml` — use the Postgres NodePort when running outside the cluster. Logs `total_users_processed`, `total_clusters_found`, and `total_suggestions_written` to MLflow under experiment `layer3-clustering`.

---

## config.yaml reference

```yaml
mlflow:
  tracking_uri: http://<host>:30500
  experiment_name: layer2-evaluation

minio:
  endpoint: http://minio.mlops.svc.cluster.local:9000   # internal k8s DNS
  bucket: data
  object: processed/train.csv
  # Override endpoint for Docker runs outside k8s:
  # -e MINIO_ENDPOINT_URL=http://<node-ip>:30900

layer1:
  model_uri: runs:/<run_id>/fasttext.bin   # MLflow artifact URI
  # If the MLflow artifact backend is unavailable, use a local file path:
  # model_uri: "/app/training/models/layer1/artifacts/fasttext.bin"
  model_version: fasttext-v1

layer2:
  model_name: sentence-transformers/all-mpnet-base-v2
  max_length: 128             # tokenizer max length
  k: 5                        # number of nearest neighbors
  similarity_threshold: 0.85  # min cosine similarity to trust Layer 2
  min_history: 10             # min stored transactions before Layer 2 activates
  store_path: /app/artifacts/user_store.pkl   # PVC mounted at /app/artifacts

postgres:
  dsn: "postgresql://mlops_user:mlops_pass@postgres.mlops.svc.cluster.local:5432/mlops"
  # Override at runtime: -e POSTGRES_DSN=...
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
| `psycopg2-binary` | Postgres connection for Layer 3 pipeline |
| `requests` | Anthropic API calls in Layer 3 namer |
