# model_pipeline/

End-to-end evaluation and inference pipeline for the three-layer transaction categorizer. This README is the single reference for running evaluations locally via Docker, understanding how layers connect, and troubleshooting known issues.

---

## Directory Structure

```
model_pipeline/
├── evaluate.py                        # Layer 1+2 batch evaluation (--eval-csv flag)
├── layer2/
│   ├── build_store.py                 # Offline: builds user_store.pkl from MinIO CSV
│   ├── embedder.py                    # all-mpnet-base-v2 wrapper (embed / embed_batch)
│   ├── matcher.py                     # cosine similarity, get_top_k, majority_vote
│   ├── predictor.py                   # unified Layer 1+2 inference interface
│   ├── user_store.py                  # runtime in-memory store with atomic persistence
│   ├── evaluate_moneydata.py          # cold-start demo (moneydata 2015-2020 bootstrap)
│   ├── evaluate_moneydata_sliding.py  # sliding window evaluation (2015-2022)
│   ├── config.yaml                    # shared config for all layers (Layer 2, 3, MLflow, MinIO, Postgres)
│   ├── Dockerfile                     # single image for all evaluation jobs
│   └── requirements.txt               # Python deps (Layer 2 + Layer 3)
└── layer3/
    ├── cluster.py                     # DBSCAN clustering on per-user embeddings
    ├── namer.py                       # LLM cluster naming via Anthropic API
    ├── pipeline.py                    # weekly production run → Postgres
    ├── evaluate.py                    # cluster quality + LLM naming accuracy
    └── migrations/
        └── 001_create_layer3_suggestions.sql
```

---

## How the Layers Connect

```
Incoming transaction
        │
        ▼
  Layer 1 (fastText)
  population-level, always runs
        │
        ▼
  Layer 2 (all-mpnet-base-v2)
  per-user semantic similarity
  overrides Layer 1 if user has ≥ min_history
  transactions AND cosine similarity ≥ threshold
        │
        ▼
  Final prediction returned to serving
        │
        ▼  (offline, weekly)
  Layer 3 (DBSCAN + Claude)
  clusters user embeddings from layer2_examples (prod) or user_store.pkl (local)
  names clusters via Anthropic API
  writes pending suggestions to Postgres
```

---

## Shared Docker Image

All evaluation and pipeline jobs use a single image built from the project root:

```bash
docker build -f model_pipeline/layer2/Dockerfile -t actualbudget-evaluate .
```

The Dockerfile copies both `model_pipeline/` and `training/` (needed for `training.utils.normalize_payee`), installs `model_pipeline/layer2/requirements.txt`, and removes `config.yaml` — it is always mounted at runtime.

**Rebuild required when:** `requirements.txt` changes (e.g. after `psycopg2-binary` was added for Layer 3).

---

## config.yaml

Located at `model_pipeline/layer2/config.yaml`. Mounted at runtime into every container at `/app/model_pipeline/layer2/config.yaml`. Never baked into the image.

```yaml
mlflow:
  tracking_uri: "http://129.114.25.143:30500"
  experiment_name: "layer2-evaluation"

minio:
  endpoint: "http://minio.mlops.svc.cluster.local:9000"  # internal k8s DNS
  bucket: "data"
  object: "processed/train.csv"                          # used by build_store

layer1:
  model_uri: "file:///app/training/models/layer1/artifacts/fasttext.bin"
  # Use file:// URI when MLflow artifact backend cannot serve the .bin file.
  # For a working MLflow run: "runs:/<run-id>/fasttext.bin"
  model_version: "fasttext-v1"

layer2:
  model_name: "sentence-transformers/all-mpnet-base-v2"
  max_length: 128
  k: 5
  similarity_threshold: 0.90
  min_history: 10
  store_path: "http://129.114.25.143:30900/data/user_store/user_store_full.pkl"  # MinIO object

layer3:
  eps: 0.15
  min_samples: 3
  store_path: "http://129.114.25.143:30900/data/user_store/user_store_full.pkl"  # MinIO object

postgres:
  dsn: "postgresql://mlops_user:mlops_pass@postgres.mlops.svc.cluster.local:5432/mlops"
  # Internal k8s DSN — override with POSTGRES_DSN env var for Docker runs outside cluster
```

---

## Infrastructure Details

| Service | Internal DNS | Reachable from host (`--network host`) |
|---|---|---|
| MLflow | `mlflow.mlops.svc.cluster.local` | `129.114.25.143:30500` |
| MinIO | `minio.mlops.svc.cluster.local:9000` | `129.114.25.143:30900` |
| Postgres | `postgres.mlops.svc.cluster.local:5432` | `10.43.98.71:5432` (ClusterIP — no NodePort) |

---

## Known Issues and Workarounds

### 1. MLflow artifact backend returns HTTP 500 for fasttext.bin

The fasttext model was trained and registered in MLflow but the `.bin` file was never uploaded to the MinIO artifact store. `mlflow.artifacts.download_artifacts("runs:/...")` returns a 500 from the backend.

**Fix:** Set `layer1.model_uri` in `config.yaml` to a `file://` URI pointing to the locally trained artifact, and volume-mount it into the container:

```yaml
layer1:
  model_uri: "file:///app/training/models/layer1/artifacts/fasttext.bin"
```

```bash
-v "$(pwd)/training/models/layer1/artifacts/fasttext.bin:/app/training/models/layer1/artifacts/fasttext.bin"
```

### 2. MinIO internal DNS not reachable from Docker outside k8s

`minio.mlops.svc.cluster.local` only resolves inside the cluster. Docker containers on the host cannot use it even with `--network host` (DNS resolution fails).

**Fix:** Pass `MINIO_ENDPOINT_URL` env var — `build_store.py` and all MinIO clients check this first:

```bash
-e MINIO_ENDPOINT_URL=http://129.114.25.143:30900
```

### 3. Postgres has no NodePort — ClusterIP only

The postgres service is ClusterIP-only (`10.43.98.71:5432`). From the host node with `--network host`, the ClusterIP is reachable because k3s sets up the necessary iptables rules.

**Fix:** Use the ClusterIP directly:

```bash
-e POSTGRES_DSN="postgresql://mlops_user:mlops_pass@10.43.98.71:5432/mlops"
```

Do not use the internal DNS (`postgres.mlops.svc.cluster.local`) — it does not resolve from Docker containers.

### 4. Layer 3 scripts use relative imports — must run with `-m`

`pipeline.py` and `evaluate.py` in `layer3/` use `from .cluster import ...` and `from .namer import ...`. Running them as scripts (`python model_pipeline/layer3/pipeline.py`) raises `ImportError: attempted relative import with no known parent package`.

**Fix:** Always use the `-m` flag:

```bash
python -m model_pipeline.layer3.pipeline
python -m model_pipeline.layer3.evaluate
```

### 5. user_store.pkl is in MinIO, not a local PVC mount

`user_store.pkl` lives at `data/user_store/user_store_full.pkl` in MinIO (bucket `data`). Docker containers access it via the NodePort (`--network host` + `MINIO_ENDPOINT_URL`). No local volume mount is needed.

**Pull locally for inspection:**

```bash
mc alias set myminio http://129.114.25.143:30900 minioadmin minioadmin123
mc cp myminio/data/user_store/user_store_full.pkl ./artifacts/user_store_full.pkl
```

**Build fresh and upload:**

```bash
docker run --rm --network host \
  -v "$(pwd)/model_pipeline/layer2/config.yaml:/app/model_pipeline/layer2/config.yaml" \
  -v "$(pwd)/training/models/layer1/artifacts/fasttext.bin:/app/training/models/layer1/artifacts/fasttext.bin" \
  -e MINIO_ENDPOINT_URL=http://129.114.25.143:30900 \
  -e MINIO_ACCESS_KEY=minioadmin \
  -e MINIO_SECRET_KEY=minioadmin123 \
  actualbudget-evaluate \
  python -m model_pipeline.layer2.build_store
```

---

## Evaluation Commands

All commands run from the **project root**.

### Layer 1 + Layer 2 — CEX Evaluation

Evaluates the full Layer 1+2 pipeline on `eval_cex.csv` (2024 CEX, in-distribution). Builds the user store from `train.csv` (2022 CEX, first 70% per user).

**Step 1 — Build user store:**

```bash
docker run --rm --network host \
  -v "$(pwd)/model_pipeline/layer2/config.yaml:/app/model_pipeline/layer2/config.yaml" \
  -v "$(pwd)/training/models/layer1/artifacts/fasttext.bin:/app/training/models/layer1/artifacts/fasttext.bin" \
  -e MINIO_ENDPOINT_URL=http://129.114.25.143:30900 \
  -e MINIO_ACCESS_KEY=minioadmin \
  -e MINIO_SECRET_KEY=minioadmin123 \
  actualbudget-evaluate \
  python -m model_pipeline.layer2.build_store
```

**Step 2 — Evaluate:**

```bash
docker run --rm --network host \
  -v "$(pwd)/model_pipeline/layer2/config.yaml:/app/model_pipeline/layer2/config.yaml" \
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

MLflow experiment: `layer2-evaluation_cex`
Metrics logged: `weighted_f1`, `macro_f1`, `layer2_routing_pct`, `layer1_weighted_f1`, `layer2_weighted_f1`, `dropped_rows`

---

### Layer 1 + Layer 2 — MoneyData Sliding Window

Simulates user onboarding year by year (2015–2022). Each eval year uses all prior years as the bootstrap store. No pre-built store needed — built in-memory per year.

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

MLflow experiment: `layer1-layer2-evaluation`
Metrics logged per year (step = eval_year): `weighted_f1`, `macro_f1`, `layer2_routing_pct`, `layer1_only_weighted_f1`, `bootstrap_size`, `eval_size`

---

### Layer 3 — Cluster Quality Evaluation

Measures DBSCAN cluster quality and LLM naming accuracy against ground-truth labels. Requires `user_store.pkl`.

```bash
docker run --rm --network host \
  -v "$(pwd)/model_pipeline/layer2/config.yaml:/app/model_pipeline/layer2/config.yaml" \
  -e MINIO_ENDPOINT_URL=http://129.114.25.143:30900 \
  -e MINIO_ACCESS_KEY=minioadmin \
  -e MINIO_SECRET_KEY=minioadmin123 \
  -e ANTHROPIC_API_KEY=<your-key> \
  actualbudget-evaluate \
  python -m model_pipeline.layer3.evaluate
```

MLflow experiment: `layer3-evaluation`
Metrics logged: `mean_silhouette`, `median_silhouette`, `mean_coverage`, `mean_cluster_size`, `n_noise_pct`, `naming_accuracy`, `n_pure_clusters_evaluated`, `silhouette_eps_tight/default/loose`, `coverage_eps_tight/default/loose`
Artifact: `layer3_eval_results.csv` (per-user breakdown)

> `ANTHROPIC_API_KEY` is required for naming accuracy. Without it, `namer.py` falls back to the majority label and naming accuracy will show as 0.

---

### Layer 3 — Pipeline (production run, writes to Postgres)

Clusters all users, names each cluster via the Anthropic API, and inserts pending suggestions into `public.layer3_suggestions`. Run weekly.

Source is controlled by `LAYER3_SOURCE` (default `postgres`):
- Unset / `postgres` → reads real users from `layer2_examples` in Postgres
- `minio` → reads synthetic users from `user_store.pkl` in MinIO (local testing only)

**Production:**

```bash
docker run --rm --network host \
  -v "$(pwd)/model_pipeline/layer2/config.yaml:/app/model_pipeline/layer2/config.yaml" \
  -e ANTHROPIC_API_KEY=<your-key> \
  -e POSTGRES_DSN="postgresql://mlops_user:mlops_pass@10.43.98.71:5432/mlops" \
  actualbudget-evaluate \
  python -m model_pipeline.layer3.pipeline
```

**Local testing (MinIO):**

```bash
docker run --rm --network host \
  -v "$(pwd)/model_pipeline/layer2/config.yaml:/app/model_pipeline/layer2/config.yaml" \
  -e LAYER3_SOURCE=minio \
  -e MINIO_ENDPOINT_URL=http://129.114.25.143:30900 \
  -e MINIO_ACCESS_KEY=minioadmin \
  -e MINIO_SECRET_KEY=minioadmin123 \
  -e ANTHROPIC_API_KEY=<your-key> \
  -e POSTGRES_DSN="postgresql://mlops_user:mlops_pass@10.43.98.71:5432/mlops" \
  actualbudget-evaluate \
  python -m model_pipeline.layer3.pipeline
```

MLflow experiment: `layer3-clustering`
Metrics logged: `total_users_processed`, `total_clusters_found`, `total_suggestions_written`

---

## MLflow Experiments Summary

| Experiment | Script | Key metrics |
|---|---|---|
| `layer2-evaluation` | `model_pipeline.evaluate` (all eval CSVs, tagged by `eval_split`) | weighted_f1, macro_f1, layer2_routing_pct |
| `layer1-layer2-evaluation` | `evaluate_moneydata_sliding` | per-year weighted_f1, layer2_routing_pct (step = year) |
| `layer3-evaluation` | `model_pipeline.layer3.evaluate` | silhouette, coverage, naming_accuracy |
| `layer3-clustering` | `model_pipeline.layer3.pipeline` | total_suggestions_written |

MLflow UI: `http://129.114.25.143:30500`

---

## Postgres — layer3_suggestions Table

```sql
CREATE TABLE IF NOT EXISTS layer3_suggestions (
    id                      SERIAL PRIMARY KEY,
    user_id                 TEXT NOT NULL,
    cluster_id              TEXT NOT NULL UNIQUE,
    suggested_category_name TEXT NOT NULL,
    payee_list              TEXT[] NOT NULL,
    status                  TEXT NOT NULL DEFAULT 'pending'
                            CHECK (status IN ('pending', 'approved', 'rejected')),
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

Migration already applied in production (`mlops.public.layer3_suggestions`). Do not re-run `001_create_layer3_suggestions.sql`.

Verify:

```bash
psql "postgresql://mlops_user:mlops_pass@10.43.98.71:5432/mlops" \
  -c "\d public.layer3_suggestions"
```
