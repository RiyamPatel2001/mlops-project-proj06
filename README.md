# ActualBudget Transaction Categorizer — proj06

An end-to-end MLOps system that automatically categorizes personal finance transactions inside [ActualBudget](https://actualbudget.com). Built on a three-layer ML pipeline: a population-level classifier, a per-user personalization layer, and a custom-category discovery layer.

## Team

| Member | Responsibility |
|---|---|
| Saketh | Data pipeline |
| Riyam | Model training + evaluation |
| Jayraj | Serving |
| Puneeth | DevOps / Infrastructure |

---

## System Overview

```
Raw transaction (payee string, amount, date)
        │
        ▼
┌───────────────────────────────────────────────┐
│  Layer 1 — Population Classifier              │
│  fastText model trained on 2022 CEX data      │
│  Predicts one of 29 spending categories       │
│  Input: payee string (normalized)             │
└───────────────────────┬───────────────────────┘
                        │  always runs
                        ▼
┌───────────────────────────────────────────────┐
│  Layer 2 — Personalization                    │
│  all-mpnet-base-v2 semantic similarity        │
│  Overrides Layer 1 when the user's own        │
│  transaction history gives a better signal    │
│  Falls back to Layer 1 for cold-start users   │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
              Final prediction
                        │
                        ▼
┌───────────────────────────────────────────────┐
│  Layer 3 — Custom Category Discovery          │
│  DBSCAN clustering on per-user embeddings     │
│  LLM (Claude) names each cluster              │
│  Writes suggested custom categories to        │
│  Postgres for user review                     │
└───────────────────────────────────────────────┘
```

---

## Quick Reference — Docker Commands

All commands run from the **project root**. The cluster IP is `129.114.25.143`.

### Build the evaluation image (once, or after requirements change)

```bash
docker build -f model_pipeline/layer2/Dockerfile -t actualbudget-evaluate .
```

---

### Layer 1 — Training

```bash
# Build
docker build -f training/Dockerfile -t categorizer-training training/

# Train (CPU)
docker run --rm --network host \
  -v "$(pwd)/training/config.yaml:/app/training/config.yaml" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/training/models/layer1/artifacts:/app/training/models/layer1/artifacts" \
  -e MLFLOW_TRACKING_URI=http://129.114.25.143:30500 \
  -e MINIO_ACCESS_KEY=minioadmin \
  -e MINIO_SECRET_KEY=minioadmin123 \
  -e GIT_SHA="$(git rev-parse HEAD)" \
  categorizer-training
```

---

### Layer 1 — Evaluation

```bash
# Evaluate on CEX (in-distribution)
docker run --rm --network host \
  -v "$(pwd)/training/config.yaml:/app/training/config.yaml" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/training/models/layer1/artifacts:/app/training/models/layer1/artifacts" \
  -e MLFLOW_TRACKING_URI=http://129.114.25.143:30500 \
  -e MINIO_ENDPOINT_URL=http://129.114.25.143:30900 \
  -e MINIO_ACCESS_KEY=minioadmin \
  -e MINIO_SECRET_KEY=minioadmin123 \
  --entrypoint python \
  categorizer-training \
  /app/training/eval_layer1.py \
    --run-id <mlflow-run-id> \
    --model-type fasttext \
    --eval-csv processed/eval_cex.csv \
    --run-name eval-fasttext-cex

# Evaluate on MoneyData (OOD — disable quality gate)
docker run --rm --network host \
  -v "$(pwd)/training/config.yaml:/app/training/config.yaml" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/training/models/layer1/artifacts:/app/training/models/layer1/artifacts" \
  -e MLFLOW_TRACKING_URI=http://129.114.25.143:30500 \
  -e MINIO_ENDPOINT_URL=http://129.114.25.143:30900 \
  -e MINIO_ACCESS_KEY=minioadmin \
  -e MINIO_SECRET_KEY=minioadmin123 \
  --entrypoint python \
  categorizer-training \
  /app/training/eval_layer1.py \
    --run-id <mlflow-run-id> \
    --model-type fasttext \
    --eval-csv processed/eval_moneydata.csv \
    --run-name eval-fasttext-moneydata \
    --no-quality-gate
```

---

### Layer 1 + Layer 2 — CEX Evaluation

> **Prerequisite:** pull `user_store.pkl` from the PVC if it already exists, or run Step 1 to build it fresh.
>
> ```bash
> kubectl cp -n mlops <serving-pod-name>:/app/artifacts/user_store.pkl ./artifacts/user_store.pkl
> ```

**Step 1 — Build user store** from `train.csv` (2022 CEX):

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

**Step 2 — Evaluate Layer 1+2** on `eval_cex.csv` (2024 CEX):

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

---

### Layer 1 + Layer 2 — MoneyData Sliding Window

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

### Layer 3 — Cluster Quality Evaluation

> **Prerequisite:** `user_store.pkl` must exist at `./artifacts/user_store.pkl`.

```bash
docker run --rm --network host \
  -v "$(pwd)/model_pipeline/layer2/config.yaml:/app/model_pipeline/layer2/config.yaml" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -e ANTHROPIC_API_KEY=<your-key> \
  actualbudget-evaluate \
  python -m model_pipeline.layer3.evaluate
```

---

### Layer 3 — Pipeline (writes suggestions to Postgres)

> **Prerequisite:** `user_store.pkl` must exist at `./artifacts/user_store.pkl`.

```bash
docker run --rm --network host \
  -v "$(pwd)/model_pipeline/layer2/config.yaml:/app/model_pipeline/layer2/config.yaml" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -e ANTHROPIC_API_KEY=<your-key> \
  -e POSTGRES_DSN="postgresql://mlops_user:mlops_pass@10.43.98.71:5432/mlops" \
  actualbudget-evaluate \
  python -m model_pipeline.layer3.pipeline
```

---

## Layer 1 — Population Classifier

**What it does:** Classifies any transaction into one of 29 standard spending categories using only the payee string. Trained on 2022 Consumer Expenditure Survey (CEX) data.

**Model:** fastText (n-gram classifier). Also supports tfidf_logreg, MiniLM, DistilBERT, mpnet — fastText is used in production for its speed.

**Input:** Normalized payee string (e.g. `"whole foods mkt"` → category `"Groceries"`).

**Output:** Category string + confidence score.

**Training:** `training/train.py` — reads raw CSV from MinIO, splits by user, trains, evaluates against quality gate (weighted F1 ≥ 0.90), and promotes to the serving registry if it beats the current production model by ≥ 1%.

**Evaluation datasets:**
- `data/processed/eval_cex.csv` — 2024 CEX data (in-distribution)
- `data/processed/eval_moneydata.csv` — UK MoneyData (out-of-distribution)

See [`training/README.md`](training/README.md) for Docker build/run commands.

---

## Layer 2 — Personalization

**What it does:** Overrides the Layer 1 prediction for users who have enough transaction history. Embeds the incoming payee with `all-mpnet-base-v2`, finds the top-k most similar payees in the user's store, and uses majority vote to decide whether the user's own history gives a more reliable category.

**When Layer 2 activates:** User has ≥ `min_history` (default: 10) stored transactions AND max cosine similarity to stored payees ≥ `similarity_threshold` (default: 0.85). Otherwise falls back to Layer 1.

**User store:** A per-user dictionary of `{embeddings, labels, payees}` built offline from historical data and persisted as `user_store.pkl` on a PVC (`/app/artifacts`). Shared between the serving pods and the Layer 3 pipeline.

**Cold-start:** New users automatically accumulate history on every prediction. Layer 2 activates once the threshold is crossed — no manual intervention needed.

**Evaluation:**
- **CEX** (`model_pipeline/evaluate.py`): builds store from `train.csv` (2022), evaluates Layer 1+2 on `eval_cex.csv` (2024). Logs weighted F1, macro F1, Layer 2 routing %.
- **MoneyData sliding window** (`evaluate_moneydata_sliding.py`): iterates eval years 2015–2022, each time using all prior years as the bootstrap store. Shows how F1 and routing % improve as history accumulates.

See [`model_pipeline/layer2/README.md`](model_pipeline/layer2/README.md) for Docker build/run commands.

---

## Layer 3 — Custom Category Discovery

**What it does:** Runs weekly offline. Clusters each user's transaction embeddings with DBSCAN to find groups of semantically similar payees that may belong to a user-defined category (e.g. a user who frequently visits the same local gym not covered by the 29 standard categories). Each cluster is named by Claude (Anthropic API) and written as a pending suggestion to Postgres. The user can approve or reject suggestions through ActualBudget.

**Pipeline (`pipeline.py`):** Loads `user_store.pkl` → DBSCAN per user → LLM naming → INSERT into `layer3_suggestions` table with `ON CONFLICT (cluster_id) DO NOTHING`.

**Evaluation (`evaluate.py`):** Measures cluster quality (silhouette, coverage, noise %) and naming accuracy (LLM suggestion vs. majority ground-truth label on pure clusters). Also sweeps eps at tight/default/loose to aid hyperparameter selection.

**Postgres table:** `public.layer3_suggestions` — columns: `user_id`, `cluster_id` (unique), `suggested_category_name`, `payee_list` (TEXT[]), `status` (pending/approved/rejected), `created_at`.

**LLM:** `namer.py` calls `claude-sonnet-4-20250514` via the Anthropic API. Requires `ANTHROPIC_API_KEY`. Falls back to majority label on API failure.

See [`model_pipeline/layer2/README.md`](model_pipeline/layer2/README.md) for Docker build/run commands covering Layer 3 evaluation and pipeline.

---

## Data Pipeline

Raw data flows from CEX PUMD source files through ingestion, preprocessing, and feature computation before landing in MinIO. Processed CSVs are consumed by training and evaluation.

```
CEX PUMD source files
        │
        ▼
data_pipeline/ingestion/     →  data/raw/
data_pipeline/preprocessing/ →  data/processed/
data_pipeline/feature_computation/
data_pipeline/batch_pipeline/
data_pipeline/drift_detection/
```

**Datasets:**
| File | Years | Rows | Purpose |
|---|---|---|---|
| `data/processed/train.csv` | 2022 | ~48K | Layer 1 training + Layer 2 store build |
| `data/processed/eval_cex.csv` | 2024 | ~63K | Layer 1+2 evaluation (in-distribution) |
| `data/processed/eval_moneydata.csv` | 2015–2022 | ~6K | Layer 1+2 sliding window evaluation (OOD, UK) |
| `data/processed/production.csv` | 2023 | — | Production simulation seed |

**Reproducing the data:**
1. Download CEX PUMD CSV files from https://www.bls.gov/cex/pumd_data.htm
2. Extract the FMLI files for each year
3. Run:
```bash
python generate_transactions.py --year 2022 \
  --input_files fmli222.csv fmli223.csv fmli224.csv fmli231.csv \
  --output transactions_2022.csv
```

---

## Serving

FastAPI app exposing `/classify`, `/feedback`, and `/metrics`. Loads Layer 1 (fastText) and Layer 2 (Predictor) at startup. Routes each request through both layers and returns the final prediction with source (`layer1` or `layer2`) and confidence.

The serving app shares the `artifacts-pvc` with the Layer 3 pipeline — both mount `/app/artifacts` where `user_store.pkl` lives.

---

## Monitoring and Triggers

Prometheus scrapes `/metrics` on the serving app. Key signals:

| Metric | What it tracks |
|---|---|
| `serving_prediction_outputs_total` | Prediction volume by category |
| `serving_prediction_confidence` | Confidence score histogram |
| `serving_feedback_total` | User correction rate |
| `serving_suggestion_responses_total` | Layer 3 suggestion accept/reject rate |

**Promotion:** Conservative — requires ≥ 1 percentage point offline accuracy improvement before updating the production registry.

**Rollback triggers:**
- User correction rate > 25% over 2 hours
- Low-confidence ratio > 35% over 30 minutes
- Classify error rate > 5% over 10 minutes

---

## Infrastructure

Deployed on Chameleon Cloud. k3s cluster with ArgoCD for GitOps. All platform services (MLflow, MinIO, Postgres, Prometheus, Grafana) run inside the cluster.

| Service | Internal DNS | External (NodePort) |
|---|---|---|
| MLflow | `mlflow.mlops.svc.cluster.local` | `129.114.25.143:30500` |
| MinIO | `minio.mlops.svc.cluster.local:9000` | `129.114.25.143:30900` |
| Postgres | `postgres.mlops.svc.cluster.local:5432` | ClusterIP only — use `10.43.98.71:5432` from host |
| Adminer | — | `129.114.25.143:30081` |

**Adminer access:**
- System: `PostgreSQL`
- Server: `postgres`
- Username / password / database: use values from the `postgres-credentials` k8s secret

See [`devops/README.md`](devops/README.md) for cluster setup, Terraform, Ansible, and k3s bootstrap.

---

## Component READMEs

| README | What it covers |
|---|---|
| [`training/README.md`](training/README.md) | Layer 1 training — models, Docker commands for CPU/GPU/sweep/retrain, quality gate, MLflow logging |
| [`model_pipeline/layer2/README.md`](model_pipeline/layer2/README.md) | Layer 2 + Layer 3 — user store, all evaluation Docker commands, config reference |
| [`devops/README.md`](devops/README.md) | Cluster setup — Terraform, Ansible, k3s bootstrap, platform services |
| [`serving/serving_initial/README.md`](serving/serving_initial/README.md) | Serving evaluation — latency/throughput benchmarks, FastText + FastAPI selection rationale |
