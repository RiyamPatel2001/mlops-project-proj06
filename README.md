# Serving project — step-by-step runbook

Work from your **Chameleon GPU node** (or any machine with Docker + NVIDIA drivers where the project is cloned). Replace paths and URLs with yours where noted.

---

## 0. Prerequisites

- SSH access to the reserved node (`ssh -i ~/.ssh/id_rsa_chameleon cc@<FLOATING_IP>`).
- This repository cloned on the node, e.g. `~/serving-project`.
- GPU drivers + NVIDIA Container Toolkit (your `chameleon/3_setup_server.ipynb` flow).

All `docker compose` commands below assume:

```bash
cd ~/serving-project   # repository root containing docker/, evaluation/, models/, etc.
```

---

## Fast path: Phase 3-4 CSVs done; only restore models

Use this if your **`evaluation/results/*_evaluation.csv`**, **`fastapi_benchmark.csv`**, and any copies under **`Evaluation_Results/`** or **`results/`** are already saved and you **only** lost the **`models/`** directory (or never had it on this machine).

**You do not need to repeat Phases 3 or 4** for scoring purposes if those CSVs are intact.

### What to run now

1. SSH to your Chameleon node (or use any machine that can reach MLflow).

2. Go to the repo root and download with a **virtual environment** (recommended on Ubuntu 24.04+; system Python is PEP 668 “externally managed” and blocks `pip install --user`).

```bash
cd ~/serving-project   # or ~/mlops-project-proj06

sudo apt-get update
sudo apt-get install -y python3-pip python3-venv

python3 -m venv .venv
source .venv/bin/activate
pip install mlflow
python scripts/download_models.py \
  --tracking-uri http://129.114.26.151:8000 \
  --output-dir ./models
deactivate   # optional
```

Later SSH sessions: `cd` to the repo, then `source .venv/bin/activate` before any `python` that needs `mlflow`.

**`No module named pip`:** Run `sudo apt-get install -y python3-pip` first, then use the **venv** commands above (not `pip install --user`).

**`externally-managed-environment`:** Do not use `--break-system-packages`. Use the **venv** block above.

**`docker: command not found`:** Your node may not have Docker installed yet (install it when you do Phase 3 setup, or `sudo apt-get install -y docker.io`). Until then, use the **venv** method above instead of a `docker run ...` downloader.

**Optional — Docker one-liner** (only if Docker works and you prefer not to use venv):

```bash
docker run --rm -v "$PWD:/work" -w /work python:3.10-slim bash -ce \
  "pip install -q mlflow && python scripts/download_models.py \
  --tracking-uri http://129.114.26.151:8000 --output-dir ./models"
```

3. Optional but useful for some containers / calibration paths:

```bash
cp data/transactions.csv models/
```

4. **Next work depends on your goal:**
   - **FastAPI only (Phase 4 again later):** With `models/` back, `docker/docker-compose-fastapi.yaml` can serve again using the env vars you used before. You still **do not** need to re-run `benchmark_fastapi.ipynb` if you kept the CSV.
   - **Phase 5 (Triton) with ONNX:** Triton expects files under `models/<model>_onnx/` (e.g. `model.onnx`). MLflow download does **not** recreate those. If those folders are gone, either restore them from backup or run **only the ONNX export / relevant sections** of `evaluation/eval_distilbert.ipynb`, `eval_minilm.ipynb`, and `eval_tfidf_logreg.ipynb` once inside the **eval** Jupyter container (`docker-compose-eval.yaml`), then copy into `triton/models/...` as in Phase 5 below.
   - **Phase 5 Python backends:** Copy MLflow artifacts into `triton/models/*/1/model_files/` or FastText binaries into `triton/models/fasttext_classifier/1/` per your Triton layout.

**Chameleon lease / server setup:** Skip [section 1](#1-chameleon-setup-notebooks-in-chameleon) if your node is already running and you can SSH in.

---

## 1. Chameleon setup (notebooks in `chameleon/`)

Run in the **Chameleon Jupyter** UI at [jupyter.chameleoncloud.org](https://jupyter.chameleoncloud.org), in order:

1. `1_create_lease.ipynb` — reserve GPU node (e.g. `gpu_rtx_6000`, lease name per assignment).
2. `2_create_server.ipynb` — bare-metal instance + floating IP.
3. `3_setup_server.ipynb` — Docker + NVIDIA toolkit on the node.

Then SSH to the node and continue below.

---

## 2. Clone repo and download MLflow models

On the Chameleon node:

```bash
git clone <YOUR_REPO_URL> ~/serving-project
cd ~/serving-project

sudo apt-get update && sudo apt-get install -y python3-pip python3-venv
python3 -m venv .venv && source .venv/bin/activate
pip install mlflow
python scripts/download_models.py \
  --tracking-uri http://129.114.26.151:8000 \
  --output-dir ./models
```

This restores `models/distilbert/`, `models/minilm/`, `models/fasttext/`, `models/tfidf_logreg/`. If the URI or experiment ID differs, use `--tracking-uri` and `--experiment-id` after checking MLflow.

On Ubuntu 24.04+, use a **venv** as shown (not `pip install --user`). If **`No module named pip`**, **`externally-managed-environment`**, or **Docker missing**, see [Fast path](#fast-path-phase-3-4-csvs-done-only-restore-models).

Optional (calibration / notebooks that read raw transactions):

```bash
cp data/transactions.csv models/
```

---

## 3. Phase 3 — model evaluation notebooks (CPU / GPU / OpenVINO)

Skip this section if you already kept the Phase 3 result CSVs and are only restoring `models/` — see [Fast path](#fast-path-phase-3-4-csvs-done-only-restore-models).

Use **`docker/docker-compose-eval.yaml`**. It defines three services; start only what you need.

### CPU evaluations (strategies 1–6, CPU containers)

```bash
docker compose -f docker/docker-compose-eval.yaml up jupyter -d
docker exec jupyter_eval jupyter server list
```

Open the printed Jupyter URL (use an SSH tunnel to the node if needed). Working directory in the container is typically `/home/jovyan/work`. Run:

- `evaluation/eval_distilbert.ipynb`
- `evaluation/eval_minilm.ipynb`
- `evaluation/eval_tfidf_logreg.ipynb`
- `evaluation/eval_fasttext.ipynb`

These write under `evaluation/results/` (mounted from the host).

Stop when done:

```bash
docker compose -f docker/docker-compose-eval.yaml stop jupyter
```

### GPU sections (strategy 7 — ONNX CUDA / TensorRT)

```bash
docker compose -f docker/docker-compose-eval.yaml up jupyter_gpu -d
docker exec jupyter_eval_gpu jupyter server list
```

Re-run the **section 7** cells in each **transformer** notebook (`eval_distilbert.ipynb`, `eval_minilm.ipynb`). Port **8889** maps to this Jupyter (see compose file).

### OpenVINO (strategy 8)

```bash
docker compose -f docker/docker-compose-eval.yaml up jupyter_openvino -d
docker exec jupyter_eval_openvino jupyter server list
```

Re-run **section 8** in the notebooks that support OpenVINO. Port **8890** maps to this Jupyter.

**Which compose file?** For Phase 3, always **`docker-compose-eval.yaml`** (`jupyter`, `jupyter_gpu`, `jupyter_openvino`).

---

## 4. Phase 4 — FastAPI benchmarks

Skip this section if you already have `fastapi_benchmark.csv` (or equivalent) and only needed to repopulate `models/` — see [Fast path](#fast-path-phase-3-4-csvs-done-only-restore-models).

Use **`docker/docker-compose-fastapi.yaml`**.

1. Edit **`docker/docker-compose-fastapi.yaml`** — set `MODEL_TYPE`, `MODEL_BACKEND`, `MODEL_PATH`, and related env vars for the model you want to serve (see `fastapi_serving/app.py`).

2. **Stop anything else bound to host port 8000** (e.g. Triton from Phase 5).

3. Start FastAPI + Jupyter for the benchmark notebook:

```bash
docker compose -f docker/docker-compose-fastapi.yaml up -d
curl -s http://localhost:8000/health
```

4. Get Jupyter URL and open `evaluation/benchmark_fastapi.ipynb`; run sequential and concurrent benchmarks. Results go to `evaluation/results/fastapi_benchmark.csv` (or paths printed in the notebook).

Stop:

```bash
docker compose -f docker/docker-compose-fastapi.yaml down
```

---

## 5. Phase 5 — Triton + `perf_analyzer`

Use **`docker/docker-compose-triton.yaml`**.

1. **Populate** `triton/models/...` with artifacts (ONNX files, Python backend weights, etc.) — see project notes for copying from `models/` and `models/*_onnx/`.

2. **Do not** run FastAPI and Triton at the same time on the same host if both map **8000** — stop FastAPI first.

3. Start Triton and the Triton SDK Jupyter (the Jupyter service **builds** a thin image on top of `tritonserver:*-py3-sdk` so `jupyter` / `python3 -m jupyter notebook` exists — upstream SDK images may ship without a `jupyter` binary):

```bash
docker compose -f docker/docker-compose-triton.yaml build triton_server jupyter
docker compose -f docker/docker-compose-triton.yaml up -d
docker logs triton_server -f
```

**TF-IDF ONNX + locale:** If logs show `Failed to construct locale with name:en_US.UTF-8` for `tfidf_logreg_classifier`, the custom **`Dockerfile.triton-server`** (installed `locales` + `en_US.UTF-8`) fixes it—run **`build triton_server`** after pulling.

**Skip broken TF-IDF at startup:** Compose runs Triton in **explicit model-control mode** and only loads **`distilbert_classifier`** + **`minilm_classifier`**. This prevents the recurring `failed to load all models` startup error caused by TF‑IDF ZipMap output type mismatch.

**TF-IDF ONNX + SEQUENCE:** If logs show `ONNX_TYPE_SEQUENCE` / `output_probability`, the ONNX was exported with ZipMap (default skl2onnx). Re-export with the updated **`scripts/tfidf_onnx_helpers.py`** (`zipmap=False`), then copy `models/tfidf_logreg_onnx/model.onnx` to **`triton/models/tfidf_logreg_classifier/1/`** and restart Triton. **`triton/models/tfidf_logreg_classifier/config.pbtxt`** must list tensor outputs **`output_label`** (INT64) and **`output_probability`** (FP32).

**GPU:** If `triton_server` fails with `could not select device driver "nvidia"`, install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on the host, run `sudo nvidia-ctk runtime configure --runtime=docker`, restart Docker, then `up -d` again.

**`jupyter_triton` exits with `jupyter: not found`:** Pull the latest `docker/Dockerfile.triton-jupyter` + `docker-compose-triton.yaml`, run `docker compose ... build jupyter`, then `up -d` again.

4. Open notebooks under **`/workspace/evaluation`** in Jupyter. Run `evaluation/benchmark_triton.ipynb` **inside** `jupyter_triton` (uses hostname `triton_server` on the Docker network). Token: `docker exec jupyter_triton jupyter server list`.

Stop:

```bash
docker compose -f docker/docker-compose-triton.yaml down
```

---

## 6. Phase 6 — system optimizations

Run **`evaluation/system_optimizations.ipynb`** on a machine that can edit `triton/models/**/config.pbtxt` and restart Triton (same host as Phase 5). Use `nvtop` / `nvidia-smi` in another SSH session while testing.

---

## 7. Phase 7 — merge results and fill the table

```bash
python3 scripts/compile_results.py
```

Then update **`results/serving_options_table.csv`** with endpoint URL, MLflow run IDs, git SHA, and mark best options. Final merged output: `results/serving_options_table_final.csv`.

---

## Quick reference: which `docker compose` file?

| Phase | Compose file | Main services |
|--------|----------------|----------------|
| 3 — Eval notebooks | `docker/docker-compose-eval.yaml` | `jupyter`, `jupyter_gpu`, `jupyter_openvino` |
| 4 — FastAPI + bench | `docker/docker-compose-fastapi.yaml` | `fastapi_server`, `jupyter` |
| 5 — Triton | `docker/docker-compose-triton.yaml` | `triton_server`, `jupyter` (Triton SDK) |

Always run `docker compose` from the **repository root** (parent of `docker/`).

---

## TF-IDF ONNX note

`evaluation/eval_tfidf_logreg.ipynb` exports **`models/tfidf_logreg_onnx/model.onnx`**. If the MLflow pipeline uses a non-convertible vectorizer (e.g. character n-grams), the notebook trains a **word-level surrogate** on **`transactions.csv`** and exports that to ONNX so the file exists for Triton and ORT. Section 1 (sklearn) still reflects the **MLflow** model; ONNX rows use the exported graph (read the messages printed in the notebook).

**Where to put `transactions.csv` on the server:** The name is **gitignored**, so `git clone` often does not create it. The helper looks in order: **`data/transactions.csv`**, **`models/transactions.csv`**, then **`transactions.csv`** at the repo root. Copy it from your laptop (or course files) into one of those paths on the Chameleon node **before** starting Jupyter Docker, or the surrogate ONNX step will fail with “Missing … transactions.csv”.

The notebook loads **`scripts/tfidf_onnx_helpers.py` by path**; that file must exist in your checkout (run `git pull` or copy it onto the Chameleon node if you only updated the `.ipynb`).

**If ONNX is still skipped in Jupyter**, run the same surrogate export from the repo root (needs `pandas`, `scikit-learn`, `skl2onnx`, `onnx` — e.g. inside your project venv or the eval Docker container):

```bash
python3 scripts/export_tfidf_onnx_surrogate.py
```

The surrogate uses **integer-encoded class labels** so `skl2onnx` conversion succeeds (string labels often make conversion fail silently in notebooks).
