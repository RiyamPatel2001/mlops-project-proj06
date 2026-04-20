# ML Safeguarding Reference

All mechanisms that protect model quality, prevent data leakage, and ensure safe deployment across the Layer 1 training pipeline and Layer 2 inference pipeline.

---

## 1. Quality Gates

**File:** `training/evaluate.py:116–130`

The most critical safeguard. Both thresholds must pass for a training or retraining run to be considered successful.

| Metric | Default Threshold | Configurable |
|---|---|---|
| `weighted_f1` | >= 0.75 | Yes — `config.yaml` under `quality_gate.weighted_f1` |
| `macro_f1` | >= 0.55 | Yes — `config.yaml` under `quality_gate.macro_f1` |

**On failure:** raises `SystemExit(1)` — the container exits with a non-zero code, blocking any downstream model promotion or CI step.

**On pass:** sets MLflow tag `quality_gate = "passed"`.

Two metrics are used intentionally:
- `weighted_f1` catches overall performance degradation (proportional to class frequency).
- `macro_f1` catches minority-class degradation — a high `weighted_f1` can mask a model that completely fails on rare transaction categories.

To override thresholds per run, add to `config.yaml`:
```yaml
quality_gate:
  weighted_f1: 0.80
  macro_f1: 0.60
```

---

## 2. Retraining Gate

**File:** `training/retrain.py:108–114, 144–148`

Retraining is only meaningful if there is new feedback data. Two conditions must hold before a run proceeds:

1. **Dataset discovery:** MinIO is scanned for `retraining/retraining_dataset_v*.csv`. If none exist, the script exits with code 0 (not a failure — expected on fresh deployments).
2. **Quality gate pass:** The same `evaluate_and_log()` quality gate from Section 1 is enforced. A retrained model that does not meet the thresholds never reaches the artifact store.

If the gate fails during retraining, the old model remains in production. The failed run is still logged to the `layer1-retraining` MLflow experiment for post-mortem analysis.

---

## 3. Data Isolation — User-Stratified Split

**File:** `training/utils.py:38–60`

No user appears in both the training and validation sets. This simulates new-user generalization and prevents the model from memorizing per-user patterns.

| Parameter | Value | Configured In |
|---|---|---|
| Validation fraction | 20% of users | `config.yaml: val_frac: 0.2` |
| Seed | 42 | `config.yaml: random_state: 42` |

The split is seeded — runs are fully reproducible given the same config.

---

## 4. Data Isolation — Temporal Split (Layer 2)

**File:** `model_pipeline/layer2/build_store.py:63–72`

For the Layer 2 personalized store, a strict chronological split is used per user:

- **First 70%** of each user's transactions (by date): used to build the store.
- **Last 30%**: held out for evaluation.

This prevents future data leakage into the store and accurately simulates the cold-start ramp-up scenario.

---

## 5. Payee Normalization

**File:** `training/utils.py:18–33`

Raw payee strings are normalized before training and before inference. The pipeline is deterministic and applied consistently:

1. Uppercase + strip whitespace
2. Remove store/branch numbers (`#\d+`)
3. Remove platform order suffixes (`*ORDER_ID`)
4. Remove trailing digits
5. Collapse repeated whitespace

This prevents the same merchant from appearing as hundreds of distinct tokens (e.g., `STARBUCKS #1234` and `STARBUCKS #9999` both become `STARBUCKS`), which would fragment training signal and inflate vocabulary size.

---

## 6. Per-Class F1 Logging

**File:** `training/evaluate.py:88–91`

In addition to the aggregate quality gate, individual F1 scores are logged for every category as separate MLflow metrics (`f1_{category_name}`). This makes minority-class degradation visible in the MLflow UI even when macro/weighted averages appear healthy.

Metric names are sanitized: spaces, slashes, and ampersands become underscores.

---

## 7. Unknown Category Filtering at Eval Time

**File:** `model_pipeline/evaluate.py:159–170`  
**File:** `training/temporal_experiment.py:92–94`

Evaluation datasets may contain categories that were not seen during training (e.g., a new transaction type introduced after the training cutoff). Rows with unknown categories are dropped before computing metrics, and the count is logged as `dropped_rows` / `realworld_dropped_rows` in MLflow. A high drop count signals distribution shift worth investigating.

---

## 8. Temporal Generalization Experiment

**File:** `training/temporal_experiment.py:172–257`

A dedicated experiment (`layer1-temporal-generalization`) evaluates whether the model trained on 2022 data generalizes to 2024 data. Metrics are logged separately for two eval sets:

| Metric | Description |
|---|---|
| `weighted_f1_synthetic` / `macro_f1_synthetic` | Performance on synthetic 2024 data |
| `weighted_f1_realworld` / `macro_f1_realworld` | Performance on real-world 2024 data |
| `shared_payees` | Payees present in both 2022 and 2024 |
| `new_payees` | Payees only in 2024 (unseen at training time) |

The artifact `payee_overlap.json` is saved per run for deeper drift analysis.

---

## 9. Layer 2 Cold-Start Protection

**File:** `model_pipeline/layer2/predictor.py:86–98`  
**Config:** `model_pipeline/layer2/config.yaml: min_history: 10`

Layer 2 uses a user's personal transaction history to make predictions. If a user has fewer than `min_history` (default: 10) stored transactions, Layer 2 is skipped and Layer 1 is used instead. Transactions continue to accumulate until the threshold is met.

---

## 10. Layer 2 Confidence Threshold

**Config:** `model_pipeline/layer2/config.yaml: similarity_threshold: 0.85`

Even when sufficient history exists, Layer 2 only overrides Layer 1 if the nearest-neighbor similarity score meets the threshold (0.85). Below this, the Layer 1 prediction is used. This prevents low-confidence Layer 2 guesses from degrading overall accuracy.

---

## 11. MLflow Server Reachability Check

**File:** `training/train.py:84–93`

Before starting a training run, the script pings the MLflow tracking URI with a 5-second timeout. If the server is unreachable, a `RuntimeError` is raised immediately. This fail-fast behavior prevents silent fallback to the local file store, which would make the run invisible to the experiment tracking dashboard.

---

## 12. Git SHA Traceability

**File:** `training/train.py:306`, `training/retrain.py:131`

Every MLflow run is tagged with the `git_sha` of the commit used to produce it. The SHA is captured from the live git repo or from the `GIT_SHA` environment variable (for Docker runs). This makes every artifact traceable back to an exact code state.

---

## 13. FastText Learning Rate Guard

**File:** `training/models/layer1/fasttext.py:112–118`

FastText is known to produce NaN loss when `lr >= 1.2`. A runtime check validates the configured learning rate before training begins and raises a `ValueError` with an explanation if violated.

---

## 14. Gradient Clipping (Transformers)

**File:** `training/models/layer1/transformer_base.py:241`

Transformer models use `torch.nn.utils.clip_grad_norm_(max_norm=1.0)` on every backward pass. This prevents gradient explosion during fine-tuning, which can otherwise silently corrupt model weights.

---

## 15. Batch Inference to Prevent OOM

**File:** `training/models/layer1/transformer_base.py:82–84`

Transformer inference over the validation set is chunked into batches of 64. This prevents out-of-memory errors on large datasets when running on GPU-constrained hardware.

---

## 16. Atomic User Store Writes

**File:** `model_pipeline/layer2/user_store.py:44–47`

The Layer 2 user store is written atomically:

1. Serialized to a `.tmp` file.
2. Renamed to the final path using `os.replace()`.

`os.replace()` is atomic on POSIX systems. This means a crash or interruption during a write cannot leave the store in a partially-written state.

---

## 17. Embedding Normalization Guard

**File:** `model_pipeline/layer2/embedder.py:33–37`

Before cosine similarity is computed, all embeddings are L2-normalized. Norms are clamped to a minimum of `1e-9` before division to prevent divide-by-zero on zero-length vectors (e.g., empty payee strings after normalization).

---

## 18. Zero-Division Handling in Classification Reports

**File:** `training/evaluate.py:73, 80`

All `classification_report()` calls pass `zero_division=0`. If a class receives zero predictions (e.g., a very rare category not present in the validation set), its precision, recall, and F1 are set to 0 rather than emitting a warning or raising an exception.

---

## 19. Class Weight Balancing

**File:** `training/models/layer1/tfidf_logreg.py:56`  
**Config:** `config.yaml: tfidf_logreg.class_weight: balanced`

The logistic regression model uses `class_weight="balanced"`, which adjusts the penalty for each class inversely proportional to its frequency. This prevents high-frequency transaction categories from dominating the learned decision boundary.

---

## 20. Temp File Cleanup on Retraining

**File:** `training/retrain.py:123–167`

The downloaded retraining dataset is stored in a system temp file. The entire training block is wrapped in `try/finally` to guarantee the temp file is deleted even if training fails or the quality gate raises `SystemExit`. This prevents incremental disk exhaustion on repeated retraining runs.

---

## 21. NumPy Version Compatibility Check

**File:** `training/models/layer1/fasttext.py:32–37`

FastText-wheel 0.9.2 is incompatible with NumPy 2.x. A runtime import check validates `numpy.__version__ < "2"` and raises an `ImportError` with a clear explanation if violated. The `requirements.txt` pins `numpy==1.26.*` to prevent accidental upgrades.

---

## 22. Sweep Failure Tracking

**File:** `training/sweep-cpu.py:104, 129`

The hyperparameter sweep script tracks failed individual runs and exits with code 1 if any run fails. It also supports a `--dry-run` flag to print all planned configurations before executing, allowing validation of the sweep grid without triggering actual training.

---

## Summary

| Safeguard | File | Failure Behavior |
|---|---|---|
| Quality gate (weighted F1 >= 0.75) | `evaluate.py:118` | `SystemExit(1)` |
| Quality gate (macro F1 >= 0.55) | `evaluate.py:119` | `SystemExit(1)` |
| No retraining without data | `retrain.py:108–114` | `sys.exit(0)` |
| User-stratified train/val split | `utils.py:38–60` | Prevents data leakage |
| Temporal split for Layer 2 store | `build_store.py:63–72` | Prevents future leakage |
| Payee normalization | `utils.py:18–33` | Silent (deterministic) |
| Per-class F1 logging | `evaluate.py:88–91` | Surfaces minority class failures |
| Unknown category filtering | `evaluate.py:159–170` | Drops + logs `dropped_rows` |
| Layer 2 cold-start guard | `predictor.py:86` | Falls back to Layer 1 |
| Layer 2 confidence threshold | `layer2/config.yaml` | Falls back to Layer 1 |
| MLflow server reachability | `train.py:84–93` | `RuntimeError` (fail fast) |
| Git SHA traceability | `train.py:306` | Logged to every MLflow run |
| FastText LR guard | `fasttext.py:112–118` | `ValueError` |
| Gradient clipping | `transformer_base.py:241` | Silent (clamps gradient) |
| Batch inference (OOM guard) | `transformer_base.py:82` | Prevents OOM |
| Atomic store writes | `user_store.py:44–47` | Prevents store corruption |
| Embedding normalization guard | `embedder.py:33–37` | Prevents divide-by-zero |
| Zero-division in reports | `evaluate.py:73,80` | Returns 0 for empty classes |
| Class weight balancing | `tfidf_logreg.py:56` | Config-driven |
| Temp file cleanup | `retrain.py:164–167` | Always runs (try/finally) |
| NumPy version check | `fasttext.py:32–37` | `ImportError` |
| Sweep failure tracking | `sweep-cpu.py:129` | `sys.exit(1)` if any fail |
