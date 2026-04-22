# proj06 — Infrastructure & DevOps Guide

**Owner:** Puneeth (DevOps/Platform)  
**Last updated:** April 22, 2026  
**Cluster:** KVM@TACC — `129.114.25.143`  
**GPU VM:** CHI@UC — `192.5.86.191`

---

## 1. Architecture Overview

```
GitHub (refactor-project branch)
        │
        ├── push → GitHub Actions → builds Docker images → ghcr.io
        │                                    │
        │                          ArgoCD Image Updater
        │                          pins SHA into manifests
        │                                    │
        └──────────────────────────► ArgoCD syncs → k3s cluster
                                              │
                              ┌───────────────┼───────────────┐
                              │               │               │
                         MLflow :30500   MinIO :30901   Postgres
                              │               │               │
                         ActualBudget    Grafana :30030  Prometheus
                              │               │               │
                         Classifier     Alertmanager    Rollback
                           :30508          :30093        Receiver
```

---

## 2. All Service URLs

| Service | URL | Credentials |
|---|---|---|
| **MLflow** | `http://129.114.25.143:30500` | none |
| **MinIO Console** | `http://129.114.25.143:30901` | `minioadmin / minioadmin123` |
| **MinIO API** | `http://129.114.25.143:30900` | `minioadmin / minioadmin123` |
| **ActualBudget** | `http://129.114.25.143:30506` | `admin@admin.com / mlops1234` |
| **Adminer (DB UI)** | `http://129.114.25.143:30081` | `mlops_user / mlops_pass / mlops` |
| **Grafana** | `http://129.114.25.143:30030` | `admin / mlops1234` |
| **Prometheus** | `http://129.114.25.143:30090` | none |
| **ArgoCD** | `http://129.114.25.143:30808` | `admin / kubectl get secret argocd-initial-admin-secret -n argocd -o jsonpath='{.data.password}' \| base64 -d` |
| **Classifier API** | `http://129.114.25.143:30508` | none |
| **Alertmanager** | `http://129.114.25.143:30093` | none |

---

## 3. SSH Access

### Production cluster (KVM@TACC)
```bash
ssh -i ~/.ssh/id_rsa_chameleon cc@129.114.25.143
```

### GPU VM (CHI@UC) — Riyam's training
```bash
ssh cc@192.5.86.191
```
*(uses your own SSH key — added to authorized_keys)*

---

## 4. Kubernetes

### Namespace
All workloads run in the `mlops` namespace.

### Check everything is running
```bash
kubectl get pods -n mlops
kubectl get pods -n argocd
```

### Expected running pods
```
actualbudget
actualbudget-proxy
adminer
alertmanager
grafana
minio
mlflow-proj06
postgres
prometheus
rollback-receiver
transaction-classifier
```

### Useful commands
```bash
# Logs for a service
kubectl logs -n mlops -l app=transaction-classifier --tail=50

# Restart a deployment
kubectl rollout restart deployment/<name> -n mlops

# Manual rollback
kubectl rollout undo deployment/transaction-classifier -n mlops

# Check HPA (autoscaling)
kubectl get hpa -n mlops

# Check CronJobs
kubectl get cronjobs -n mlops
```

---

## 5. Secrets

All secrets are in the `mlops` namespace. Created via bootstrap, **not** via sealed secrets (sealed secrets were encrypted on old cluster — use plain kubectl secrets on this cluster).

| Secret | Keys | Used by |
|---|---|---|
| `minio-credentials` | `accesskey`, `secretkey` | MLflow, training, serving |
| `postgres-credentials` | `username`, `password`, `dbname` | Postgres, MLflow |
| `proj06-env` | All env vars combined | Serving, training, batch pipeline |

```bash
# View secret keys (not values)
kubectl get secret proj06-env -n mlops -o jsonpath='{.data}' | python3 -c "import sys,json; [print(k) for k in json.load(sys.stdin)]"
```

---

## 6. Storage

### Cluster node (KVM@TACC)
| Disk | Size | What's on it |
|---|---|---|
| `/dev/vda3` (root) | 40GB | OS, k3s, all PVC data (MinIO, MLflow, Postgres) |
| `/dev/vdb` (Cinder) | 150GB | Containerd images + backup mirror |

### PVCs
```bash
kubectl get pvc -n mlops
```
All PVCs use `local-path` provisioner — data lives at `/var/lib/rancher/k3s/storage/` on the node.

⚠️ **PVC data does NOT survive cluster termination.** That's why backups exist.

### MinIO buckets
- `mlflow-artifacts/` — MLflow model artifacts
- `data/` — training data, retraining datasets, Postgres dumps

---

## 7. CI/CD Pipeline

**Trigger:** Push to `refactor-project` branch

**Flow:**
```
git push → GitHub Actions builds images → pushes to ghcr.io
         → ArgoCD Image Updater detects new tag
         → commits SHA-pinned manifest to git
         → ArgoCD syncs → rolling deploy on cluster
```

**Images built:**
| Image | Built from |
|---|---|
| `transaction-classifier` | `serving/Dockerfile` |
| `training-cpu` | `training/Dockerfile` |
| `actual-custom` | `actual/sync-server.Dockerfile` |
| `batch-pipeline` | `data_pipeline/batch_pipeline/Dockerfile.batch` |
| `data-generator` | `data_pipeline/data_generator/Dockerfile.generator` |
| `data-ingestion` | `data_pipeline/ingestion/Dockerfile.ingest` |
| `layer2-store` | `model_pipeline/layer2/Dockerfile` |

---

## 8. Model Registry

The classifier loads models based on `devops/layer1_registry.json`:

```json
{
  "tiers": {
    "good":  {"run_id": "b8f1ad8433b7492e82726429df5b66a0", "model_name": "minilm"},
    "fast":  {"run_id": "5af29fbaa4b04abc9d21b18a19ccc736", "model_name": "fasttext"},
    "cheap": {"run_id": "bd19d31c1aa94500a0fa3a4f2cee4c94", "model_name": "tfidf_logreg"}
  }
}
```

**When Riyam trains new models:**
1. Update `devops/layer1_registry.json` with new MLflow run IDs
2. Push to `refactor-project`
3. CI/CD auto-deploys — no manual steps needed

**To manually write registry to serving PVC:**
```bash
kubectl cp devops/layer1_registry.json mlops/<classifier-pod>:/app/models/layer1_registry.json
```

---

## 9. Scheduled Jobs

| CronJob | Schedule | What it does |
|---|---|---|
| `retraining-trigger` | Sundays 4am UTC | Retrains models, updates registry if quality gate passes |
| `layer3-pipeline` | Sundays 2am UTC | Runs batch pipeline, produces versioned retraining dataset |
| `postgres-backup` | Every hour :00 | `pg_dump` → MinIO `data/backups/postgres/` |
| `minio-backup` | Every hour :30 | `mc mirror` MinIO → `/mnt/docker-storage/backup/` on vdb |

---

## 10. Monitoring & Alerting

**Prometheus** scrapes `/metrics` from the classifier every 30s.

**Alert rules** (auto-trigger rollback):
| Alert | Condition | Action |
|---|---|---|
| `ClassifierDown` | Pod unreachable 2min | Rollback |
| `ClassifierErrorRateHigh` | >5% errors over 10min | Rollback |
| `UserCorrectionRateHigh` | >25% corrections over 2h | Rollback |
| `PredictionLowConfidenceSpike` | >35% low confidence over 30min | Rollback |
| `ClassifierLatencyP95High` | p95 >1.5s over 10min | Investigate |
| `ServingDatabaseDisconnected` | DB unreachable 5min | Rollback |

**Auto-rollback flow:**
```
Prometheus fires alert → Alertmanager → rollback-receiver webhook
→ kubectl rollout undo deployment/transaction-classifier -n mlops
```

**Grafana dashboard:** `proj06 — MLOps Platform`  
Panels: Classifier status, error rate, p95 latency, confidence distribution, feedback quality.

---

## 11. Canary Deployments

When Riyam pushes a new model and you want to test with 10% traffic before full rollout:

```bash
# 1. Activate canary with new image
kubectl set image deployment/transaction-classifier-canary \
  fastapi=ghcr.io/riyampatel2001/mlops-project-proj06/transaction-classifier:<NEW_SHA> \
  -n mlops
kubectl scale deployment/transaction-classifier --replicas=9 -n mlops
kubectl scale deployment/transaction-classifier-canary --replicas=1 -n mlops

# 2a. Promote if metrics look good (after 30min observation)
kubectl set image deployment/transaction-classifier \
  fastapi=ghcr.io/riyampatel2001/mlops-project-proj06/transaction-classifier:<NEW_SHA> \
  -n mlops
kubectl scale deployment/transaction-classifier --replicas=1 -n mlops
kubectl scale deployment/transaction-classifier-canary --replicas=0 -n mlops

# 2b. Rollback if metrics bad
kubectl scale deployment/transaction-classifier-canary --replicas=0 -n mlops
kubectl scale deployment/transaction-classifier --replicas=1 -n mlops
```

---

## 12. Backup & Recovery

### Current backups
- **Hourly:** Postgres dump → MinIO, MinIO mirror → vdb
- **Manual snapshot:** `proj06-vdb-snapshot-1` (150GiB) taken April 22, 2026

### Full recovery from scratch
```bash
# 1. Provision VM (Terraform)
cd devops/iac/tf
terraform init && terraform apply

# 2. Get floating IP from output, then bootstrap
scp -i ~/.ssh/id_rsa_chameleon devops/iac/bootstrap_k3s.sh cc@<IP>:~
ssh -i ~/.ssh/id_rsa_chameleon cc@<IP> 'bash bootstrap_k3s.sh'

# 3. Restore Postgres from backup
kubectl exec -n mlops -it <postgres-pod> -- psql -U mlops_user mlops < backup.sql

# 4. All done — ArgoCD syncs everything else automatically
```

### Restore from vdb snapshot
1. Go to KVM UI → Volumes → Volume Snapshots
2. Click `proj06-vdb-snapshot-1` → Create Volume
3. Attach new volume to new node as `/dev/vdb`
4. All MinIO mirrors and Postgres dumps are on it

---

## 13. GPU VM (Riyam's Training)

| Item | Value |
|---|---|
| IP | `192.5.86.191` |
| Site | CHI@UC |
| GPU | NVIDIA Quadro RTX 6000/8000 |
| RAM | 187GB |
| Disk | 210GB local NVMe |
| SSH | `ssh cc@192.5.86.191` |

**Pre-configured env vars** (in `~/.bashrc`):
```bash
MLFLOW_TRACKING_URI=http://129.114.25.143:30500
MLFLOW_S3_ENDPOINT_URL=http://129.114.25.143:30900
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin123
```

**Install NVIDIA drivers (first time only):**
```bash
sudo apt-get install -y nvidia-driver-535 && sudo reboot
```

---

## 14. Terraform (Reprovisioning)

```bash
cd devops/iac/tf
cp clouds.yaml ~/.config/openstack/clouds.yaml   # or set env vars
terraform init
terraform apply
```

**Required files:**
- `clouds.yaml` — OpenStack app credentials (keep secret, not in git)
- `terraform.tfvars` — suffix, reservation ID, key pair name

**Current reservation:** `e68b6d8e-13ee-4e5d-9611-0dafa43773de` (may expire — book new lease if needed)

---

## 15. Common Issues & Fixes

| Problem | Fix |
|---|---|
| Pod stuck `Pending` with nodeSelector error | `kubectl patch deployment <name> -n mlops --type=json -p='[{"op":"remove","path":"/spec/template/spec/nodeSelector"}]'` |
| ArgoCD blank page | Use `http://` not `https://` — browser auto-upgrades |
| Classifier in mock mode | Models not in MLflow yet — waiting on Riyam to train |
| Disk pressure on node | Check `df -h /` — if >85%, containerd images may have filled root disk |
| MinIO empty after rebuild | PVCs are local-path — recreate buckets: `mc mb minio/mlflow-artifacts && mc mb minio/data` |
| ArgoCD `ComparisonError` | Duplicate env var in a deployment YAML — check `kubectl describe app <name> -n argocd` |
