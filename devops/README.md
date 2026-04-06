# DevOps / Platform — MLOps proj06

**Owner:** Puneeth (pk3058@nyu.edu)
**Cluster:** `mlops-k8s-proj06` — 3-node Kubernetes (Kubespray) on Chameleon Cloud (KVM@TACC)
**Floating IP:** `129.114.24.242`

---

## Architecture Overview

```
Chameleon Cloud — KVM@TACC
├── node1-mlops-proj06  (m1.medium: 2 vCPU, 4 GB RAM)  ← floating IP 129.114.24.242
│   ├── K8s control-plane
│   ├── Pod: actualbudget          (NodePort 30506)
│   └── Pod: postgres              (ClusterIP 5432)
│
├── node2-mlops-proj06  (m1.medium: 2 vCPU, 4 GB RAM)
│   ├── K8s control-plane
│   └── Pod: mlflow-proj06         (NodePort 30500)
│
└── node3-mlops-proj06  (m1.medium: 2 vCPU, 4 GB RAM)
    └── Pod: minio                 (ClusterIP 9000/9001)

kube-system namespace:
  ├── sealed-secrets               (secrets management — bonus)
  ├── argocd + argocd-image-updater (GitOps — bonus)
  ├── calico                       (CNI)
  └── metrics-server, coredns, local-path-provisioner
```

**Service URLs:**
| Service | URL |
|---|---|
| ActualBudget | `http://129.114.24.242:30506` |
| MLflow UI | `http://129.114.24.242:30500` |
| Transaction Classifier (inference) | `http://129.114.24.242:30508/classify` |

**Team MLflow URI (Riyam sets this when running training container):**
```
MLFLOW_TRACKING_URI=http://129.114.24.242:30500
```

---

## Infrastructure Requirements Table

Evidence: `kubectl top nodes` and `kubectl top pods -n mlops` on live Chameleon cluster, April 6 2026.

**Node utilisation (3× m1.medium — 2 vCPU / 4 GB each):**
| Node | Role | CPU Used | CPU % | Memory Used | Memory % |
|------|------|----------|-------|-------------|---------|
| node1 | control-plane + worker | 266m | 14% | 1963Mi | 59% |
| node2 | control-plane + worker | 250m | 13% | 2354Mi | 71% |
| node3 | worker | 235m | 12% | 1687Mi | 47% |

**Pod resource sizing (`mlops` namespace):**
| Service | CPU Request | CPU Limit | Mem Request | Mem Limit | GPU | Actual CPU | Actual Mem | Right-sizing rationale |
|---------|-------------|-----------|-------------|-----------|-----|-----------|------------|----------------------|
| ActualBudget | 250m | 500m | 256Mi | 512Mi | None | 1m | 47Mi | Lightweight Node.js app; spikes only on budget file import |
| MLflow server | 500m | 1000m | 512Mi | 1Gi | None | 3m | 636Mi | psycopg2 + boto3 + Flask loaded in memory at startup |
| PostgreSQL | 250m | 500m | 256Mi | 512Mi | None | 12m | 102Mi | Single-writer MLflow metadata backend; low concurrency |
| MinIO | 250m | 500m | 512Mi | 1Gi | None | 1m | 78Mi | Object store idle; memory grows with concurrent artifact uploads |
| Transaction Classifier | 500m | 1000m | 512Mi | 1Gi | None | — | — | FastText CPU inference; sized for <200ms p99 latency target |
| Sealed Secrets ctrl | 50m | 200m | 64Mi | 256Mi | None | — | — | Lightweight controller in kube-system; event-driven only |
| Argo CD Image Updater | 100m | 200m | 64Mi | 128Mi | None | — | — | Polls ghcr.io every 2 min; negligible CPU |
| K8s system overhead | ~300m | — | ~800Mi | — | None | measured | measured | calico + coredns + metrics-server + kube-proxy |
| Training (Riyam) | — | — | — | — | 1× RTX 6000 | — | — | Separate GPU node at CHI@UC — not in this cluster |

**VM choice justification:** 3× `m1.medium` (2 vCPU / 4 GB) gives a proper multi-node cluster matching the Kubespray lab reference. Total capacity: 6 vCPU / 12 GB. Measured peak usage: ~750m CPU / ~6 GB RAM across all 3 nodes — ~50% headroom at peak, within team cost rules.

---

## All Containers — Joint Deliverable

Table of all containers across all roles with Dockerfile and K8S manifest links.

| Role | Container | Image / Dockerfile | K8S Manifest |
|------|-----------|-------------------|--------------|
| **DevOps** | ActualBudget | `actualbudget/actual-server:latest` (upstream) | [`devops/k8s/actualbudget/deployment.yaml`](k8s/actualbudget/deployment.yaml) |
| **DevOps** | MLflow tracking server | `ghcr.io/mlflow/mlflow:v2.14.1` (upstream) | [`devops/k8s/mlflow/deployment.yaml`](k8s/mlflow/deployment.yaml) |
| **DevOps** | PostgreSQL (MLflow backend) | `postgres:16` (upstream) | [`devops/k8s/postgres/deployment.yaml`](k8s/postgres/deployment.yaml) |
| **DevOps** | MinIO (artifact object store) | `minio/minio:RELEASE.2024-11-07T00-52-20Z` (upstream) | [`devops/k8s/minio/deployment.yaml`](k8s/minio/deployment.yaml) |
| **Serving** | Transaction Classifier (FastAPI) | [`serving/docker/Dockerfile.fastapi`](../serving/docker/Dockerfile.fastapi) | [`devops/k8s/serving/deployment.yaml`](k8s/serving/deployment.yaml) |
| **Training** | Training container (CPU) | [`training/Dockerfile`](../training/Dockerfile) | None — runs on CHI@UC GPU node via Docker |
| **Training** | Training container (GPU) | [`training/Dockerfile.gpu`](../training/Dockerfile.gpu) | None — runs on CHI@UC GPU node via Docker |
| **Data** | Feature computation | [`data/pipelines/feature_computation/Dockerfile`](../data/pipelines/feature_computation/Dockerfile) | None — runs as batch job |

---

## IaC / CaC Stack

| Layer | Tool | Location |
|-------|------|----------|
| VM provisioning | Terraform (OpenStack provider) | `devops/iac/tf/` |
| Node configuration | Ansible + pre_k8s playbook | `devops/iac/ansible/pre_k8s/` |
| Kubernetes install | Kubespray (cluster.yml) | `devops/iac/gourmetgram-iac/ansible/k8s/kubespray/` |
| Kubespray inventory | Custom hosts.yaml + group_vars | `devops/iac/ansible/k8s/inventory/mycluster/` |
| Secrets management | Sealed Secrets v0.27.1 | `devops/k8s/sealed-secrets/` |
| GitOps delivery | Argo CD v2.11.0 + Image Updater v0.15.1 | `devops/k8s/argocd/` |

---

## Secrets Hygiene

No secrets are committed to Git. All credentials use Sealed Secrets:

| Secret | Keys | Used by |
|--------|------|---------|
| `postgres-credentials` | username, password, dbname | MLflow → PostgreSQL |
| `minio-credentials` | accesskey, secretkey | MLflow → MinIO artifact store |

The SealedSecret YAML files (`*-sealed.yaml`) are encrypted with the cluster's public key — safe to commit. Only this cluster can decrypt them.

To create a new sealed secret:
```bash
kubectl create secret generic <name> \
  --namespace mlops \
  --from-literal=<key>=<value> \
  --dry-run=client -o yaml \
| kubeseal --controller-name=sealed-secrets \
    --controller-namespace=kube-system --format yaml \
| python3 -c "import sys,yaml; d=yaml.safe_load(sys.stdin); d['spec'].pop('template',None); print(yaml.dump(d))" \
> devops/k8s/sealed-secrets/<name>-sealed.yaml
kubectl apply -f devops/k8s/sealed-secrets/<name>-sealed.yaml
```

---

## Teardown (after grading)

```bash
cd devops/iac/tf
terraform destroy -auto-approve
```

Destroys all 3 VMs, security groups, floating IP, and private network on Chameleon.
