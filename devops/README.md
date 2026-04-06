# DevOps / Platform — proj06

Owner: Puneeth (pk3058@nyu.edu)

3-node Kubernetes cluster running on Chameleon KVM@TACC. All shared platform services (MLflow, PostgreSQL, MinIO, ActualBudget) run here. Cluster floating IP: `129.114.27.175`

---

## What was built

- Terraform provisions 3x `m1.medium` VMs on KVM@TACC (2 vCPU, 4 GB each)
- Kubespray installs Kubernetes v1.30.4 across all 3 nodes
- All platform services deployed as K8s workloads in the `mlops` namespace
- Secrets managed via Sealed Secrets — no plaintext credentials in the repo
- GitOps via Argo CD — pushes to `devops/k8s/` auto-sync to the cluster
- Argo CD Image Updater watches ghcr.io and auto-deploys new serving container tags

## Services running

| Service | URL | Notes |
|---------|-----|-------|
| ActualBudget | `http://129.114.27.175:30506` | The open source finance app we're extending |
| MLflow | `http://129.114.27.175:30500` | Experiment tracking for Riyam's training runs |
| Transaction Classifier | `http://129.114.27.175:30508/classify` | Jayraj's FastAPI serving container |

Riyam's training containers should set:
```
MLFLOW_TRACKING_URI=http://129.114.27.175:30500
```

## How to bring it up from scratch

**1. Provision VMs with Terraform**
```bash
cd devops/iac/tf
terraform init
terraform apply -auto-approve
```
Creates private network (192.168.1.0/24), 3 VMs at .11/.12/.13, floating IP on node1.

**2. Pre-K8s node config**
```bash
cd devops/iac/ansible
ANSIBLE_PIPELINING=False ansible-playbook -i inventory-proj06.yml \
  pre_k8s/pre_k8s_configure.yml --become
```
Disables swap, sets hostnames, opens required ports.

**3. Install Kubernetes with Kubespray**
```bash
cd devops/iac/gourmetgram-iac/ansible/k8s/kubespray
ANSIBLE_PIPELINING=False ansible-playbook \
  -i ../../inventory/mycluster/hosts.yaml cluster.yml --become
```
Installs K8s v1.30.4 with calico CNI, local-path provisioner, metrics-server, helm, Argo CD.

Key override in group_vars: `override_system_hostname: false` — without this Kubespray times out through the proxy chain.

**4. Deploy platform services**
```bash
export KUBECONFIG=~/.kube/config-proj06
kubectl apply -f devops/k8s/namespace.yaml
kubectl apply -f devops/k8s/sealed-secrets/controller.yaml
kubectl apply -f devops/k8s/sealed-secrets/
kubectl apply -f devops/k8s/postgres/
kubectl apply -f devops/k8s/minio/
kubectl apply -f devops/k8s/mlflow/
kubectl apply -f devops/k8s/actualbudget/
kubectl apply -f devops/k8s/argocd/image-updater-install.yaml
kubectl apply -f devops/k8s/argocd/app-platform.yaml
kubectl apply -f devops/k8s/argocd/app-serving.yaml
```
Sealed Secrets goes first so credentials are available before MLflow starts.

## IaC stack

| Layer | Tool | Location |
|-------|------|----------|
| VM provisioning | Terraform (OpenStack provider) | `devops/iac/tf/` |
| Node config | Ansible pre_k8s playbook | `devops/iac/ansible/pre_k8s/` |
| K8s install | Kubespray 2.26.0 | `devops/iac/ansible/k8s/` |
| Secrets | Sealed Secrets v0.27.1 | `devops/k8s/sealed-secrets/` |
| GitOps | Argo CD v2.11.0 + Image Updater v0.15.1 | `devops/k8s/argocd/` |

## Secrets

No plaintext credentials anywhere in the repo. PostgreSQL and MinIO credentials are sealed with the cluster's public key using `kubeseal`. The encrypted YAML files are safe to commit — only this cluster can decrypt them.

The MLflow deployment pulls credentials via `secretKeyRef` pointing to the sealed secret names. Nothing hardcoded.

## Resource requests

| Pod | CPU Request | CPU Limit | Mem Request | Mem Limit |
|-----|-------------|-----------|-------------|-----------|
| ActualBudget | 250m | 500m | 256Mi | 512Mi |
| MLflow | 500m | 1000m | 512Mi | 1Gi |
| PostgreSQL | 250m | 500m | 256Mi | 512Mi |
| MinIO | 250m | 500m | 512Mi | 1Gi |
| Transaction Classifier | 500m | 1000m | 512Mi | 1Gi |

## Teardown

```bash
cd devops/iac/tf
terraform destroy -auto-approve
```
Deletes all 3 VMs, security groups, floating IP, and private network on Chameleon. Run this after grading.
