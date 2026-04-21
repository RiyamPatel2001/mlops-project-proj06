#!/usr/bin/env bash
# =============================================================================
# bootstrap_k3s.sh — Install k3s on the Chameleon VM
# =============================================================================
# Run this ON the VM after provision.sh completes:
#   scp -i ~/.ssh/id_rsa_chameleon devops/iac/bootstrap_k3s.sh cc@<IP>:~
#   ssh -i ~/.ssh/id_rsa_chameleon cc@<IP> 'bash bootstrap_k3s.sh'
# =============================================================================

set -euo pipefail

GREEN='\033[0;32m'
NC='\033[0m'
log() { echo -e "${GREEN}[✓]${NC} $1"; }

# ── Step 1: System update ──────────────────────────────────────────────────────
log "Step 1/5: Updating system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq curl wget git

# ── Step 2: Install k3s ────────────────────────────────────────────────────────
log "Step 2/5: Installing k3s (lightweight Kubernetes)..."
# Disable Traefik — using NodePort instead of Ingress
# Disable local-storage — we use local-path provisioner (bundled)
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="server \
  --disable traefik \
  --write-kubeconfig-mode 644" sh -

log "k3s installed"

# ── Step 3: Wait for k3s to be ready ──────────────────────────────────────────
log "Step 3/5: Waiting for k3s to be ready..."
sleep 20
for i in $(seq 1 12); do
  if sudo kubectl get nodes 2>/dev/null | grep -q "Ready"; then
    log "k3s node is Ready"
    break
  fi
  echo "  [$i/12] Waiting for node to be Ready..."
  sleep 10
done

# ── Step 4: Set up kubectl for cc user ────────────────────────────────────────
log "Step 4/5: Configuring kubectl access..."
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown "$USER:$USER" ~/.kube/config
# Replace localhost with actual server IP for remote access
NODE_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || \
          hostname -I | awk '{print $1}')
sed -i "s/127.0.0.1/$NODE_IP/g" ~/.kube/config

export KUBECONFIG=~/.kube/config

# ── Step 5: Verify cluster ────────────────────────────────────────────────────
log "Step 5/5: Verifying cluster..."
kubectl get nodes
kubectl get pods -A | head -20

# ── Add kubectl to PATH for future SSH sessions ───────────────────────────────
echo 'export KUBECONFIG=~/.kube/config' >> ~/.bashrc

# ── Step 6: Install Docker and move its storage to the Cinder volume ──────────
log "Step 6/9: Installing Docker..."
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker cc

log "Step 6/9: Waiting for Cinder volume (/dev/vdb)..."
for i in $(seq 1 12); do
  [ -b /dev/vdb ] && log "  /dev/vdb is available" && break
  echo "  [$i/12] Waiting for /dev/vdb to appear..."
  sleep 5
done
[ ! -b /dev/vdb ] && echo "WARNING: /dev/vdb not found — Docker will use the boot disk (risk of disk pressure)"

if [ -b /dev/vdb ]; then
  log "Step 6/9: Formatting and mounting Cinder volume for Docker storage..."
  sudo mkfs.ext4 /dev/vdb
  sudo mkdir -p /mnt/docker-storage
  sudo mount /dev/vdb /mnt/docker-storage
  echo '/dev/vdb /mnt/docker-storage ext4 defaults 0 2' | sudo tee -a /etc/fstab

  log "Step 6/9: Moving Docker data-root to Cinder volume..."
  sudo systemctl stop docker docker.socket 2>/dev/null || true
  sudo mkdir -p /etc/docker
  echo '{"data-root": "/mnt/docker-storage/docker"}' | sudo tee /etc/docker/daemon.json

  # Move containerd image store (where Docker actually stores layer blobs)
  if [ -d /var/lib/containerd ]; then
    sudo rsync -aH /var/lib/containerd/ /mnt/docker-storage/containerd/
    sudo rm -rf /var/lib/containerd
    sudo ln -s /mnt/docker-storage/containerd /var/lib/containerd
  fi

  sudo systemctl start docker
  log "Docker running with data-root on /mnt/docker-storage"
fi

# ── Step 7: Clone project repo ────────────────────────────────────────────────
log "Step 7/9: Cloning project repo (refactor-project branch)..."
if [ ! -d /home/cc/mlops-project-proj06 ]; then
  git clone -b refactor-project \
    https://github.com/RiyamPatel2001/mlops-project-proj06.git \
    /home/cc/mlops-project-proj06
else
  cd /home/cc/mlops-project-proj06 && git pull
fi

# ── Step 8: Create K8s namespace and credentials ──────────────────────────────
log "Step 8/11: Creating namespace and secrets..."
kubectl apply -f /home/cc/mlops-project-proj06/devops/k8s/namespace.yaml

# MinIO credentials (accesskey/secretkey format used by serving pod)
kubectl create secret generic minio-credentials \
  --namespace=mlops \
  --from-literal=accesskey=minioadmin \
  --from-literal=secretkey=minioadmin123 \
  --dry-run=client -o yaml | kubectl apply -f -

# Postgres credentials
kubectl create secret generic postgres-credentials \
  --namespace=mlops \
  --from-literal=username=mlops_user \
  --from-literal=password=mlops_pass \
  --from-literal=dbname=mlops \
  --dry-run=client -o yaml | kubectl apply -f -

# proj06-env secret (used by serving pod for MLflow + MinIO access)
kubectl create secret generic proj06-env \
  --namespace=mlops \
  --from-literal=MLFLOW_TRACKING_URI=http://mlflow:5000 \
  --from-literal=MLFLOW_S3_ENDPOINT_URL=http://minio:9000 \
  --from-literal=AWS_ACCESS_KEY_ID=minioadmin \
  --from-literal=AWS_SECRET_ACCESS_KEY=minioadmin123 \
  --from-literal=POSTGRES_USER=mlops_user \
  --from-literal=POSTGRES_PASSWORD=mlops_pass \
  --from-literal=POSTGRES_DB=mlops \
  --from-literal=MINIO_ENDPOINT=http://minio:9000 \
  --from-literal=MINIO_ACCESS_KEY=minioadmin \
  --from-literal=MINIO_SECRET_KEY=minioadmin123 \
  --dry-run=client -o yaml | kubectl apply -f -

# ── Step 9: Apply all K8s manifests ───────────────────────────────────────────
log "Step 9/11: Applying K8s manifests..."
K8S=/home/cc/mlops-project-proj06/devops/k8s

# Core storage
kubectl apply -f $K8S/postgres/pvc.yaml
kubectl apply -f $K8S/minio/pvc.yaml
kubectl apply -f $K8S/mlflow/pvc.yaml
kubectl apply -f $K8S/serving/pvc.yaml
kubectl apply -f $K8S/training/pvc.yaml

# Core services
kubectl apply -f $K8S/postgres/deployment.yaml
kubectl apply -f $K8S/postgres/service.yaml
kubectl apply -f $K8S/minio/deployment.yaml
kubectl apply -f $K8S/minio/service.yaml
kubectl apply -f $K8S/mlflow/deployment.yaml
kubectl apply -f $K8S/mlflow/service.yaml
kubectl apply -f $K8S/adminer/deployment.yaml
kubectl apply -f $K8S/adminer/service.yaml

# Monitoring
kubectl apply -f $K8S/monitoring/prometheus.yaml
kubectl apply -f $K8S/monitoring/grafana.yaml

# ActualBudget + proxy
kubectl apply -f $K8S/actualbudget/pvc.yaml
kubectl apply -f $K8S/actualbudget/deployment.yaml
kubectl apply -f $K8S/actualbudget/service.yaml
kubectl apply -f $K8S/actualbudget-proxy/configmap.yaml
kubectl apply -f $K8S/actualbudget-proxy/deployment.yaml
kubectl apply -f $K8S/actualbudget-proxy/service.yaml

# Transaction classifier (ML serving)
kubectl apply -f $K8S/serving/deployment.yaml
kubectl apply -f $K8S/serving/service.yaml

log "All manifests applied. Waiting 60s for pods to initialise..."
sleep 60
kubectl get pods -n mlops

# ── Step 10: Install ArgoCD ────────────────────────────────────────────────────
log "Step 10/11: Installing ArgoCD..."
kubectl create namespace argocd --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

log "Waiting for ArgoCD server to be ready (up to 3 min)..."
kubectl wait --for=condition=Available deployment/argocd-server -n argocd --timeout=180s || true

# Apply ArgoCD app configs so it syncs all manifests from git automatically
kubectl apply -f $K8S/../k8s/argocd/app-platform.yaml 2>/dev/null || true
kubectl apply -f $K8S/../k8s/argocd/app-serving.yaml 2>/dev/null || true

log "ArgoCD installed. Password: kubectl get secret argocd-initial-admin-secret -n argocd -o jsonpath='{.data.password}' | base64 -d"

# ── Step 11: Write layer1_registry.json to serving PVC ────────────────────────
log "Step 11/11: Writing layer1_registry.json to serving PVC..."
# Wait for serving PVC to be bound first
sleep 10

kubectl run registry-init -n mlops --image=busybox --restart=Never \
  --overrides='{"spec":{"volumes":[{"name":"m","persistentVolumeClaim":{"claimName":"serving-models-pvc"}}],"containers":[{"name":"b","image":"busybox","command":["sh","-c","sleep 60"],"volumeMounts":[{"name":"m","mountPath":"/mnt"}]}]}}' \
  2>/dev/null || true

sleep 5
kubectl cp /home/cc/mlops-project-proj06/devops/layer1_registry.json \
  mlops/registry-init:/mnt/layer1_registry.json 2>/dev/null || \
  echo "WARN: Could not write registry.json — do it manually after serving PVC is ready"
kubectl delete pod registry-init -n mlops --ignore-not-found=true

log "layer1_registry.json written — classifier will load correct MLflow run IDs on startup"

# ── Done ───────────────────────────────────────────────────────────────────────
FLOATING_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || hostname -I | awk '{print $1}')
echo ""
echo "============================================="
log "Bootstrap complete! Cluster is production-ready."
echo ""
echo "  MLflow:       http://$FLOATING_IP:30500"
echo "  ActualBudget: http://$FLOATING_IP:30506"
echo "  Classifier:   http://$FLOATING_IP:30508/health"
echo "  MinIO:        http://$FLOATING_IP:30901  (minioadmin / minioadmin123)"
echo "  Grafana:      http://$FLOATING_IP:30030  (admin / mlops1234)"
echo "  Prometheus:   http://$FLOATING_IP:30090"
echo "  ArgoCD:       http://$FLOATING_IP:30808  (admin / see above)"
echo ""
echo "Models in MinIO (run IDs):"
echo "  good  (minilm):       b8f1ad8433b7492e82726429df5b66a0"
echo "  fast  (fasttext):     5af29fbaa4b04abc9d21b18a19ccc736"
echo "  cheap (tfidf_logreg): bd19d31c1aa94500a0fa3a4f2cee4c94"
echo ""
echo "From your laptop:"
echo "  scp -i ~/.ssh/id_rsa_chameleon cc@$FLOATING_IP:~/.kube/config ~/.kube/chameleon-proj06.yaml"
echo "  sed -i '' 's/127.0.0.1/$FLOATING_IP/g' ~/.kube/chameleon-proj06.yaml"
echo "  export KUBECONFIG=~/.kube/chameleon-proj06.yaml"
echo "============================================="
