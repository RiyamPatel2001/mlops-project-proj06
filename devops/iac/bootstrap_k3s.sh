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
log "Step 8/9: Creating namespace and secrets..."
kubectl apply -f /home/cc/mlops-project-proj06/devops/k8s/namespace.yaml

# MinIO credentials
kubectl create secret generic minio-credentials \
  --namespace=mlops \
  --from-literal=MINIO_ROOT_USER=minioadmin \
  --from-literal=MINIO_ROOT_PASSWORD=minioadmin123 \
  --dry-run=client -o yaml | kubectl apply -f -

# Postgres credentials
kubectl create secret generic postgres-credentials \
  --namespace=mlops \
  --from-literal=POSTGRES_USER=mlops_user \
  --from-literal=POSTGRES_PASSWORD=mlops_pass \
  --from-literal=POSTGRES_DB=mlops \
  --dry-run=client -o yaml | kubectl apply -f -

# ── Step 9: Apply all K8s manifests ───────────────────────────────────────────
log "Step 9/9: Applying K8s manifests..."
K8S=/home/cc/mlops-project-proj06/devops/k8s

# Core storage
kubectl apply -f $K8S/postgres/pvc.yaml
kubectl apply -f $K8S/minio/pvc.yaml
kubectl apply -f $K8S/mlflow/pvc.yaml
kubectl apply -f $K8S/serving/pvc.yaml

# Core services
kubectl apply -f $K8S/postgres/deployment.yaml
kubectl apply -f $K8S/postgres/service.yaml
kubectl apply -f $K8S/minio/deployment.yaml
kubectl apply -f $K8S/minio/service.yaml
kubectl apply -f $K8S/mlflow/deployment.yaml
kubectl apply -f $K8S/mlflow/service.yaml

# ActualBudget (custom build) + nginx proxy for COOP/COEP headers
kubectl apply -f $K8S/actualbudget/pvc.yaml
kubectl apply -f $K8S/actualbudget/deployment.yaml
kubectl apply -f $K8S/actualbudget/service.yaml
kubectl apply -f $K8S/actualbudget-proxy/configmap.yaml
kubectl apply -f $K8S/actualbudget-proxy/deployment.yaml
kubectl apply -f $K8S/actualbudget-proxy/service.yaml

# Transaction classifier (ML serving)
kubectl apply -f $K8S/serving/deployment.yaml
kubectl apply -f $K8S/serving/service.yaml

log "All manifests applied. Waiting 30s for pods to start..."
sleep 30
kubectl get pods -n mlops

# ── Done ───────────────────────────────────────────────────────────────────────
echo ""
echo "============================================="
log "k3s bootstrap complete!"
echo ""
echo "StorageClasses available:"
kubectl get storageclass
echo ""
echo "Cluster is ready. Next: copy kubeconfig to your laptop."
echo ""
echo "From your laptop:"
echo "  scp -i ~/.ssh/id_rsa_chameleon cc@<VM-IP>:~/.kube/config ~/.kube/chameleon-proj06.yaml"
echo "  sed -i '' 's/127.0.0.1/<VM-IP>/g' ~/.kube/chameleon-proj06.yaml"
echo "  export KUBECONFIG=~/.kube/chameleon-proj06.yaml"
echo "  kubectl get nodes"
echo "============================================="
