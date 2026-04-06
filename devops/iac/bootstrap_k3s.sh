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
