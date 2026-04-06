#!/usr/bin/env bash
# =============================================================================
# provision.sh — Provision Chameleon Cloud VM for MLOps proj06 K8S cluster
# =============================================================================
# Prerequisites:
#   - openstack CLI installed (pip install python-openstackclient)
#   - OpenStack RC file sourced: source ~/chameleon-openrc.sh
#   - SSH keypair registered on Chameleon (id_rsa_chameleon)
#
# Usage:
#   source ~/chameleon-openrc.sh
#   bash devops/iac/provision.sh
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
INSTANCE_NAME="mlops-k8s-proj06"
FLAVOR="m1.large"                   # 4 vCPU, 8 GB RAM
IMAGE="CC-Ubuntu22.04"
KEYPAIR_NAME="id_rsa_chameleon"
SEC_GROUP="mlops-proj06-sg"
NETWORK="sharednet1"

# ── Colors for output ─────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
die()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }

# ── Verify openstack CLI is authenticated ─────────────────────────────────────
log "Verifying OpenStack authentication..."
openstack token issue -f value -c id > /dev/null 2>&1 || \
  die "Not authenticated. Run: source ~/chameleon-openrc.sh"

# ── Check instance doesn't already exist ─────────────────────────────────────
if openstack server show "$INSTANCE_NAME" &>/dev/null; then
  warn "Instance $INSTANCE_NAME already exists. Getting its floating IP..."
  FLOATING_IP=$(openstack server show "$INSTANCE_NAME" \
    -f json | python3 -c "
import sys, json
data = json.load(sys.stdin)
addrs = data.get('addresses', {})
for net, ips in addrs.items():
    for ip in ips:
        if ip.get('OS-EXT-IPS:type') == 'floating':
            print(ip['addr'])
" 2>/dev/null || echo "")
  if [ -n "$FLOATING_IP" ]; then
    log "Floating IP: $FLOATING_IP"
    echo ""
    echo "SSH: ssh -i ~/.ssh/id_rsa_chameleon cc@$FLOATING_IP"
    exit 0
  fi
fi

# ── Step 1: Create security group ─────────────────────────────────────────────
log "Step 1/5: Creating security group $SEC_GROUP..."
if ! openstack security group show "$SEC_GROUP" &>/dev/null; then
  openstack security group create "$SEC_GROUP" \
    --description "MLOps proj06 K8S node security group"
  # SSH
  openstack security group rule create "$SEC_GROUP" \
    --protocol tcp --dst-port 22 --remote-ip 0.0.0.0/0
  # k3s API server
  openstack security group rule create "$SEC_GROUP" \
    --protocol tcp --dst-port 6443 --remote-ip 0.0.0.0/0
  # MLflow NodePort
  openstack security group rule create "$SEC_GROUP" \
    --protocol tcp --dst-port 30500 --remote-ip 0.0.0.0/0
  # ActualBudget NodePort
  openstack security group rule create "$SEC_GROUP" \
    --protocol tcp --dst-port 30506 --remote-ip 0.0.0.0/0
  # Transaction Classifier (FastAPI) NodePort
  openstack security group rule create "$SEC_GROUP" \
    --protocol tcp --dst-port 30508 --remote-ip 0.0.0.0/0
  # Allow all internal traffic within the security group
  openstack security group rule create "$SEC_GROUP" \
    --protocol tcp --dst-port 1:65535 --remote-group "$SEC_GROUP" 2>/dev/null || true
  log "Security group created with all required rules"
else
  warn "Security group $SEC_GROUP already exists — skipping creation"
fi

# ── Step 2: Boot instance ──────────────────────────────────────────────────────
log "Step 2/5: Booting instance $INSTANCE_NAME..."
# Try m1.large first; if KVM@TACC is at capacity, fall back to m1.medium
FLAVOR_FALLBACK="m1.medium"
USED_FLAVOR=""
for TRY_FLAVOR in "$FLAVOR" "$FLAVOR_FALLBACK"; do
  log "  Trying flavor $TRY_FLAVOR..."
  openstack server create \
    --flavor "$TRY_FLAVOR" \
    --image "$IMAGE" \
    --key-name "$KEYPAIR_NAME" \
    --security-group "$SEC_GROUP" \
    --network "$NETWORK" \
    "$INSTANCE_NAME" -f value -c id > /dev/null 2>&1 && USED_FLAVOR="$TRY_FLAVOR" && break
  warn "  Flavor $TRY_FLAVOR failed — KVM@TACC may be at capacity, trying next..."
  # Clean up any ERROR'd instance before retrying
  sleep 10
  openstack server delete "$INSTANCE_NAME" --wait 2>/dev/null || true
  sleep 5
done
[ -z "$USED_FLAVOR" ] && die "All flavors failed. KVM@TACC is at capacity — retry in a few minutes."
log "Instance creation submitted with flavor $USED_FLAVOR"

# ── Step 3: Wait for ACTIVE ────────────────────────────────────────────────────
log "Step 3/5: Waiting for ACTIVE status..."
STATUS=""
for i in $(seq 1 36); do
  STATUS=$(openstack server show "$INSTANCE_NAME" -f value -c status 2>/dev/null || echo "UNKNOWN")
  echo "  [$i/36] Status: $STATUS"
  [ "$STATUS" = "ACTIVE" ] && break
  [ "$STATUS" = "ERROR" ] && die "Instance entered ERROR state (KVM@TACC capacity issue — delete and retry)"
  sleep 10
done
[ "$STATUS" != "ACTIVE" ] && die "Instance did not become ACTIVE after 6 minutes"
log "Instance is ACTIVE (flavor: $USED_FLAVOR)"

# ── Step 4: Associate floating IP ─────────────────────────────────────────────
log "Step 4/5: Associating floating IP..."
# Find an unassociated floating IP first
FLOATING_IP=$(openstack floating ip list \
  --status DOWN \
  -f value -c "Floating IP Address" 2>/dev/null | head -1 || echo "")

if [ -z "$FLOATING_IP" ]; then
  warn "No unassociated floating IP found — allocating a new one"
  FLOATING_IP=$(openstack floating ip create public \
    -f value -c floating_ip_address)
fi

openstack server add floating ip "$INSTANCE_NAME" "$FLOATING_IP"
log "Floating IP $FLOATING_IP associated"

# ── Step 5: Wait for SSH ───────────────────────────────────────────────────────
log "Step 5/5: Waiting for SSH to become available..."
for i in $(seq 1 20); do
  if ssh -o StrictHostKeyChecking=no \
         -o ConnectTimeout=5 \
         -o BatchMode=yes \
         -i ~/.ssh/id_rsa_chameleon \
         cc@"$FLOATING_IP" "echo ok" &>/dev/null; then
    log "SSH is ready"
    break
  fi
  echo "  [$i/20] Waiting for SSH..."
  sleep 15
done

# ── Done ───────────────────────────────────────────────────────────────────────
echo ""
echo "============================================="
log "Provisioning complete!"
echo "  Instance : $INSTANCE_NAME"
echo "  Flavor   : $USED_FLAVOR"
echo "  IP       : $FLOATING_IP"
echo "============================================="
echo ""
echo "Next steps:"
echo "  1. Copy bootstrap script to VM:"
echo "     scp -i ~/.ssh/id_rsa_chameleon devops/iac/bootstrap_k3s.sh cc@$FLOATING_IP:~"
echo ""
echo "  2. Run bootstrap on VM:"
echo "     ssh -i ~/.ssh/id_rsa_chameleon cc@$FLOATING_IP 'bash bootstrap_k3s.sh'"
echo ""
echo "  3. Copy kubeconfig to laptop:"
echo "     scp -i ~/.ssh/id_rsa_chameleon cc@$FLOATING_IP:~/.kube/config ~/.kube/chameleon-proj06.yaml"
echo "     sed -i '' \"s/127.0.0.1/$FLOATING_IP/g\" ~/.kube/chameleon-proj06.yaml"
echo "     export KUBECONFIG=~/.kube/chameleon-proj06.yaml"
echo ""
