#!/usr/bin/env bash
# =============================================================================
# teardown.sh — Delete Chameleon VM and security group for MLOps proj06
# =============================================================================
# Run AFTER grading is complete to free up Chameleon quota.
#
# Usage:
#   source ~/chameleon-openrc.sh
#   bash devops/iac/teardown.sh
# =============================================================================

set -euo pipefail

INSTANCE_NAME="mlops-k8s-proj06"
SEC_GROUP="mlops-proj06-sg"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }

echo -e "${RED}WARNING: This will permanently delete $INSTANCE_NAME and $SEC_GROUP.${NC}"
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5

# ── Delete instance ────────────────────────────────────────────────────────────
if openstack server show "$INSTANCE_NAME" &>/dev/null; then
  log "Deleting instance $INSTANCE_NAME..."
  openstack server delete "$INSTANCE_NAME" --wait
  log "Instance deleted"
else
  warn "Instance $INSTANCE_NAME not found — already deleted?"
fi

# ── Delete security group ──────────────────────────────────────────────────────
if openstack security group show "$SEC_GROUP" &>/dev/null; then
  log "Deleting security group $SEC_GROUP..."
  openstack security group delete "$SEC_GROUP"
  log "Security group deleted"
else
  warn "Security group $SEC_GROUP not found — already deleted?"
fi

log "Teardown complete. Chameleon quota freed."
