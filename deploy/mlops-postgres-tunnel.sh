#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-10.56.0.162}"
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_USER="${REMOTE_USER:-cc}"
REMOTE_K8S_NAMESPACE="${REMOTE_K8S_NAMESPACE:-mlops}"
REMOTE_SERVICE_NAME="${REMOTE_SERVICE_NAME:-postgres}"
LOCAL_PORT="${LOCAL_PORT:-15432}"
REMOTE_FORWARD_PORT="${REMOTE_FORWARD_PORT:-25432}"
DOCKER_HOST_GATEWAY_BIND_ADDRESS="${DOCKER_HOST_GATEWAY_BIND_ADDRESS:-172.17.0.1}"
SSH_KEY_PATH="${SSH_KEY_PATH:-/home/cc/.ssh/id_ed25519_mlops_pg_tunnel}"

exec /usr/bin/autossh \
  -M 0 \
  -- \
  -T \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes \
  -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile=/home/cc/.ssh/known_hosts \
  -i "${SSH_KEY_PATH}" \
  -p "${REMOTE_PORT}" \
  -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_FORWARD_PORT}" \
  -L "${DOCKER_HOST_GATEWAY_BIND_ADDRESS}:${LOCAL_PORT}:127.0.0.1:${REMOTE_FORWARD_PORT}" \
  "${REMOTE_USER}@${REMOTE_HOST}" \
  kubectl -n "${REMOTE_K8S_NAMESPACE}" port-forward --address 127.0.0.1 "svc/${REMOTE_SERVICE_NAME}" "${REMOTE_FORWARD_PORT}:5432"
