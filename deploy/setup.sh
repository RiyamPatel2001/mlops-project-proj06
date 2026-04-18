#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
ENV_TEMPLATE="${SCRIPT_DIR}/.env.example"

docker_cmd="docker"

set_docker_cmd() {
    if docker info >/dev/null 2>&1; then
        docker_cmd="docker"
    else
        docker_cmd="sudo docker"
    fi
}

ensure_swap() {
    if swapon --show | grep -q .; then
        echo "[2/4] Swap already enabled, skipping."
        return
    fi

    echo "[2/4] No swap detected. Creating a 4G swap file for safer image builds..."
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile

    if ! grep -q '^/swapfile ' /etc/fstab; then
        echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
    fi
}

echo "=== Chameleon Cloud Deployment Setup ==="
echo ""

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "[1/4] Installing Docker..."
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl git gnupg
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
      https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    sudo usermod -aG docker "$USER"
    echo "Docker installed. You may need to log out and back in for group changes."
else
    echo "[1/4] Docker already installed, skipping."
fi

ensure_swap
set_docker_cmd

# Check for .env file
if [ ! -f "${ENV_FILE}" ]; then
    echo ""
    echo "[3/4] No .env file found. Creating from template..."
    cp "${ENV_TEMPLATE}" "${ENV_FILE}"
    echo "*** IMPORTANT: Edit .env and replace <TEAMMATE_IP> with real IPs ***"
    echo "    Run: nano .env"
    echo ""
    echo "Then re-run this script."
    exit 1
else
    echo "[3/4] .env file found."
fi

if grep -Eq '<[^>]+>' "${ENV_FILE}"; then
    echo ""
    echo "[3/4] .env still contains placeholder values."
    echo "       Replace them before deploying:"
    echo "       nano ${ENV_FILE}"
    exit 1
fi

echo "[4/4] Building and starting containers..."
echo "       This will take 10-20 minutes on first run."
echo ""
cd "${SCRIPT_DIR}"

if ${docker_cmd} image inspect actual-custom:latest >/dev/null 2>&1; then
    echo "       Reusing preloaded actual-custom:latest image; building serving only."
    ${docker_cmd} compose build serving
else
    echo "       Building serving and custom Actual locally for ${DOCKER_PLATFORM:-linux/amd64}."
    ${docker_cmd} compose build serving actual
fi

${docker_cmd} compose up -d --no-build

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "  ML Serving API:    http://$(hostname -I | awk '{print $1}'):8000"
echo "  ML Serving Docs:   http://$(hostname -I | awk '{print $1}'):8000/docs"
echo "  Actual Budget:     http://$(hostname -I | awk '{print $1}'):5006"
echo ""
echo "  View logs:         ${docker_cmd} compose logs -f"
echo "  Stop everything:   ${docker_cmd} compose down"
