#!/usr/bin/env bash
set -euo pipefail

# Google Cloud VM setup script for Gomoku AI inference server.
# Usage example:
#   sudo MODEL_URL=https://... MODEL_CHECK_INTERVAL_MS=60000 bash deploy/gcp_ai_server_setup.sh

APP_DIR="/opt/gomoku-ai"
SERVICE_NAME="gomoku-ai-server"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
ENV_DIR="/etc/gomoku-ai"
ENV_FILE="${ENV_DIR}/server.env"
LOG_DIR="/var/log/gomoku-ai"
NODE_VERSION="20"

require_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    echo "[x] Please run this script as root (sudo)."
    exit 1
  fi
}

install_prereqs() {
  if ! command -v curl >/dev/null 2>&1; then
    apt-get update
    apt-get install -y curl
  fi
  apt-get update
  apt-get install -y ca-certificates gnupg build-essential git rsync
}

install_node() {
  if ! command -v node >/dev/null 2>&1 || [[ "$(node -v 2>/dev/null | cut -c2-)" != ${NODE_VERSION}* ]]; then
    echo "[*] Installing Node.js ${NODE_VERSION}.x via NodeSource..."
    install -d -m 0755 /etc/apt/keyrings
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_${NODE_VERSION}.x nodistro main" > /etc/apt/sources.list.d/nodesource.list
    apt-get update
    apt-get install -y nodejs
  fi
}

create_service_user() {
  if ! id -u gomoku >/dev/null 2>&1; then
    echo "[*] Creating service user 'gomoku'"
    useradd --system --shell /usr/sbin/nologin --home ${APP_DIR} gomoku
  fi
}

sync_repository() {
  mkdir -p "${APP_DIR}"
  rsync -a --delete --exclude '.git' --exclude 'node_modules' ./ "${APP_DIR}/"
  chown -R gomoku:gomoku "${APP_DIR}"
}

install_dependencies() {
  cd "${APP_DIR}"
  sudo -u gomoku HOME="${APP_DIR}" npm ci
  sudo -u gomoku HOME="${APP_DIR}" npm run build
}

install_service_unit() {
  install -D -m 0644 "${APP_DIR}/deploy/gomoku-ai-server.service" "${SERVICE_FILE}"
}

seed_env_file() {
  mkdir -p "${ENV_DIR}"
  if [[ ! -f "${ENV_FILE}" ]]; then
    cat <<'EOF' > "${ENV_FILE}"
# Port the HTTP server listens on
PORT=8080
# Location of the TensorFlow model on disk
MODEL_PATH=/opt/gomoku-ai/gomoku_model_prod/model.json
# MODEL_URL=https://you-supabase-url/storage/v1/object/public/models/latest/model.json
# MODEL_CHECK_INTERVAL_MS=60000
# Enable GPU if CUDA drivers are installed on the VM
TF_USE_GPU=0
TF_FORCE_GPU_ALLOW_GROWTH=1
EOF
  fi

  local updated="0"
  apply_override() {
    local key="$1"
    local value="$2"
    if [[ -z "${value}" ]]; then
      return
    fi
    if grep -q "^${key}=" "${ENV_FILE}"; then
      sed -i "s|^${key}=.*|${key}=${value}|" "${ENV_FILE}"
    else
      echo "${key}=${value}" >> "${ENV_FILE}"
    }
    updated="1"
  }

  apply_override MODEL_URL "${MODEL_URL:-}"
  apply_override MODEL_CHECK_INTERVAL_MS "${MODEL_CHECK_INTERVAL_MS:-}"
  apply_override PORT "${PORT:-}"
  apply_override MODEL_PATH "${MODEL_PATH:-}"
  apply_override TF_USE_GPU "${TF_USE_GPU:-}"

  if [[ "${updated}" == "1" ]]; then
    echo "[*] Updated ${ENV_FILE} with provided overrides."
  fi
}

prepare_logs() {
  mkdir -p "${LOG_DIR}"
  chown -R gomoku:gomoku "${LOG_DIR}"
  chmod 0755 "${LOG_DIR}"
}

reload_restart() {
  systemctl daemon-reload
  systemctl enable "${SERVICE_NAME}"
  systemctl restart "${SERVICE_NAME}"
}

print_summary() {
  cat <<'EOT'
============================================
Gomoku AI inference server deployment complete.
Service  : gomoku-ai-server
Logs     : /var/log/gomoku-ai/
Config   : /etc/gomoku-ai/server.env

Key commands:
  sudo systemctl status gomoku-ai-server --no-pager
  sudo journalctl -u gomoku-ai-server -f
  sudo systemctl restart gomoku-ai-server

Make sure TCP/8080 is open in the GCP firewall to allow web access.
============================================
EOT
}

require_root
install_prereqs
install_node
create_service_user
sync_repository
install_dependencies
install_service_unit
seed_env_file
prepare_logs
reload_restart
print_summary