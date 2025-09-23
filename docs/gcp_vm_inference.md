# Google Cloud VM Deployment Guide

This guide explains how to run the Gomoku AI inference server on a Google Cloud Compute Engine VM and expose it to the web.

## 1. Provision the VM

1. In Google Cloud Console, create a Compute Engine VM (Ubuntu 22.04 LTS or Debian 12 recommended).
2. Choose a machine type (e.g., 2-standard-2). For GPU inference you would need a GPU-capable machine with NVIDIA drivers, but CPU is sufficient for most cases.
3. In the Networking tab:
   - Add a firewall rule (or select the "Allow HTTP traffic" checkbox) that opens TCP port **8080**, which is the default port used by the server.

## 2. Prepare the VM

SSH into the VM and install git:

`ash
sudo apt-get update
sudo apt-get install -y git rsync
`

Clone this repository into your home directory:

`ash
git clone https://github.com/your-account/gomoku-ai.git
cd gomoku-ai
`

## 3. Populate environment variables

If you plan to load models from Supabase or expose additional services, edit deploy/gcp_ai_server_setup.sh or prepare the environment file ahead of time. The script will seed /etc/gomoku-ai/server.env with sensible defaults:

`ash
PORT=8080
MODEL_PATH=/opt/gomoku-ai/gomoku_model_prod/model.json
TF_USE_GPU=0
TF_FORCE_GPU_ALLOW_GROWTH=1
`

Set MODEL_URL if you want the server to download the model from a remote Supabase URL instead of using the bundled local model.

## 4. Run the setup script

Execute the automated setup script as root (it installs Node.js 20, the project dependencies, builds the TypeScript sources, and registers a systemd service):

`ash
sudo bash deploy/gcp_ai_server_setup.sh
`

The script will:

- Install Node.js 20 via NodeSource.
- Create a service user gomoku.
- Sync the repository to /opt/gomoku-ai.
- Run 
pm ci and 
pm run build as the gomoku user.
- Copy the systemd unit file to /etc/systemd/system/gomoku-ai-server.service.
- Seed /etc/gomoku-ai/server.env if it does not exist.
- Start and enable the gomoku-ai-server service.

## 5. Verify the service

Check the systemd status and tail the logs:

`ash
sudo systemctl status gomoku-ai-server --no-pager
sudo journalctl -u gomoku-ai-server -f
`

The HTTP API should now respond on port 8080. Test it locally:

`ash
curl http://localhost:8080/health
`

After opening the firewall, you can also access http://<vm-external-ip>:8080/health from your browser.

## 6. Updating the deployment

When you push new code:

1. SSH into the VM and cd /opt/gomoku-ai.
2. Pull changes and rebuild:

   `ash
   sudo -u gomoku git pull
   sudo -u gomoku npm ci
   sudo -u gomoku npm run build
   sudo systemctl restart gomoku-ai-server
   `

If you made significant changes (new files, removed files), rerun sudo bash deploy/gcp_ai_server_setup.sh from the repository root to resync the checkout and reapply the service file.

## 7. Optional: HTTPS / reverse proxy

For production use, front the service with an HTTPS-capable reverse proxy (e.g., Cloud HTTPS Load Balancer, Nginx, or Caddy) and terminate TLS there. The Gomoku server itself is stateless, so it can sit behind a load balancer if you need redundancy.

## 8. Troubleshooting

- **Port not reachable**: ensure the Compute Engine firewall rule allows inbound TCP/8080 and that no OS-level firewall blocks it.
- **Model not found**: update /etc/gomoku-ai/server.env with either a valid MODEL_PATH (local) or MODEL_URL (remote) and restart the service.
- **TensorFlow GPU errors**: set TF_USE_GPU=0 if the VM lacks CUDA support, or install NVIDIA drivers and libraries.

With these steps, the VM acts as a persistent web-facing inference server for the Gomoku AI.

