GPU Quickstart for Training Pipeline

Prerequisites
- Windows 11 with WSL2, or Linux
- NVIDIA GPU with recent driver
- Docker Desktop + NVIDIA Container Toolkit

Start (GPU)
- From repo root: `npm run docker:gpu`
- Windows helper: `scripts/start_docker_gpu.ps1`
- Follow logs: `docker compose logs -f trainer`
- Expect to see: `[TF] Using @tensorflow/tfjs-node-gpu backend.` in trainer logs

Local Backend Check (optional)
- Build TypeScript once: `npm run build`
- Print backend info: `npm run check:tf`
  - If backend is not `tensorflow`, heavy distillation training is skipped outside Docker (copy-only fallback).

Environment
- In GPU container: `TF_USE_GPU=1`, `TF_FORCE_GPU_ALLOW_GROWTH=1`
- Locally you can force CPU by setting `TF_USE_GPU=0` in `.env`

Troubleshooting
- Ensure `nvidia-smi` works on the host
- Docker Desktop → Settings → Resources → WSL integration enabled
- NVIDIA Container Toolkit installed and available to Docker runtime

