FROM node:20-bullseye

WORKDIR /app

# Install minimal system deps (useful for tfjs-node prebuilt binding fallback)
RUN apt-get update && apt-get install -y python3 make g++ && rm -rf /var/lib/apt/lists/*

# Copy package manifests first for better layer caching
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy rest of the source
COPY . .

# Build TypeScript
RUN npm run build

# Persist important data dirs as Docker volumes
VOLUME ["/app/replay_buffer", "/app/gomoku_model_prod", "/app/past_models", "/app/logs"]

# Reasonable defaults (can be overridden at runtime)
ENV BOARD_SIZE=15 \
    FOREVER=true \
    RUN_DISTILLATION=true \
    GATING_ENABLED=true \
    UPLOAD_MODEL_AFTER=true \
    AUTO_GENERATE_OPENING_BOOK=true \
    IMPORT_OPENING_BOOK=true \
    PIPELINE_CYCLES=0 \
    PIPELINE_INTERVAL_MS=0

# Start the full training pipeline (infinite cycles by default)
CMD ["node", "dist/start_pipeline.js"]
