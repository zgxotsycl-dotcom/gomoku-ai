# --- Stage 1: Build ---
FROM node:20-slim as builder

WORKDIR /app

# Copy all source files
COPY . .

# Install all dependencies, including devDependencies
RUN npm install --also=dev

# Build the project for CPU environment
RUN npm run build:cpu

# --- Stage 2: Production ---
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# Install Node.js and other essentials
RUN apt-get update && \
    apt-get install -y curl git && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only production dependencies manifest
COPY package.json package-lock.json* ./

# Install only production dependencies
RUN npm install --omit=dev

# Copy the built application from the builder stage
COPY --from=builder /app/dist ./dist

# Copy other necessary files
COPY model_main ./model_main

CMD ["npm", "run", "start:pipeline"]
