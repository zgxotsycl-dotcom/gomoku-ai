# --- Stage 1: Build --- 
FROM node:20-slim as builder

WORKDIR /app

# Copy all source files from the root
COPY . .

# Set the working directory to the scripts folder
WORKDIR /app/training_scripts

# Install all dependencies
RUN npm install --also=dev

# Build the project
RUN npm run build:cpu

# --- Stage 2: Production --- 
FROM node:20-slim

WORKDIR /app

# Copy only production dependencies manifest
COPY --from=builder /app/training_scripts/package.json ./package.json
COPY --from=builder /app/training_scripts/package-lock.json* ./package-lock.json*

# Install only production dependencies
RUN npm install --omit=dev

# Copy the built application from the builder stage
COPY --from=builder /app/training_scripts/dist ./dist

# Copy the model for the server to use
COPY --from=builder /app/training_scripts/model_main ./model_main

CMD ["node", "dist/server.js"]
