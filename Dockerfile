# Build Stage
FROM node:18-alpine as build

WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install

COPY frontend/ .
RUN npm run build

# Final Stage
FROM python:3.11-slim

WORKDIR /app

# Install backend dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ backend/

# Copy built frontend static files
COPY --from=build /app/frontend/dist /app/frontend/dist

# Environment variables
ENV PYTHONUNBUFFERED=1

# Create data directory for persistence
RUN mkdir -p /app/data
VOLUME /app/data

# Expose port
EXPOSE 3000

# Run the application with multiple workers to handle concurrent requests
# This prevents new requests from being queued behind long-running Ollama inference
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "3000", "--workers", "4"]
