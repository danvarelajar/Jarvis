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
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434

# Create data directory for persistence
RUN mkdir -p /app/data
VOLUME /app/data

# Expose port
EXPOSE 3000

# Run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "3000"]
