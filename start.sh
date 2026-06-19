#!/bin/bash
set -e

echo "🚀 Starting Exam Anxiety Detector (monolithic mode)..."

# ── Start FastAPI backend on internal port 8000 ──────────────────────────────
echo "Starting FastAPI backend on port 8000..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# ── Wait for the backend to be ready (model load can take ~30s on cold start) ─
echo "Waiting for backend to be ready..."
for i in $(seq 1 60); do
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend is ready!"
    break
  fi
  echo "  ...waiting ($i/60)"
  sleep 3
done

# ── Start Streamlit frontend on the public $PORT ──────────────────────────────
echo "Starting Streamlit frontend on port $PORT..."
streamlit run frontend/app.py \
  --server.port "$PORT" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false

# If Streamlit exits, kill the backend too
kill $BACKEND_PID
