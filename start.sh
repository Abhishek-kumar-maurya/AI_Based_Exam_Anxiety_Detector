#!/bin/bash
# start.sh — launches both services on Render (free tier, single dyno)
set -e

echo "🚀 Starting Exam Anxiety Detector…"

# Start FastAPI backend on port 8000 (internal)
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for the backend to become healthy (model load can take ~30 s on cold start)
echo "Waiting for backend…"
for i in $(seq 1 60); do
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend ready!"
    break
  fi
  echo "  …waiting ($i/60)"
  sleep 3
done

# Start Streamlit frontend on the public $PORT
echo "Starting Streamlit on port $PORT…"
streamlit run frontend/app.py \
  --server.port "$PORT" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false

# Clean up backend when Streamlit exits
kill $BACKEND_PID
