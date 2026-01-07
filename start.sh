#!/bin/bash

# Start FastAPI in background
echo "🚀 Starting Vellum Engine (FastAPI)..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Wait for API to warm up
sleep 3

# Start Streamlit
echo "🪶 Starting Vellum UI (Streamlit)..."
streamlit run app.py --server.port 8501
