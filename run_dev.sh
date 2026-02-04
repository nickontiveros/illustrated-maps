#!/bin/bash
# Development startup script for MapGen

echo "Starting MapGen development environment..."
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Please run this script from the maps directory"
    exit 1
fi

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start backend
echo "Starting API backend on http://localhost:8000..."
uvicorn mapgen.api.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
sleep 2

# Start frontend
echo "Starting frontend on http://localhost:5173..."
cd frontend && npm run dev &
FRONTEND_PID=$!

echo ""
echo "==================================="
echo "MapGen Development Environment"
echo "==================================="
echo "Backend API: http://localhost:8000"
echo "Frontend:    http://localhost:5173"
echo "API Docs:    http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Wait for either process to exit
wait
