#!/bin/bash

# Target the specific Python 3.12 virtual environment path for the SK branch
VENV_PATH="/Users/ericbeier/projectaria_sk_python_env"

if [ ! -d "$VENV_PATH" ]; then
    echo "Creating new Python 3.12 Virtual Environment at $VENV_PATH"
    python3.12 -m venv "$VENV_PATH"
fi

if [ -f "$VENV_PATH/bin/activate" ]; then
    echo "Activating Python 3.12 Virtual Environment at $VENV_PATH"
    source "$VENV_PATH/bin/activate"
else
    echo "Error: Python 3.12 Virtual environment not found at $VENV_PATH"
    exit 1
fi

echo "Installing requirements from requirements-sk.txt..."
pip install -r requirements-sk.txt

# Create a 'runs' directory if it doesn't exist
RUNS_DIR="runs"
mkdir -p "$RUNS_DIR"

# Generate a timestamped run directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="$RUNS_DIR/run_$TIMESTAMP"
mkdir -p "$RUN_DIR"

RAW_OUTPUT_FILE="$RUN_DIR/live_raw_log.txt"

echo "========================================="
echo "🚀 Aria Gen 2 Python 3.12 Environment Active!"
echo "========================================="
echo "Starting the Live Gaze Stream Trigger 2..."
echo ""

echo "Pre-flight check: Verifying DinoV2 microservice is reachable..."
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" https://projectaria.hubscope.systems/health)

if [ "$HTTP_STATUS" -eq 200 ]; then
    echo "✅ Microservice is up and healthy!"
else
    echo "❌ Error: Microservice is unreachable or returned status $HTTP_STATUS."
    echo "Please ensure the dinov2_host docker container is running."
    exit 1
fi
echo ""
echo "Intent triggers will be displayed here in real-time."
echo ""
echo "The current run session data is being logged to:"
echo "   $RUN_DIR"
echo "-----------------------------------------"

# Execute the stream processing script
python3 live_gaze_trigger_2.py --run-dir "$RUN_DIR" "$@"
