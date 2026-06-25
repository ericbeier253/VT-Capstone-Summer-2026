#!/bin/bash

# Target the specific Python 3.12 virtual environment path
VENV_PATH="/Users/ericbeier/projectaria_gen2_python_env"

if [ -f "$VENV_PATH/bin/activate" ]; then
    echo "Activating Python 3.12 Virtual Environment at $VENV_PATH"
    source "$VENV_PATH/bin/activate"
else
    echo "Error: Python 3.12 Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Create a 'runs' directory if it doesn't exist
RUNS_DIR="runs"
mkdir -p "$RUNS_DIR"

# Generate a timestamped txt filename
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RAW_OUTPUT_FILE="$RUNS_DIR/live_raw_log_$TIMESTAMP.txt"

echo "========================================="
echo "🚀 Aria Gen 2 Python 3.12 Environment Active!"
echo "========================================="
echo "Starting the Live Gaze Stream Trigger..."
echo ""
echo "Intent triggers will be displayed here in real-time."
echo ""
echo "The raw eyegaze stream is being continuously logged to:"
echo "   $RAW_OUTPUT_FILE"
echo "-----------------------------------------"

# Execute the stream processing script
python3 live_gaze_trigger.py --raw-output "$RAW_OUTPUT_FILE"
