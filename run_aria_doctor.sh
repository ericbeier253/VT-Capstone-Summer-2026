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

echo "========================================="
echo "🩺 Running aria_doctor..."
echo "========================================="

aria_doctor "$@"
