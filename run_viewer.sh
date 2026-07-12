#!/bin/bash

echo "========================================="
echo "👁️ Starting Project Aria Gaze Viewer..."
echo "========================================="

# Run streamlit safely using the active python3 executable
python3 -m streamlit run viewer.py "$@"
