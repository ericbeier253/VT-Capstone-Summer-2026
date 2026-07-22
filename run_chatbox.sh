#!/bin/bash

echo "========================================="
echo "👁️ Starting Project Aria Chat Bot..."
echo "========================================="

# Run streamlit safely using the active python3 executable
python3 -m streamlit run chatbot_viewer.py "$@"
