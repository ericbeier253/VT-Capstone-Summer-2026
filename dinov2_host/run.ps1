# Setup virtual environment and run the server
$ErrorActionPreference = "Stop"

# Create venv if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
}

# Activate venv
.\venv\Scripts\activate

# Install PyTorch with explicit CUDA 12.1 support first
Write-Host "Installing PyTorch for CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install the rest of the requirements
Write-Host "Installing remaining dependencies..."
pip install -r requirements.txt

Write-Host "Starting API on 0.0.0.0:8000..."
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
