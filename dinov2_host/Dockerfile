FROM python:3.10-slim

WORKDIR /app

# Install PyTorch with CUDA 12.1 support first
# We do this separately to ensure we pull from the correct index
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy the requirements file
COPY requirements.txt .

# Install the remaining dependencies
# (It's safe if torch is in requirements.txt, pip will see it's already satisfied by the cu121 version)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Run the FastAPI application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
