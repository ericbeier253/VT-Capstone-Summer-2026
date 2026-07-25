from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status, Header
from fastapi.security import APIKeyHeader
import os
from dotenv import load_dotenv
from embedder import DinoEmbedder

load_dotenv()

app = FastAPI(title="DinoV2 Embedding Host", description="Microservice for extracting DinoV2 embeddings.")

# Get API key from environment variable, default to something for ease of use but warn
API_KEY = os.getenv("DINOV2_API_KEY", "your_super_secret_api_key_here")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key_header: str = Depends(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN, detail="Could not validate API Key"
    )

embedder = None

@app.on_event("startup")
async def startup_event():
    global embedder
    print("Loading DinoV2 Model onto GPU (if available)...")
    embedder = DinoEmbedder()
    print("Model loaded successfully!")

@app.post("/embed", dependencies=[Depends(get_api_key)])
async def create_embedding(files: list[UploadFile] = File(...)):
    for file in files:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not an image.")
    
    contents_list = [await file.read() for file in files]
    try:
        embeddings = embedder.embed_image_bytes_batch(contents_list)
        return {
            "embeddings": [
                {"filename": f.filename, "embedding": emb} 
                for f, emb in zip(files, embeddings)
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}
