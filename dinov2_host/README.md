# DinoV2 GPU Host Microservice

This microservice allows you to run the DinoV2 model on a Windows PC with an NVIDIA GPU and serve embeddings to your local network and remote partners.

## Quick Start (Docker)

The recommended way to run this microservice is using Docker Compose. This ensures the environment is fully contained and your NVIDIA GPU is properly exposed to the container.

1. Open PowerShell or a terminal.
2. Navigate to the `dinov2_host` directory.
3. Run the following command to build and start the container in the background:
   ```bash
   docker compose up -d --build
   ```
4. The server will start on `http://0.0.0.0:8000`. You can access the API docs locally at `http://localhost:8000/docs`.

*(Note: If you are on Windows, ensure Docker Desktop is running and WSL2 is enabled for GPU support).*

### API Usage

To get embeddings, make a `POST` request to `/embed` with one or more image files and your API key.

**Example using cURL (Multiple Images):**
```bash
curl -X POST "http://<YOUR_PC_IP>:8000/embed" \
     -H "X-API-Key: my_secure_api_key_123" \
     -F "files=@test_image1.jpg" \
     -F "files=@test_image2.jpg"
```

## Security & Remote Access (No Lateral Movement)

To allow partners to use your PC without opening ports on your home router (which prevents lateral movement):

3. **Cloudflare Tunnel (Included in Docker!):**
   - We have added the `cloudflared` tunnel directly into the `docker-compose.yml` file.
   - It uses a permanent named tunnel. The `TUNNEL_TOKEN` is securely stored in your `.env` file.
   - When you run `docker compose up -d`, the tunnel starts automatically and links to your custom Cloudflare domain.
   - The tunnel *only* exposes this single container, keeping your home network totally isolated.

2. **API Key Authentication:**
   - Change the `$env:DINOV2_API_KEY` in `run.ps1` to a secure, random string.
   - Share this key with your partners. The API will reject any requests without the correct `X-API-Key` header.
