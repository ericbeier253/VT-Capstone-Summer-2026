import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load .env file
from pathlib import Path
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("❌ ERROR: GEMINI_API_KEY not found in .env!")
    exit(1)

print("✅ Found GEMINI_API_KEY in .env!")
print("Attempting to hit Gemini API endpoint...")

# Initialize Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY)

# We'll use a very simple text prompt just to check if the endpoint is up and responding
prompt = "Respond with exactly one word: 'READY' if you are online and functional."

start_time = time.time()
try:
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )
    
    elapsed = time.time() - start_time
    print(f"\n🎉 SUCCESS! Gemini responded in {elapsed:.2f} seconds.")
    print(f"Response: {response.text}")
    
except Exception as e:
    print(f"\n❌ FAILED to connect to Gemini API.")
    print(f"Error details: {e}")
