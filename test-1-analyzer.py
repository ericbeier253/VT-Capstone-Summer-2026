import os
from google import genai
from vision import GeminiAnalyzer

# Load .env file
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, val = line.strip().split('=', 1)
                os.environ[key] = val.strip().strip('"').strip("'")
                #print(key, ":", val.strip().strip('"').strip("'"))

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

analyzer = GeminiAnalyzer(client)

analysis = analyzer.analyze("test.jpg")

print(analysis.model_dump_json(indent=2))