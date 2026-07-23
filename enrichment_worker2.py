import queue
import threading
import json
from PIL import Image

class AsyncEnrichmentWorker:
    def __init__(self, storage_handler, gemini_client):
        self.storage_handler = storage_handler
        self.gemini_client = gemini_client
        self.llm_queue = queue.Queue()
        
    def start(self):
        worker_thread = threading.Thread(target=self._worker_thread, daemon=True)
        worker_thread.start()
        print("[Enrichment Worker] Background daemon started.")
        
    def enqueue(self, row_obj, run_id):
        self.llm_queue.put({"row_obj": row_obj, "run_id": run_id})
        
    def _worker_thread(self):
        while True:
            task = self.llm_queue.get()
            if task is None:
                break
            
            row_obj = task["row_obj"]
            run_id = task["run_id"]
            
            print(f"\n[Enrichment Worker] Processing capture from {row_obj.timestamp:.3f} s...")
            
            llm_analysis = None
            if self.gemini_client:
                try:
                    prompt = """
                    Analyze the image. The red crosshair indicates the user's gaze vector.
                    
                    You MUST return exactly one valid JSON object with meta information about the overall scene, and an itemized list of objects found in the image. Format it exactly like this:
                    {
                      "scene_meta": {
                        "description": "General description of the overall scene",
                        "environment": "Indoors, outdoors, office, kitchen, etc.",
                        "lighting": "Bright, dim, natural, artificial, etc."
                      },
                      "objects": [
                        {
                          "object_name": "Name of the object",
                          "object_description": "Detailed description of the object",
                          "object_location": "Contextual location of the object in the image",
                          "is_gaze_target": true
                        }
                      ]
                    }
                    """
                    image = Image.open(row_obj.img_path)
                    
                    response = self.gemini_client.models.generate_content(
                        model="gemini-3.5-flash-lite",
                        contents=[prompt, image]
                    )
                    text = response.text.replace("```json", "").replace("```", "").strip()
                    llm_analysis = json.loads(text)
                    print(f"[Enrichment Worker] Successfully enriched capture {row_obj.timestamp:.3f} s.")
                except Exception as e:
                    print(f"[Enrichment Worker] Error analyzing image with Gemini: {e}")
                    llm_analysis = {"error": str(e)}
            else:
                print("[Enrichment Worker] Gemini Client not initialized. Skipping enrichment.")
                
            # Post to storage handler
            if self.storage_handler:
                log = self.storage_handler.save_event(row_obj.timestamp, row_obj.depth, row_obj.img_path, run_id, llm_analysis)
                print(log, end='')
                
            self.llm_queue.task_done()
