import os
import time
import argparse
from google import genai

import aria.sdk_gen2 as sdk_gen2
import aria.stream_receiver as receiver

from storage_handler import StorageHandler
from enrichment_worker import AsyncEnrichmentWorker
from stream_callbacks import GazeSessionState

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True, help="Path to the output directory for this run session")
    parser.add_argument("--cloud", action="store_true", help="Use Google Cloud for storage")
    parser.add_argument("--local", action="store_true", help="Use local Postgres storage (default)")
    args = parser.parse_args()
    
    # Load .env file
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, val = line.strip().split('=', 1)
                    os.environ[key] = val.strip().strip('"').strip("'")

    # 1. Initialize Storage
    storage_mode = "cloud" if args.cloud else "local"
    gcp_project = os.environ.get("GCP_PROJECT")
    gcs_bucket = os.environ.get("GCS_BUCKET")
    storage_handler = StorageHandler(mode=storage_mode, gcp_project=gcp_project, gcs_bucket=gcs_bucket)
    
    # 2. Initialize Gemini & Worker
    api_key = os.environ.get("GEMINI_API_KEY")
    gemini_client = genai.Client(api_key=api_key) if api_key else None
    if not api_key:
        print("Warning: GEMINI_API_KEY not found in .env. LLM enrichment will be disabled.")
        
    enrichment_worker = AsyncEnrichmentWorker(storage_handler, gemini_client)
    enrichment_worker.start()
    
    # Setup Run Directory
    run_dir = args.run_dir
    os.makedirs(run_dir, exist_ok=True)
    raw_output = os.path.join(run_dir, "live_raw_log.txt")
    raw_file_handle = open(raw_output, "w")
    raw_file_handle.write("--- Starting Live Raw Eyegaze Logging ---\n")
    raw_file_handle.write("| Timestamp | Yaw | Pitch | Depth | Vergence | Valid Gaze? | Valid Spatial? |\n")
    raw_file_handle.write("|---|---|---|---|---|---|---|\n")
    raw_file_handle.flush()
    
    # Connect to device
    print("Connecting to Aria Gen2 device...")
    device_client = sdk_gen2.DeviceClient()
    config = sdk_gen2.DeviceClientConfig()
    device_client.set_client_config(config)
    
    try:
        device = device_client.connect()
    except Exception as e:
        print(f"Failed to connect to device: {e}")
        raw_file_handle.close()
        return

    # 3. Initialize Session State
    session_state = GazeSessionState(
        enrichment_worker=enrichment_worker,
        run_dir=run_dir,
        device=device,
        raw_file_handle=raw_file_handle
    )

    # Set up streaming
    print("Configuring streaming profile...")
    streaming_config = sdk_gen2.HttpStreamingConfig()
    streaming_config.profile_name = "profile9"
    streaming_config.streaming_interface = sdk_gen2.StreamingInterface.USB_NCM
    device.set_streaming_config(streaming_config)

    print("Starting streaming on device...")
    device.start_streaming()

    # Set up HTTP Server Receiver
    print("Starting local receiver server...")
    server_config = sdk_gen2.HttpServerConfig()
    server_config.address = "0.0.0.0"
    server_config.port = 6768
    
    stream_receiver = receiver.StreamReceiver(
        enable_image_decoding=True, 
        enable_raw_stream=False
    )
    stream_receiver.set_server_config(server_config)
    
    # Register callbacks from state
    stream_receiver.register_eye_gaze_callback(session_state.eyegaze_callback)
    stream_receiver.register_rgb_callback(session_state.image_callback)
    stream_receiver.register_device_calib_callback(session_state.device_calib_callback)
    
    stream_receiver.start_server()

    print(f"\n✅ Live streaming is active! Listening for eye gaze events...")
    print(f"   Writing run session logs and images to: {args.run_dir}")
    print("   Intents will be printed here in the console.")
    print("   Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping streaming and server...")
        print("(Note: The following 'Enqueued' logs refer to raw hardware packets, not LLM enrichment tasks)")
    finally:
        device.stop_streaming()
        stream_receiver.stop_server()
        raw_file_handle.write("--- Stopped Live Raw Eyegaze Logging ---\n")
        raw_file_handle.close()
        print("Cleanup complete.")

if __name__ == "__main__":
    main()
