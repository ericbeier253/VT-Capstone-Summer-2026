import sys
import os
import time
import argparse
import threading
from dataclasses import dataclass
import pickle
import requests

import aria.sdk_gen2 as sdk_gen2
import aria.stream_receiver as receiver
from projectaria_tools.core.mps import EyeGaze
from projectaria_tools.core.sensor_data import ImageData, ImageDataRecord
from PIL import Image, ImageDraw

from gaze_trigger import GazeDwellTrigger
from storage_handler import CloudStorageHandler

@dataclass
class GazeEventRow:
    timestamp: float
    depth: float
    img_path: str

# Global variables for the callback to use
raw_file_handle = None
trigger = None
trigger_count = 0
run_dir = None
rgb_cam_calib = None
cpf_to_rgb = None
storage_mode = "local"
cloud_storage_handler = None

# Image caching variables
latest_rgb_image = None
latest_rgb_timestamp = None
rgb_lock = threading.Lock()

def image_callback(image_data: ImageData, image_record: ImageDataRecord):
    global latest_rgb_image, latest_rgb_timestamp
    np_img = image_data.to_numpy_array()
    ts = image_record.capture_timestamp_ns / 1e9
    with rgb_lock:
        latest_rgb_image = np_img
        latest_rgb_timestamp = ts

def device_calib_callback(calibration):
    global rgb_cam_calib, cpf_to_rgb
    if calibration and rgb_cam_calib is None:
        try:
            rgb_cam_calib = calibration.get_camera_calib("camera-rgb")
            cpf_to_rgb = calibration.get_transform_cpf_sensor("camera-rgb")
            print("Successfully loaded device calibration from stream for gaze overlays.")
        except Exception as e:
            print(f"Error extracting camera calibration: {e}")

def eyegaze_callback(eyegaze_data: EyeGaze):
    global trigger_count
    
    # We use tracking_timestamp.total_seconds() to get the time in seconds
    timestamp_sec = eyegaze_data.tracking_timestamp.total_seconds()
    yaw = eyegaze_data.yaw
    pitch = eyegaze_data.pitch
    
    if raw_file_handle:
        raw_file_handle.write(f"| {timestamp_sec:.4f} | {yaw:.4f} | {pitch:.4f} | {eyegaze_data.depth:.4f} | {str(eyegaze_data.vergence)} | {eyegaze_data.combined_gaze_valid} | {eyegaze_data.spatial_gaze_point_valid} |\n")
        raw_file_handle.flush()
        
    # Only process if gaze is considered valid by the device
    if eyegaze_data.combined_gaze_valid:
        if trigger.process_gaze(yaw, pitch, timestamp_sec):
            trigger_count += 1
            
            # Save the adjacent image
            saved_img_path = ""
            with rgb_lock:
                if latest_rgb_image is not None and run_dir is not None:
                    try:
                        img = Image.fromarray(latest_rgb_image)
                        
                        # Draw gaze overlay if valid calibration exists
                        if eyegaze_data.spatial_gaze_point_valid and rgb_cam_calib is not None and cpf_to_rgb is not None:
                            # Transform gaze point to RGB camera frame
                            gaze_in_rgb_frame = cpf_to_rgb @ eyegaze_data.spatial_gaze_point_in_cpf
                            # Project 3D point to 2D image coordinates
                            pixel_point = rgb_cam_calib.project(gaze_in_rgb_frame)
                            
                            if pixel_point is not None:
                                x_pixel, y_pixel = pixel_point
                                draw = ImageDraw.Draw(img)
                                radius = 12
                                draw.ellipse((x_pixel - radius, y_pixel - radius, 
                                              x_pixel + radius, y_pixel + radius), 
                                              outline="red", width=3)
                                # Draw an inner dot
                                dot_radius = 2
                                draw.ellipse((x_pixel - dot_radius, y_pixel - dot_radius, 
                                              x_pixel + dot_radius, y_pixel + dot_radius), 
                                              fill="red")

                        filename = f"gaze_trigger_{trigger_count:03d}_{latest_rgb_timestamp:.3f}.jpg"
                        saved_img_path = os.path.join(run_dir, filename)
                        img.save(saved_img_path)
                    except Exception as e:
                        saved_img_path = f"Error saving: {e}"

            log_str = f"[TRIGGER {trigger_count:02d}] 📸 Intent captured at time {timestamp_sec:.3f} s | Gaze Vector -> Yaw: {yaw:.4f} rad, Pitch: {pitch:.4f} rad\n"
            
            row_obj = None
            if saved_img_path:
                if "Error" in saved_img_path:
                    log_str += f"   ⚠️ {saved_img_path}\n"
                else:
                    log_str += f"   🖼️  Saved image: {saved_img_path}\n"
                    
                    # Prepare the row object for the SQL database
                    row_obj = GazeEventRow(
                        timestamp=timestamp_sec,
                        depth=eyegaze_data.depth,
                        img_path=saved_img_path
                    )
                    
                    # Send POST request to FastAPI server or to GCP
                    if storage_mode == "cloud" and cloud_storage_handler:
                        run_id = os.path.basename(run_dir) if run_dir else None
                        log_str += cloud_storage_handler.save_event(row_obj.timestamp, row_obj.depth, row_obj.img_path, run_id)
                    else:
                        try:
                            response = requests.post(
                                "http://127.0.0.1:8000/insert",
                                json={
                                    "timestamp": row_obj.timestamp,
                                    "depth": row_obj.depth,
                                    "img_path": row_obj.img_path
                                },
                                timeout=2.0
                            )
                            if response.status_code == 200:
                                log_str += "   ✅ Successfully saved to local database\n"
                            else:
                                log_str += f"   ❌ DB Error: {response.text}\n"
                        except requests.exceptions.RequestException as e:
                            log_str += f"   ❌ Failed to connect to local DB: {e}\n"
                    
            print(log_str, end='')
            # You can now insert `row_obj` into your database if it is not None

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

    global raw_file_handle
    global trigger
    global run_dir
    global storage_mode
    global cloud_storage_handler
    
    if args.cloud:
        storage_mode = "cloud"
        gcp_project = os.environ.get("GCP_PROJECT")
        gcs_bucket = os.environ.get("GCS_BUCKET")
        cloud_storage_handler = CloudStorageHandler(gcp_project=gcp_project, gcs_bucket=gcs_bucket)
    else:
        storage_mode = "local"
    
    run_dir = args.run_dir
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        raw_output = os.path.join(run_dir, "live_raw_log.txt")
        raw_file_handle = open(raw_output, "w")
        raw_file_handle.write("--- Starting Live Raw Eyegaze Logging ---\n")
        raw_file_handle.write("| Timestamp | Yaw | Pitch | Depth | Vergence | Valid Gaze? | Valid Spatial? |\n")
        raw_file_handle.write("|---|---|---|---|---|---|---|\n")
        raw_file_handle.flush()
    
    # Initialize our Gaze Trigger
    trigger = GazeDwellTrigger(radial_threshold_deg=3.0, dwell_time_sec=1.0)
    
    # Connect to device
    print("Connecting to Aria Gen2 device...")
    device_client = sdk_gen2.DeviceClient()
    config = sdk_gen2.DeviceClientConfig()
    device_client.set_client_config(config)
    
    try:
        device = device_client.connect()
    except Exception as e:
        print(f"Failed to connect to device: {e}")
        if raw_file_handle:
            raw_file_handle.close()
        return

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
    
    # Enable image decoding to receive RGB stream
    stream_receiver = receiver.StreamReceiver(
        enable_image_decoding=True, 
        enable_raw_stream=False
    )
    stream_receiver.set_server_config(server_config)
    stream_receiver.register_eye_gaze_callback(eyegaze_callback)
    stream_receiver.register_rgb_callback(image_callback)
    stream_receiver.register_device_calib_callback(device_calib_callback)
    stream_receiver.start_server()

    print(f"\n✅ Live streaming is active! Listening for eye gaze events...")
    print(f"   Writing run session logs and images to: {args.run_dir}")
    print("   Intents will be printed here in the console.")
    print("   Press Ctrl+C to stop.\n")

    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping streaming and server...")
    finally:
        device.stop_streaming()
        stream_receiver.stop_server()
        if raw_file_handle:
            raw_file_handle.write("--- Stopped Live Raw Eyegaze Logging ---\n")
            raw_file_handle.close()
        print("Cleanup complete.")

if __name__ == "__main__":
    main()
