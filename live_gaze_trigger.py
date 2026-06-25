import sys
import time
import argparse

import aria.sdk_gen2 as sdk_gen2
import aria.stream_receiver as receiver
from projectaria_tools.core.mps import EyeGaze

from gaze_trigger import GazeDwellTrigger

# Global variables for the callback to use
raw_file_handle = None
trigger = None
trigger_count = 0

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
            log_str = f"[TRIGGER {trigger_count:02d}] 📸 Intent captured at time {timestamp_sec:.3f} s | Gaze Vector -> Yaw: {yaw:.4f} rad, Pitch: {pitch:.4f} rad\n"
            print(log_str, end='')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-output", type=str, required=True, help="Path to the output file for raw data logging")
    args = parser.parse_args()

    global raw_file_handle
    global trigger
    
    if args.raw_output:
        raw_file_handle = open(args.raw_output, "w")
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
        out_file_handle.close()
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
    
    # Enable only raw stream properties if image decoding isn't needed here
    stream_receiver = receiver.StreamReceiver(
        enable_image_decoding=False, 
        enable_raw_stream=False
    )
    stream_receiver.set_server_config(server_config)
    stream_receiver.register_eye_gaze_callback(eyegaze_callback)
    stream_receiver.start_server()

    print(f"\n✅ Live streaming is active! Listening for eye gaze events...")
    print(f"   Writing live raw logs to: {args.raw_output}")
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
