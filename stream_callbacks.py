import os
import threading
from dataclasses import dataclass
from PIL import Image, ImageDraw

from projectaria_tools.core.mps import EyeGaze
from projectaria_tools.core.sensor_data import ImageData, ImageDataRecord
from gaze_trigger import GazeDwellTrigger

@dataclass
class GazeEventRow:
    timestamp: float
    depth: float
    img_path: str

class GazeSessionState:
    def __init__(self, enrichment_worker, run_dir, device, raw_file_handle):
        self.enrichment_worker = enrichment_worker
        self.run_dir = run_dir
        self.device = device
        self.raw_file_handle = raw_file_handle
        
        self.trigger = GazeDwellTrigger(radial_threshold_deg=3.0, dwell_time_sec=1.0)
        self.trigger_count = 0
        
        self.latest_rgb_image = None
        self.latest_rgb_timestamp = None
        self.rgb_lock = threading.Lock()
        
        self.rgb_cam_calib = None
        self.cpf_to_rgb = None

    def image_callback(self, image_data: ImageData, image_record: ImageDataRecord):
        np_img = image_data.to_numpy_array()
        ts = image_record.capture_timestamp_ns / 1e9
        with self.rgb_lock:
            self.latest_rgb_image = np_img
            self.latest_rgb_timestamp = ts

    def device_calib_callback(self, calibration):
        if calibration and self.rgb_cam_calib is None:
            try:
                self.rgb_cam_calib = calibration.get_camera_calib("camera-rgb")
                self.cpf_to_rgb = calibration.get_transform_cpf_sensor("camera-rgb")
                print("Successfully loaded device calibration from stream for gaze overlays.")
            except Exception as e:
                print(f"Error extracting camera calibration: {e}")

    def eyegaze_callback(self, eyegaze_data: EyeGaze):
        # We use tracking_timestamp.total_seconds() to get the time in seconds
        timestamp_sec = eyegaze_data.tracking_timestamp.total_seconds()
        yaw = eyegaze_data.yaw
        pitch = eyegaze_data.pitch
        
        if self.raw_file_handle:
            self.raw_file_handle.write(f"| {timestamp_sec:.4f} | {yaw:.4f} | {pitch:.4f} | {eyegaze_data.depth:.4f} | {str(eyegaze_data.vergence)} | {eyegaze_data.combined_gaze_valid} | {eyegaze_data.spatial_gaze_point_valid} |\n")
            self.raw_file_handle.flush()
            
        # Only process if gaze is considered valid by the device
        if eyegaze_data.combined_gaze_valid:
            if self.trigger.process_gaze(yaw, pitch, timestamp_sec):
                self.trigger_count += 1
                
                # Save the adjacent image
                saved_img_path = ""
                with self.rgb_lock:
                    if self.latest_rgb_image is not None and self.run_dir is not None:
                        try:
                            img = Image.fromarray(self.latest_rgb_image)
                            
                            # Draw gaze overlay if valid calibration exists
                            if eyegaze_data.spatial_gaze_point_valid and self.rgb_cam_calib is not None and self.cpf_to_rgb is not None:
                                # Transform gaze point to RGB camera frame
                                gaze_in_rgb_frame = self.cpf_to_rgb @ eyegaze_data.spatial_gaze_point_in_cpf
                                # Project 3D point to 2D image coordinates
                                pixel_point = self.rgb_cam_calib.project(gaze_in_rgb_frame)
                                
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

                            filename = f"gaze_trigger_{self.trigger_count:03d}_{self.latest_rgb_timestamp:.3f}.jpg"
                            saved_img_path = os.path.join(self.run_dir, filename)
                            img.save(saved_img_path)
                        except Exception as e:
                            saved_img_path = f"Error saving: {e}"

                log_str = f"[TRIGGER {self.trigger_count:02d}] 📸 Intent captured at time {timestamp_sec:.3f} s | Gaze Vector -> Yaw: {yaw:.4f} rad, Pitch: {pitch:.4f} rad\n"
                
                if self.device:
                    try:
                        self.device.render_tts("beep")
                    except Exception as e:
                        log_str += f"   ⚠️ Audio Error: {e}\n"
                
                row_obj = None
                if saved_img_path:
                    if "Error" in saved_img_path:
                        log_str += f"   ⚠️ {saved_img_path}\n"
                    else:
                        log_str += f"   🖼️  Saved image: {saved_img_path}\n"
                        
                        # Prepare the row object
                        row_obj = GazeEventRow(
                            timestamp=timestamp_sec,
                            depth=eyegaze_data.depth,
                            img_path=saved_img_path
                        )
                        
                        # Queue for background processing (storage handler handles local vs cloud internally)
                        run_id = os.path.basename(self.run_dir) if self.run_dir else None
                        if self.enrichment_worker:
                            self.enrichment_worker.enqueue(row_obj, run_id)
                            log_str += "   📥 Queued for background processing.\n"
                        else:
                            log_str += "   ⚠️ No worker to process capture.\n"
                        
                print(log_str, end='')
