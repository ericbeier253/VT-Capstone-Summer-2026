import math
from typing import Optional, Tuple

class GazeDwellTrigger:
    def __init__(self, radial_threshold_deg: float = 3.0, dwell_time_sec: float = 1.0):
        """
        Initializes the Gaze Dwell Trigger.
        
        Args:
            radial_threshold_deg (float): The maximum angular drift (in degrees) allowed before resetting the dwell timer.
            dwell_time_sec (float): The amount of continuous time (in seconds) the gaze must stay within the threshold to trigger.
        """
        self.radial_threshold_deg = radial_threshold_deg
        self.dwell_time_sec = dwell_time_sec
        
        self.center_vector: Optional[Tuple[float, float, float]] = None
        self.dwell_start_time_sec: Optional[float] = None
        
        # Debounce state to prevent multiple triggers for the same continuous stare
        self.is_triggered = False

    def _yaw_pitch_to_vector(self, yaw: float, pitch: float) -> Tuple[float, float, float]:
        """Converts yaw and pitch to a 3D unit vector."""
        vx = math.tan(yaw)
        vy = math.tan(pitch)
        vz = 1.0
        mag = math.sqrt(vx**2 + vy**2 + vz**2)
        return (vx / mag, vy / mag, vz / mag)

    def _angle_between(self, v1: Tuple[float, float, float], v2: Tuple[float, float, float]) -> float:
        """Calculates the angle in degrees between two 3D unit vectors."""
        dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
        dot = max(-1.0, min(1.0, dot))
        return math.acos(dot) * 180.0 / math.pi

    def process_gaze(self, yaw: float, pitch: float, timestamp_sec: float) -> bool:
        """
        Processes a new gaze data point and determines if a snapshot should be triggered.
        
        Args:
            yaw (float): Gaze yaw angle in radians.
            pitch (float): Gaze pitch angle in radians.
            timestamp_sec (float): Timestamp of the gaze sample in seconds.
            
        Returns:
            bool: True if a "silent shutter" snapshot is triggered on this frame, False otherwise.
        """
        current_vector = self._yaw_pitch_to_vector(yaw, pitch)

        if self.center_vector is None:
            self.center_vector = current_vector
            self.dwell_start_time_sec = timestamp_sec
            self.is_triggered = False
            return False

        angle_drift = self._angle_between(self.center_vector, current_vector)

        if angle_drift > self.radial_threshold_deg:
            # Gaze moved outside the boundary, reset the center and timer
            self.center_vector = current_vector
            self.dwell_start_time_sec = timestamp_sec
            self.is_triggered = False
            return False

        # If inside the boundary, check the elapsed dwell time
        elapsed_time = timestamp_sec - self.dwell_start_time_sec
        if elapsed_time >= self.dwell_time_sec:
            if not self.is_triggered:
                # The threshold has been met and we haven't triggered yet for this stare
                self.is_triggered = True
                return True
            else:
                # Already triggered for this dwell period, wait until they look away
                return False

        return False
