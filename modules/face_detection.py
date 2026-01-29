"""Face Detection Module - Captures and detects faces from camera."""

import cv2
import numpy as np
from deepface import DeepFace
import config


class FaceDetector:
    """Handles camera capture and face detection."""

    def __init__(self, camera_index: int = config.CAMERA_INDEX):
        self.camera_index = camera_index
        self.cap = None

    def start_camera(self) -> bool:
        """Initialize camera capture. Returns True if successful."""
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        
        return self.cap.isOpened()

    def stop_camera(self) -> None:
        """Release camera resources."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

    def get_frame(self) -> np.ndarray | None:
        """Capture a single frame from camera. Returns frame or None."""
        if not self.cap:
            return None
        ret, frame = self.cap.read()
        
        return frame if ret else None

    def detect_face(self, frame: np.ndarray) -> dict | None:
        """
        Detect face in frame using DeepFace.
        Returns dict with 'face' (cropped), 'region' (x,y,w,h) or None.
        """
        try:
            faces = DeepFace.extract_faces(
                frame, 
                detector_backend=config.DETECTOR_BACKEND,
                enforce_detection=False
            )
            if faces and faces[0]['confidence'] > 0:
                return {
                    'face': faces[0]['face'],
                    'region': faces[0]['facial_area']
                }
        except Exception:
            pass
        
        return None

    def draw_bbox(self, frame: np.ndarray, region: dict, label: str = "") -> np.ndarray:
        """Draw bounding box and label on frame. Returns annotated frame."""
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        if label:
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame