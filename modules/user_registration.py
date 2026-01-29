"""User Registration Module - Handles new user enrollment."""

import numpy as np
from .face_detection import FaceDetector
from .face_recognition import FaceRecognizer
from .data_manager import DataManager


class UserRegistration:
    """Coordinates face capture, embedding extraction, and storage for registration."""

    def __init__(
        self,
        detector: FaceDetector,
        recognizer: FaceRecognizer,
        data_manager: DataManager,
    ):
        self.detector = detector
        self.recognizer = recognizer
        self.data_manager = data_manager

    def register_user(self, user_id: str, name: str) -> tuple[bool, str]:
        """
        Register new user by capturing face from camera.
        Returns (success: bool, message: str).
        """
        # Check if user already exists
        if self.data_manager.user_exists(user_id):
            return (False, "User ID already exists.")

        # Capture frame
        frame = self.detector.get_frame()
        if frame is None:
            return (False, "Failed to capture frame from camera.")

        # Detect face
        face_data = self.detector.detect_face(frame)
        if face_data is None:
            return (False, "No face detected in frame.")

        # Extract embedding
        embedding = self.recognizer.extract_embedding(frame)
        if embedding is None:
            return (False, "Failed to extract face embedding.")

        # Save to database
        if self.data_manager.add_user(user_id, name, embedding):
            return (True, f"User '{name}' registered successfully.")
        
        return (False, "Failed to save user to database.")

    def register_from_image(
        self, user_id: str, name: str, image: np.ndarray
    ) -> tuple[bool, str]:
        """
        Register user from provided image instead of camera.
        Returns (success: bool, message: str).
        """
        # Check if user already exists
        if self.data_manager.user_exists(user_id):
            return (False, "User ID already exists.")

        # Detect face
        face_data = self.detector.detect_face(image)
        if face_data is None:
            return (False, "No face detected in image.")

        # Extract embedding
        embedding = self.recognizer.extract_embedding(image)
        if embedding is None:
            return (False, "Failed to extract face embedding.")

        # Save to database
        if self.data_manager.add_user(user_id, name, embedding):
            return (True, f"User '{name}' registered successfully.")
        
        return (False, "Failed to save user to database.")