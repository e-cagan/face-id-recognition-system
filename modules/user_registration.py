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
        # TODO: 
        # 1. Check if user_id exists
        # 2. Capture frame and detect face
        # 3. Extract embedding
        # 4. Save to database
        pass

    def register_from_image(
        self, user_id: str, name: str, image: np.ndarray
    ) -> tuple[bool, str]:
        """
        Register user from provided image instead of camera.
        Returns (success: bool, message: str).
        """
        # TODO: Same as register_user but use provided image
        pass