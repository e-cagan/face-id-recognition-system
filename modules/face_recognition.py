"""Face Recognition Module - Embedding extraction and comparison."""

import numpy as np
from deepface import DeepFace
import config


class FaceRecognizer:
    """Handles facial embedding extraction and identity matching."""

    def __init__(
        self,
        model_name: str = config.RECOGNITION_MODEL,
        distance_metric: str = config.DISTANCE_METRIC,
        threshold: float = config.RECOGNITION_THRESHOLD,
    ):
        self.model_name = model_name
        self.distance_metric = distance_metric
        self.threshold = threshold

    def extract_embedding(self, face_img: np.ndarray) -> np.ndarray | None:
        """
        Extract embedding vector from face image.
        Returns 1D numpy array or None if failed.
        """
        # TODO: Use DeepFace.represent() with self.model_name
        # Return the embedding vector
        pass

    def calculate_distance(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Calculate distance between two embeddings using configured metric."""
        # TODO: Implement cosine/euclidean distance calculation
        pass

    def find_match(
        self, embedding: np.ndarray, stored_embeddings: list[tuple[str, np.ndarray]]
    ) -> tuple[str, float] | None:
        """
        Find best match from stored embeddings.
        Args:
            embedding: Query embedding
            stored_embeddings: List of (user_id, embedding) tuples
        Returns:
            (user_id, distance) if match found below threshold, else None.
        """
        # TODO: Compare with all stored embeddings, return best match
        pass