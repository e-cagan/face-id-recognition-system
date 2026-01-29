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
        try:
            result = DeepFace.represent(
                img_path=face_img,
                model_name=self.model_name,
                enforce_detection=False
            )
            if result:
                return np.array(result[0]['embedding'], dtype=np.float64)
        except Exception:
            pass
        
        return None

    def calculate_distance(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Calculate distance between two embeddings using configured metric."""
        if self.distance_metric == "cosine":
            # Cosine distance = 1 - cosine_similarity
            dot = np.dot(embedding1, embedding2)
            norm = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            return 1 - (dot / norm)
        elif self.distance_metric == "euclidean":
            return np.linalg.norm(embedding1 - embedding2)
        elif self.distance_metric == "euclidean_l2":
            # Normalize then euclidean
            e1 = embedding1 / np.linalg.norm(embedding1)
            e2 = embedding2 / np.linalg.norm(embedding2)
            return np.linalg.norm(e1 - e2)
        
        return float('inf')

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
        best_match = None
        best_distance = float('inf')

        for user_id, stored_emb in stored_embeddings:
            distance = self.calculate_distance(embedding, stored_emb)
            if distance < best_distance:
                best_distance = distance
                best_match = user_id

        if best_distance < self.threshold:
            return (best_match, best_distance)
        
        return None