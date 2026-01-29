"""Configuration settings for Face ID Recognition System."""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "database", "embeddings.db")

# Face Detection
DETECTOR_BACKEND = "opencv"  # Options: opencv, mtcnn, retinaface, ssd

# Face Recognition
RECOGNITION_MODEL = "Facenet"  # Options: VGG-Face, Facenet, Facenet512, ArcFace
DISTANCE_METRIC = "cosine"  # Options: cosine, euclidean, euclidean_l2
RECOGNITION_THRESHOLD = 0.40  # Lower = stricter matching

# Camera
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480