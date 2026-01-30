# Face ID Recognition System

**Istanbul Okan University - Computer Engineering Graduation Project**

Student: Emin Çağan Apaydın (220212054)  
Supervisor: Dr. Emel Koç

## Overview

A real-time face recognition system for identity verification using deep learning. The system captures facial images through a webcam, extracts facial embeddings, and verifies user identity by comparing against stored data.

## Features

- Real-time face detection and recognition
- User registration with facial data
- Identity verification
- Local SQLite database storage
- Simple Tkinter GUI

## Project Structure

```
face-id-system/
├── config.py              # Configuration settings
├── main.py                # GUI application entry point
├── requirements.txt       # Dependencies
├── database/
│   └── embeddings.db   # SQLite database (created at runtime)
└── modules/
    ├── __init__.py
    ├── detection.py       # Face detection (OpenCV + DeepFace)
    ├── recognition.py     # Embedding extraction and matching
    ├── registration.py    # User enrollment workflow
    └── data_manager.py    # SQLite CRUD operations
```

## Requirements

- Python 3.10+
- Webcam

## Installation

1. Clone or download the project:
```bash
cd face-id-system
```

2. Create virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python main.py
```

### Register a New User

1. Enter User ID and Name in the input fields
2. Click **Register**
3. Position your face in front of the camera
4. Click **Capture** to save

### Verify Identity

1. Click **Verify**
2. Position your face in front of the camera
3. Click **Capture** to verify

### Manage Users

- Click **Refresh** to update the user list
- Select a user and click **Delete Selected** to remove

## Configuration

Edit `config.py` to customize settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DETECTOR_BACKEND` | `opencv` | Face detector: opencv, mtcnn, retinaface, ssd |
| `RECOGNITION_MODEL` | `Facenet` | Model: VGG-Face, Facenet, Facenet512, ArcFace |
| `DISTANCE_METRIC` | `cosine` | Metric: cosine, euclidean, euclidean_l2 |
| `RECOGNITION_THRESHOLD` | `0.40` | Match threshold (lower = stricter) |
| `CAMERA_INDEX` | `0` | Camera device index |

## Technologies

- **Python** - Core programming language
- **OpenCV** - Camera capture and image processing
- **DeepFace** - Face detection and embedding extraction
- **NumPy** - Numerical operations
- **SQLite** - Local database storage
- **Tkinter** - GUI framework
- **Pillow** - Image handling for GUI

## Module Description

### Face Detection Module (`face_detection.py`)
Handles camera initialization, frame capture, face detection, and bounding box visualization.

### Face Recognition Module (`face_recognition.py`)
Extracts facial embeddings using deep learning models and performs identity matching through distance calculation.

### User Registration Module (`user_registration.py`)
Coordinates the registration workflow: face capture, embedding extraction, and database storage.

### Data Management Module (`data_manager.py`)
Manages SQLite database operations: user CRUD, embedding storage and retrieval.

## Notes

- All data is stored locally for privacy
- First run may take longer as DeepFace downloads model weights
- GPU is optional; system works on CPU
- TensorFlow warnings about CUDA can be ignored if no GPU is available

## License

This project is developed for academic purposes as part of a graduation project.