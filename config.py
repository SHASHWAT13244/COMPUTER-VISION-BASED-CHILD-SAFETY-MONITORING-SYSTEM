"""
Configuration file for the Child Safety Monitoring System
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

# Create directories if they don't exist
for dir_path in [MODELS_DIR, DATA_DIR, OUTPUT_DIR]:
    dir_path.mkdir(exist_ok=True)

# Model configurations
YOLO_MODEL = "yolov8n.pt"  # Using nano model for better performance
YOLO_MODEL_PATH = MODELS_DIR / YOLO_MODEL
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# Pose estimation
POSE_DETECTION_CONFIDENCE = 0.5
POSE_TRACKING_CONFIDENCE = 0.5

# Activity recognition
ACTIVITY_CLASSES = ["walking", "running", "sitting", "standing", "falling"]
SEQUENCE_LENGTH = 30  # Number of frames for activity recognition
FEATURES_PER_FRAME = 33 * 3  # 33 keypoints * 3 coordinates (x, y, z)

# Safety rules
SAFETY_RULES = {
    "fall_detection": True,
    "running_detection": True,
    "boundary_detection": True,
    "climbing_detection": True,
    "restricted_zones": [
        {"x": [0, 0.3], "y": [0, 0.3]},  # Top-left zone
        {"x": [0.7, 1.0], "y": [0.7, 1.0]}  # Bottom-right zone
    ]
}

# Alert configurations
ALERT_CONFIG = {
    "enable_console": True,
    "enable_email": False,
    "enable_sms": False,
    "email_recipient": "caregiver@example.com",
    "sms_recipient": "+1234567890",
    "cooldown_seconds": 10  # Minimum time between alerts for same event
}

# Video processing
VIDEO_SETTINGS = {
    "camera_index": 0,  # 0 for webcam, or path to video file
    "frame_width": 640,
    "frame_height": 480,
    "fps": 30,
    "record_output": True,
    "output_video_path": OUTPUT_DIR / "monitoring_output.avi"
}

# System settings
SYSTEM_SETTINGS = {
    "use_gpu": False,  # Set to True if CUDA is available
    "log_level": "INFO",
    "log_file": OUTPUT_DIR / "monitoring.log",
    "debug_mode": True
}