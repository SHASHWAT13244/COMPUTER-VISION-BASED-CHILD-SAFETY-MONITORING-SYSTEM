# config.py
"""
Configuration file for Child Safety Monitoring System
Contains all configurable parameters and paths
"""

import os
from pathlib import Path


class Config:
    """
    Configuration class for the entire system
    """
    
    # ==================== Paths ====================
    BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
    STATIC_DIR = os.path.join(BASE_DIR, 'static')
    TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
    CAPTURES_DIR = os.path.join(BASE_DIR, 'captures')
    RECORDINGS_DIR = os.path.join(BASE_DIR, 'recordings')
    
    # Create directories if they don't exist
    for dir_path in [DATA_DIR, MODELS_DIR, STATIC_DIR, TEMPLATES_DIR, 
                     CAPTURES_DIR, RECORDINGS_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Activity data directories
    ACTIVITY_DATA_DIR = os.path.join(DATA_DIR, 'activities')
    ACTIVITY_CLASSES = ['walking', 'running', 'sitting', 'falling', 'climbing']
    
    # Create activity data directories
    for activity in ACTIVITY_CLASSES:
        os.makedirs(os.path.join(ACTIVITY_DATA_DIR, activity), exist_ok=True)
    
    # ==================== Model Parameters ====================
    # YOLO Model
    YOLO_MODEL = 'yolov8n.pt'  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.4
    
    # MediaPipe Pose
    POSE_MIN_DETECTION_CONFIDENCE = 0.5
    POSE_MIN_TRACKING_CONFIDENCE = 0.5
    POSE_MODEL_COMPLEXITY = 1  # 0, 1, or 2
    
    # Activity Recognition (LSTM)
    SEQUENCE_LENGTH = 30  # Number of frames per sequence
    NUM_KEYPOINTS = 33  # MediaPipe pose keypoints
    LSTM_HIDDEN_SIZE = 128
    LSTM_NUM_LAYERS = 2
    LSTM_DROPOUT = 0.2
    LSTM_BIDIRECTIONAL = False
    
    # ==================== Training Parameters ====================
    EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    TRAIN_TEST_SPLIT = 0.2
    RANDOM_SEED = 42
    
    # ==================== Camera Settings ====================
    CAMERA_ID = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FPS = 30
    
    # ==================== Safety Rules ====================
    UNSAFE_ACTIVITIES = ['falling', 'climbing']
    UNSAFE_ZONES = ['kitchen', 'pool_area', 'stairs', 'balcony']
    
    # Fall detection
    FALL_DETECTION_THRESHOLD = 0.6
    MIN_FALL_FRAMES = 5
    FALL_COOLDOWN_SECONDS = 5
    
    # ==================== Alert Settings ====================
    ALERT_SOUND = True
    ALERT_DISPLAY = True
    ALERT_LOG = True
    ALERT_COOLDOWN_SECONDS = 5
    MAX_ALERTS_PER_MINUTE = 10
    
    # ==================== Performance Settings ====================
    ENABLE_PERFORMANCE_MONITORING = True
    PERFORMANCE_HISTORY_LENGTH = 100
    ENABLE_GPU = True  # Auto-detect if available
    
    # ==================== Visualization Settings ====================
    SHOW_FPS = True
    SHOW_INFO = True
    SHOW_KEYPOINTS = True
    SHOW_CONNECTIONS = True
    FONT = cv2.FONT_HERSHEY_SIMPLEX if 'cv2' in dir() else None
    
    # ==================== Web Interface Settings ====================
    FLASK_HOST = '0.0.0.0'
    FLASK_PORT = 5000
    FLASK_DEBUG = False
    SECRET_KEY = 'child-safety-monitoring-secret-key-2026'
    
    # ==================== Database Settings (optional) ====================
    DATABASE_ENGINE = 'sqlite'  # Options: sqlite, postgresql, mysql
    DATABASE_NAME = 'child_safety.db'
    DATABASE_USER = ''
    DATABASE_PASSWORD = ''
    DATABASE_HOST = 'localhost'
    DATABASE_PORT = 5432
    
    # ==================== Model Registry ====================
    @classmethod
    def get_model_path(cls, model_name):
        """Get path for a specific model"""
        return os.path.join(cls.MODELS_DIR, model_name)
    
    @classmethod
    def get_activity_label(cls, index):
        """Get activity label by index"""
        return cls.ACTIVITY_CLASSES[index] if index < len(cls.ACTIVITY_CLASSES) else 'unknown'
    
    @classmethod
    def get_activity_index(cls, label):
        """Get activity index by label"""
        return cls.ACTIVITY_CLASSES.index(label) if label in cls.ACTIVITY_CLASSES else -1
    
    @classmethod
    def get_unsafe_activities(cls):
        """Get list of unsafe activities"""
        return cls.UNSAFE_ACTIVITIES
    
    @classmethod
    def to_dict(cls):
        """Convert config to dictionary"""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """Update config from dictionary"""
        for key, value in config_dict.items():
            if hasattr(cls, key):
                setattr(cls, key, value)


# Development configuration
class DevelopmentConfig(Config):
    """Development-specific configuration"""
    FLASK_DEBUG = True
    ENABLE_GPU = False  # Use CPU for development
    FRAME_WIDTH = 320  # Lower resolution for faster processing
    FRAME_HEIGHT = 240
    FPS = 15


# Production configuration
class ProductionConfig(Config):
    """Production-specific configuration"""
    FLASK_DEBUG = False
    ENABLE_GPU = True
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FPS = 30


# Test configuration
class TestConfig(Config):
    """Test-specific configuration"""
    FLASK_DEBUG = False
    ENABLE_GPU = False
    FRAME_WIDTH = 160
    FRAME_HEIGHT = 120
    FPS = 5
    EPOCHS = 2  # Quick training for tests
    BATCH_SIZE = 8


def get_config(env='development'):
    """
    Get configuration based on environment
    
    Args:
        env: 'development', 'production', or 'test'
        
    Returns:
        Config class
    """
    configs = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'test': TestConfig
    }
    return configs.get(env, Config)