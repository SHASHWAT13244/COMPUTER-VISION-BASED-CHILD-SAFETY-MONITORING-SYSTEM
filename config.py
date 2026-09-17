# config.py
"""
Configuration file for Child Safety Monitoring System
"""

import os
import json
from pathlib import Path


class Config:
    """Configuration class for the entire system."""

    # ==================== Paths ====================
    BASE_DIR       = Path(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR       = os.path.join(BASE_DIR, 'data')
    MODELS_DIR     = os.path.join(BASE_DIR, 'saved_models')
    STATIC_DIR     = os.path.join(BASE_DIR, 'static')
    TEMPLATES_DIR  = os.path.join(BASE_DIR, 'templates')
    CAPTURES_DIR   = os.path.join(BASE_DIR, 'captures')
    RECORDINGS_DIR = os.path.join(BASE_DIR, 'recordings')
    UPLOADS_DIR    = os.path.join(STATIC_DIR, 'uploads')
    ALERTS_DIR     = os.path.join(BASE_DIR, 'alerts')
    ALERT_CONFIG_PATH       = os.path.join(BASE_DIR, 'alert_config.json')
    EVALUATION_RESULTS_PATH = os.path.join(BASE_DIR, 'evaluation_results.json')
    EVALUATION_PLOTS_PATH   = os.path.join(BASE_DIR, 'evaluation_plots.png')

    for _d in [DATA_DIR, MODELS_DIR, STATIC_DIR, TEMPLATES_DIR,
               CAPTURES_DIR, RECORDINGS_DIR, UPLOADS_DIR, ALERTS_DIR]:
        os.makedirs(_d, exist_ok=True)

    ACTIVITY_DATA_DIR = os.path.join(DATA_DIR, 'activities')
    ACTIVITY_CLASSES  = ['walking', 'running', 'sitting', 'falling', 'climbing']

    for _act in ACTIVITY_CLASSES:
        os.makedirs(os.path.join(ACTIVITY_DATA_DIR, _act), exist_ok=True)

    # ==================== Model Parameters ====================
    YOLO_MODEL           = 'yolov8n.pt'
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD        = 0.4

    POSE_MIN_DETECTION_CONFIDENCE = 0.5
    POSE_MIN_TRACKING_CONFIDENCE  = 0.5
    POSE_MODEL_COMPLEXITY         = 1

    SEQUENCE_LENGTH    = 30
    NUM_KEYPOINTS      = 33
    LSTM_HIDDEN_SIZE   = 128
    LSTM_NUM_LAYERS    = 2
    LSTM_DROPOUT       = 0.2
    LSTM_BIDIRECTIONAL = False

    # ==================== Training Parameters ====================
    EPOCHS           = 50
    BATCH_SIZE       = 32
    LEARNING_RATE    = 0.001
    TRAIN_TEST_SPLIT = 0.2
    RANDOM_SEED      = 42

    # ==================== Camera Settings ====================
    CAMERA_ID          = 0
    FRAME_WIDTH        = 640
    FRAME_HEIGHT       = 480
    FPS                = 30
    CAMERA_BUFFER_SIZE = 1

    # ==================== Real-Time Pipeline Tuning ====================
    DETECT_EVERY_N_FRAMES = 3
    POSE_EVERY_N_FRAMES   = 2
    GRAB_FLUSH_COUNT      = 3
    JPEG_QUALITY          = 70
    STREAM_MAX_FPS        = 30

    # ==================== Safety Rules ====================
    UNSAFE_ACTIVITIES        = ['falling', 'climbing']
    UNSAFE_ZONES             = ['kitchen', 'pool_area', 'stairs', 'balcony']
    FALL_DETECTION_THRESHOLD = 0.6
    MIN_FALL_FRAMES          = 5
    FALL_COOLDOWN_SECONDS    = 5

    # ==================== Alert Settings ====================
    ALERT_SOUND            = True
    ALERT_DISPLAY          = True
    ALERT_LOG              = True
    ALERT_COOLDOWN_SECONDS = 5
    MAX_ALERTS_PER_MINUTE  = 10

    # ==================== Performance Settings ====================
    ENABLE_PERFORMANCE_MONITORING = True
    PERFORMANCE_HISTORY_LENGTH    = 100
    ENABLE_GPU                    = True

    # ==================== Visualization Settings ====================
    SHOW_FPS         = True
    SHOW_INFO        = True
    SHOW_KEYPOINTS   = True
    SHOW_CONNECTIONS = True

    # ==================== Web Interface Settings ====================
    FLASK_HOST  = '0.0.0.0'
    FLASK_PORT  = 5000
    FLASK_DEBUG = False
    SECRET_KEY  = 'child-safety-monitoring-secret-key-2026'

    # ==================== Database (optional) ====================
    DATABASE_ENGINE   = 'sqlite'
    DATABASE_NAME     = 'child_safety.db'
    DATABASE_USER     = ''
    DATABASE_PASSWORD = ''
    DATABASE_HOST     = 'localhost'
    DATABASE_PORT     = 5432

    # ==================== Helpers ====================
    @classmethod
    def get_model_path(cls, model_name):
        return os.path.join(cls.MODELS_DIR, model_name)

    @classmethod
    def get_activity_label(cls, index):
        return (cls.ACTIVITY_CLASSES[index]
                if 0 <= index < len(cls.ACTIVITY_CLASSES) else 'unknown')

    @classmethod
    def get_activity_index(cls, label):
        return cls.ACTIVITY_CLASSES.index(label) if label in cls.ACTIVITY_CLASSES else -1

    @classmethod
    def get_unsafe_activities(cls):
        return cls.UNSAFE_ACTIVITIES

    @classmethod
    def get_project_path(cls, *parts):
        return os.path.join(cls.BASE_DIR, *parts)

    @classmethod
    def to_dict(cls):
        out = {}
        for klass in reversed(cls.__mro__):
            for k, v in vars(klass).items():
                if k.startswith('_') or callable(v) or isinstance(v, classmethod):
                    continue
                if isinstance(v, Path):
                    out[k] = str(v)
                elif isinstance(v, (str, int, float, bool, list, dict, tuple, type(None))):
                    out[k] = v
                else:
                    try:
                        json.dumps(v)
                        out[k] = v
                    except Exception:
                        continue
        return out

    @classmethod
    def from_dict(cls, config_dict):
        for k, v in config_dict.items():
            if hasattr(cls, k):
                setattr(cls, k, v)

    @classmethod
    def validate(cls):
        """Validate critical numeric config values to prevent div-by-zero etc."""
        def _positive_int(name):
            v = getattr(cls, name, None)
            if not isinstance(v, int) or v < 1:
                raise ValueError(
                    f"Config.{name} must be a positive integer, got {v!r}")

        _positive_int('DETECT_EVERY_N_FRAMES')
        _positive_int('POSE_EVERY_N_FRAMES')
        _positive_int('GRAB_FLUSH_COUNT')
        _positive_int('SEQUENCE_LENGTH')
        _positive_int('NUM_KEYPOINTS')

        if cls.STREAM_MAX_FPS < 1:
            raise ValueError("Config.STREAM_MAX_FPS must be >= 1")
        if not (0.0 < cls.CONFIDENCE_THRESHOLD <= 1.0):
            raise ValueError("Config.CONFIDENCE_THRESHOLD must be in (0, 1]")
        if not (0.0 < cls.IOU_THRESHOLD <= 1.0):
            raise ValueError("Config.IOU_THRESHOLD must be in (0, 1]")
        return True


Config.validate()


class DevelopmentConfig(Config):
    FLASK_DEBUG = True
    ENABLE_GPU  = False
    FRAME_WIDTH  = 480
    FRAME_HEIGHT = 360
    FPS          = 24
    DETECT_EVERY_N_FRAMES = 3
    POSE_EVERY_N_FRAMES   = 2


class ProductionConfig(Config):
    FLASK_DEBUG  = False
    ENABLE_GPU   = True
    FRAME_WIDTH  = 640
    FRAME_HEIGHT = 480
    FPS          = 30
    DETECT_EVERY_N_FRAMES = 2
    POSE_EVERY_N_FRAMES   = 1


class TestConfig(Config):
    FLASK_DEBUG  = False
    ENABLE_GPU   = False
    FRAME_WIDTH  = 160
    FRAME_HEIGHT = 120
    FPS          = 5
    EPOCHS       = 2
    BATCH_SIZE   = 8


for _cls in (DevelopmentConfig, ProductionConfig, TestConfig):
    _cls.validate()


def get_config(env='development'):
    return {
        'development': DevelopmentConfig,
        'production':  ProductionConfig,
        'test':        TestConfig,
    }.get(env, Config)
