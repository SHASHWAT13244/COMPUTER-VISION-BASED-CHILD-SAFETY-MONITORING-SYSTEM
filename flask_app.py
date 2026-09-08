# flask_app.py
"""
Flask Web Application for Child Safety Monitoring System
Provides web interface for real-time monitoring, dashboard, and alerts management
"""

import os
import sys
import cv2
import time
import json
import base64
import threading
import queue
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image
import io
import webbrowser
import socket

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, Response, request, jsonify, send_file, url_for
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import eventlet

from config import Config
from models.detector import ChildDetector
from models.pose_estimator import PoseEstimator
from models.activity_recognizer import ActivityRecognizer
from models.safety_engine import SafetyEngine
from models.tracker import PersonTracker
from utils.alert import AlertSystem
from utils.alert_advanced import AdvancedAlertSystem
from utils.visualization import Visualizer
from utils.performance_monitor import PerformanceMonitor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = Config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Global variables
camera = None
monitoring_active = False
frame_queue = queue.Queue(maxsize=10)
alert_queue = queue.Queue(maxsize=50)
current_status = {
    'activity': 'None',
    'confidence': 0.0,
    'safe': True,
    'alerts': [],
    'fps': 0,
    'frame_count': 0
}

# Initialize components
detector = ChildDetector(
    model_path=Config.YOLO_MODEL,
    conf_threshold=Config.CONFIDENCE_THRESHOLD,
    iou_threshold=Config.IOU_THRESHOLD
)

pose_estimator = PoseEstimator(
    min_detection_confidence=Config.POSE_MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=Config.POSE_MIN_TRACKING_CONFIDENCE,
    model_complexity=Config.POSE_MODEL_COMPLEXITY
)

activity_recognizer = ActivityRecognizer(
    sequence_length=Config.SEQUENCE_LENGTH,
    num_keypoints=33,
    num_classes=len(Config.ACTIVITY_CLASSES),
    hidden_size=Config.LSTM_HIDDEN_SIZE,
    num_layers=Config.LSTM_NUM_LAYERS,
    dropout=Config.LSTM_DROPOUT,
    bidirectional=Config.LSTM_BIDIRECTIONAL
)

safety_engine = SafetyEngine()
tracker = PersonTracker(max_lost_frames=10, min_confidence=0.5)
alert_system = AlertSystem(sound_enabled=False, display_enabled=True, log_enabled=True)
advanced_alert = AdvancedAlertSystem('alert_config.json')
visualizer = Visualizer(show_fps=True, show_info=True, show_activity=True)
perf_monitor = PerformanceMonitor()

# Load pre-trained model if exists
def load_pretrained_model():
    """Load pre-trained activity recognition model"""
    model_paths = [
        os.path.join(Config.MODELS_DIR, 'activity_model.pth'),
        os.path.join(Config.MODELS_DIR, 'best_activity_model.pth'),
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                activity_recognizer.load_model(path)
                logger.info(f"Loaded pre-trained model from {path}")
                return True
            except Exception as e:
                logger.error(f"Error loading model from {path}: {e}")
    
    logger.info("No pre-trained model found. Using random weights.")
    return False

load_pretrained_model()

# Start performance monitoring
perf_monitor.start_monitoring()

# Global processing variables
keypoint_buffer = []
alerts_list = []
frame_count = 0
start_time = datetime.now()
processing_lock = threading.Lock()


# ==================== SYNTHETIC SEQUENCE GENERATOR ====================

def create_synthetic_sequence(keypoints, seq_length=30):
    """
    Create a synthetic sequence from a single frame's keypoints.
    Simulates realistic movement patterns for activity recognition.
    
    Args:
        keypoints: Single frame keypoints (33, 3) or (33, 4)
        seq_length: Number of frames to generate (default: 30)
        
    Returns:
        numpy array: (seq_length, 99) - 99 = 33 keypoints × 3 coordinates
    """
    if keypoints is None:
        return None
    
    # Use x, y, z coordinates
    if keypoints.shape[1] >= 3:
        base = keypoints[:, :3].flatten()
    else:
        base = keypoints.flatten()
    
    # Create sequence with realistic variations
    sequence = []
    for i in range(seq_length):
        # Time progression
        t = i / seq_length
        phase_shift = t * 2 * np.pi
        
        # Create variations for each keypoint
        variation = np.zeros_like(base)
        for j in range(len(base)):
            keypoint_idx = j // 3
            coord = j % 3
            
            # Different movement patterns for different body parts
            # Upper body (head, shoulders, arms) - more movement
            if keypoint_idx <= 22:
                if coord == 0:  # x-axis
                    variation[j] = 0.04 * np.sin(phase_shift + keypoint_idx * 0.3)
                elif coord == 1:  # y-axis
                    variation[j] = 0.03 * np.cos(phase_shift + keypoint_idx * 0.2)
                else:  # z-axis
                    variation[j] = 0.02 * np.sin(phase_shift * 0.5 + keypoint_idx * 0.1)
            # Lower body (hips, legs, feet) - more stable
            else:
                if coord == 0:  # x-axis
                    variation[j] = 0.02 * np.sin(phase_shift + keypoint_idx * 0.2)
                elif coord == 1:  # y-axis
                    variation[j] = 0.02 * np.cos(phase_shift * 0.7 + keypoint_idx * 0.15)
                else:  # z-axis
                    variation[j] = 0.01 * np.sin(phase_shift * 0.3 + keypoint_idx * 0.1)
        
        # Add small random noise for natural variation
        noise = np.random.normal(0, 0.005, len(base))
        
        # Combine base + variation + noise
        frame_seq = base + variation + noise
        sequence.append(frame_seq)
    
    return np.array(sequence)


def create_activity_specific_sequence(keypoints, activity_type='walking', seq_length=30):
    """
    Create activity-specific synthetic sequence from a single frame.
    
    Args:
        keypoints: Single frame keypoints
        activity_type: 'walking', 'running', 'sitting', 'falling', 'climbing'
        seq_length: Number of frames to generate
        
    Returns:
        numpy array: (seq_length, 99)
    """
    if keypoints is None:
        return None
    
    if keypoints.shape[1] >= 3:
        base = keypoints[:, :3].flatten()
    else:
        base = keypoints.flatten()
    
    sequence = []
    
    # Activity-specific movement patterns
    if activity_type == 'walking':
        # Walking: rhythmic, moderate movement
        for i in range(seq_length):
            t = i / seq_length
            variation = np.zeros_like(base)
            for j in range(len(base)):
                keypoint_idx = j // 3
                coord = j % 3
                if coord == 0:
                    variation[j] = 0.06 * np.sin(t * 4 * np.pi + keypoint_idx * 0.2)
                elif coord == 1:
                    variation[j] = 0.04 * np.sin(t * 4 * np.pi + keypoint_idx * 0.15)
                else:
                    variation[j] = 0.02 * np.sin(t * 2 * np.pi + keypoint_idx * 0.1)
            sequence.append(base + variation + np.random.normal(0, 0.005, len(base)))
    
    elif activity_type == 'running':
        # Running: faster, larger movements
        for i in range(seq_length):
            t = i / seq_length
            variation = np.zeros_like(base)
            for j in range(len(base)):
                keypoint_idx = j // 3
                coord = j % 3
                if coord == 0:
                    variation[j] = 0.10 * np.sin(t * 6 * np.pi + keypoint_idx * 0.3)
                elif coord == 1:
                    variation[j] = 0.08 * np.sin(t * 6 * np.pi + keypoint_idx * 0.25)
                else:
                    variation[j] = 0.03 * np.sin(t * 3 * np.pi + keypoint_idx * 0.15)
            sequence.append(base + variation + np.random.normal(0, 0.008, len(base)))
    
    elif activity_type == 'sitting':
        # Sitting: minimal movement, stable pose
        for i in range(seq_length):
            t = i / seq_length
            variation = np.zeros_like(base)
            for j in range(len(base)):
                keypoint_idx = j // 3
                coord = j % 3
                if coord == 0:
                    variation[j] = 0.01 * np.sin(t * 2 * np.pi + keypoint_idx * 0.1)
                elif coord == 1:
                    variation[j] = 0.01 * np.cos(t * 2 * np.pi + keypoint_idx * 0.1)
                else:
                    variation[j] = 0.005 * np.sin(t * np.pi + keypoint_idx * 0.05)
            sequence.append(base + variation + np.random.normal(0, 0.003, len(base)))
    
    elif activity_type == 'falling':
        # Falling: rapid descent
        for i in range(seq_length):
            t = i / seq_length
            variation = np.zeros_like(base)
            for j in range(len(base)):
                keypoint_idx = j // 3
                coord = j % 3
                if coord == 1:  # y-axis (vertical)
                    # Rapid downward movement
                    variation[j] = -0.15 * (t ** 2) + 0.05 * np.sin(t * 2 * np.pi + keypoint_idx * 0.1)
                elif coord == 0:
                    variation[j] = 0.03 * np.sin(t * 4 * np.pi + keypoint_idx * 0.2)
                else:
                    variation[j] = 0.01 * np.sin(t * np.pi + keypoint_idx * 0.1)
            sequence.append(base + variation + np.random.normal(0, 0.01, len(base)))
    
    elif activity_type == 'climbing':
        # Climbing: upward movement
        for i in range(seq_length):
            t = i / seq_length
            variation = np.zeros_like(base)
            for j in range(len(base)):
                keypoint_idx = j // 3
                coord = j % 3
                if coord == 1:  # y-axis (vertical)
                    variation[j] = -0.08 * t + 0.04 * np.sin(t * 4 * np.pi + keypoint_idx * 0.2)
                elif coord == 0:
                    variation[j] = 0.05 * np.sin(t * 3 * np.pi + keypoint_idx * 0.3)
                else:
                    variation[j] = 0.02 * np.sin(t * 2 * np.pi + keypoint_idx * 0.15)
            sequence.append(base + variation + np.random.normal(0, 0.006, len(base)))
    
    else:
        # Default: generic movement
        for i in range(seq_length):
            t = i / seq_length
            variation = 0.04 * np.sin(t * 2 * np.pi * 2)
            sequence.append(base + variation + np.random.normal(0, 0.005, len(base)))
    
    return np.array(sequence)


# ==================== BRAVE BROWSER AUTO-OPEN FUNCTION ====================

def open_brave_browser():
    """Automatically open Brave browser after server starts"""
    time.sleep(2)
    url = 'http://localhost:5000'
    
    brave_paths = [
        "C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe",
        "C:\\Program Files (x86)\\BraveSoftware\\Brave-Browser\\Application\\brave.exe",
        os.path.expandvars("%LOCALAPPDATA%\\BraveSoftware\\Brave-Browser\\Application\\brave.exe"),
        "brave.exe"
    ]
    
    brave_found = False
    for path in brave_paths:
        try:
            if os.path.exists(path):
                webbrowser.register('brave', None, webbrowser.GenericBrowser(path))
                webbrowser.get('brave').open(url)
                brave_found = True
                logger.info(f"✅ Brave browser opened at {url}")
                break
        except Exception as e:
            logger.debug(f"Could not use {path}: {e}")
            continue
    
    if not brave_found:
        try:
            import subprocess
            result = subprocess.run(['where', 'brave'], capture_output=True, text=True)
            if result.returncode == 0:
                brave_path = result.stdout.strip().split('\n')[0]
                if brave_path and os.path.exists(brave_path):
                    webbrowser.register('brave', None, webbrowser.GenericBrowser(brave_path))
                    webbrowser.get('brave').open(url)
                    brave_found = True
                    logger.info(f"✅ Brave browser opened from PATH at {url}")
        except Exception as e:
            logger.debug(f"Could not find Brave in PATH: {e}")
    
    if not brave_found:
        logger.warning("⚠️ Brave browser not found. Opening in default browser...")
        webbrowser.open(url)
    
    logger.info("🌐 Browser should now be open at: " + url)


# ==================== ROUTES ====================

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard page"""
    return render_template('dashboard.html')

@app.route('/alerts')
def alerts_page():
    """Alerts page"""
    return render_template('alerts.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/settings')
def settings():
    """Settings page"""
    return render_template('settings.html')


# ==================== API ROUTES ====================

@app.route('/api/status')
def get_status():
    """Get current system status"""
    try:
        stats = alert_system.get_alert_statistics()
        health = perf_monitor.get_health_status()
        
        return jsonify({
            'monitoring_active': monitoring_active,
            'activity': current_status.get('activity', 'None'),
            'confidence': current_status.get('confidence', 0.0),
            'safe': current_status.get('safe', True),
            'alerts': current_status.get('alerts', [])[-10:],
            'statistics': stats,
            'fps': current_status.get('fps', 0),
            'frame_count': current_status.get('frame_count', 0),
            'health': health,
            'uptime': (datetime.now() - start_time).total_seconds(),
            'alert_count': len(current_status.get('alerts', []))
        })
    except Exception as e:
        logger.error(f"Error in get_status: {e}")
        return jsonify({
            'monitoring_active': monitoring_active,
            'activity': current_status.get('activity', 'None'),
            'confidence': current_status.get('confidence', 0.0),
            'safe': current_status.get('safe', True),
            'alerts': current_status.get('alerts', [])[-10:],
            'statistics': {'total': 0, 'by_severity': {}, 'by_activity': {}},
            'fps': current_status.get('fps', 0),
            'frame_count': current_status.get('frame_count', 0),
            'health': {'status': 'unknown', 'warnings': ['Error fetching health status']},
            'uptime': 0,
            'alert_count': len(current_status.get('alerts', [])),
            'error': str(e)
        }), 200

@app.route('/api/alerts')
def get_alerts():
    """Get all alerts"""
    try:
        limit = request.args.get('limit', 100, type=int)
        severity = request.args.get('severity', None)
        
        alerts = current_status.get('alerts', [])
        if severity:
            alerts = [a for a in alerts if a.get('severity') == severity]
        
        return jsonify({
            'alerts': alerts[-limit:],
            'total': len(alerts)
        })
    except Exception as e:
        logger.error(f"Error in get_alerts: {e}")
        return jsonify({'alerts': [], 'total': 0, 'error': str(e)}), 200

@app.route('/api/start', methods=['POST'])
def start_monitoring():
    """Start monitoring"""
    global monitoring_active, camera
    
    try:
        if not monitoring_active:
            camera_id = 0
            if request.is_json:
                camera_id = request.json.get('camera_id', 0)
            
            camera = cv2.VideoCapture(camera_id)
            if not camera.isOpened():
                return jsonify({'error': 'Could not open camera'}), 500
            
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
            camera.set(cv2.CAP_PROP_FPS, Config.FPS)
            
            monitoring_active = True
            threading.Thread(target=process_video, daemon=True).start()
            
            socketio.emit('status_update', {'monitoring': True})
            logger.info("Monitoring started")
            return jsonify({'status': 'started'})
        
        return jsonify({'status': 'already_running'})
    except Exception as e:
        logger.error(f"Error starting monitoring: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop', methods=['POST'])
def stop_monitoring():
    """Stop monitoring"""
    global monitoring_active, camera
    
    try:
        monitoring_active = False
        if camera:
            camera.release()
            camera = None
        
        socketio.emit('status_update', {'monitoring': False})
        logger.info("Monitoring stopped")
        return jsonify({'status': 'stopped'})
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_system():
    """Reset system state"""
    global keypoint_buffer, alerts_list, current_status
    
    try:
        with processing_lock:
            keypoint_buffer = []
            alerts_list = []
            current_status['alerts'] = []
            current_status['activity'] = 'None'
            current_status['confidence'] = 0.0
            current_status['safe'] = True
            activity_recognizer.reset_buffer()
            safety_engine.reset()
            tracker.reset()
        
        socketio.emit('status_update', {'reset': True})
        logger.info("System reset")
        return jsonify({'status': 'reset'})
    except Exception as e:
        logger.error(f"Error resetting system: {e}")
        return jsonify({'error': str(e)}), 500


# ==================== IMAGE UPLOAD WITH PROPER CONFIDENCE ====================

@app.route('/api/upload', methods=['POST'])
def upload_image():
    """Upload and process an image with proper activity recognition"""
    global current_status, alerts_list, keypoint_buffer
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Read image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        # Process the image with enhanced single-frame analysis
        result = process_single_image_enhanced(frame)
        
        # ✅ CRITICAL FIX: Update global status with image analysis results
        with processing_lock:
            # Update activity
            if result.get('activity') and result['activity'] != 'None':
                current_status['activity'] = result['activity']
                current_status['confidence'] = result.get('confidence', 0.0)
                current_status['safe'] = result.get('safe', True)
                
                # ✅ Generate alert for unsafe activities
                if not result.get('safe', True) and result.get('alert'):
                    alert_info = {
                        'id': len(alerts_list) + 1,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'message': result.get('message', f'Unsafe activity detected: {result["activity"]}'),
                        'severity': result.get('severity', 'high'),
                        'activity': result.get('activity', 'unknown'),
                        'confidence': result.get('confidence', 0.0),
                        'source': 'image_upload',
                        'image_analyzed': True
                    }
                    
                    alerts_list.append(alert_info)
                    current_status['alerts'] = alerts_list
                    
                    # ✅ Send alert via socket to all clients
                    socketio.emit('alert', alert_info)
                    
                    # ✅ Send via advanced alert system
                    try:
                        advanced_alert.send_alert(alert_info)
                    except Exception as e:
                        logger.error(f"Error sending advanced alert: {e}")
                    
                    logger.info(f"🚨 ALERT generated from image upload: {alert_info['message']}")
            
            # ✅ Update keypoint buffer for consistency
            keypoints = pose_estimator.extract_keypoints(frame)
            if keypoints is not None:
                keypoints_flat = keypoints[:, :3].flatten()
                keypoint_buffer.append(keypoints_flat)
                if len(keypoint_buffer) > Config.SEQUENCE_LENGTH * 2:
                    keypoint_buffer = keypoint_buffer[-Config.SEQUENCE_LENGTH:]
        
        # ✅ Broadcast status update to all connected clients
        socketio.emit('status_update', {
            'activity': current_status.get('activity', 'None'),
            'confidence': current_status.get('confidence', 0.0),
            'safe': current_status.get('safe', True),
            'alert_count': len(alerts_list),
            'fps': current_status.get('fps', 0),
            'source': 'image_upload',
            'timestamp': datetime.now().isoformat()
        })
        
        # Save uploaded image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"uploaded_{timestamp}.jpg"
        filepath = os.path.join('static/uploads', filename)
        os.makedirs('static/uploads', exist_ok=True)
        cv2.imwrite(filepath, frame)
        
        # Get activity probabilities if available
        probabilities = None
        if result.get('activity') and result['activity'] != 'None':
            try:
                keypoints = pose_estimator.extract_keypoints(frame)
                if keypoints is not None:
                    synthetic_seq = create_synthetic_sequence(keypoints, Config.SEQUENCE_LENGTH)
                    if synthetic_seq is not None:
                        _, _, probs = activity_recognizer.predict_activity(
                            synthetic_seq, 
                            return_probabilities=True
                        )
                        if probs is not None:
                            probabilities = {
                                activity_recognizer.activity_labels[i]: float(probs[i])
                                for i in range(len(probs))
                            }
            except Exception as e:
                logger.error(f"Error getting probabilities: {e}")
        
        return jsonify({
            'result': result,
            'image_url': f'/static/uploads/{filename}',
            'probabilities': probabilities,
            'status_updated': True,
            'alert_generated': not result.get('safe', True)
        })
        
    except Exception as e:
        logger.error(f"Error uploading image: {e}")
        return jsonify({'error': str(e)}), 500


def process_single_image_enhanced(frame):
    """
    Process a single image with enhanced synthetic sequence generation.
    This provides proper activity recognition with confidence scores.
    """
    result = {
        'activity': 'None',
        'confidence': 0.0,
        'safe': True,
        'message': 'All safe',
        'severity': 'low',
        'alert': None,
        'detections': [],
        'has_child': False,
        'processed_image': None,
        'pose_detected': False
    }
    
    try:
        # Detect children
        detections = detector.detect(frame)
        result['detections'] = detections
        result['has_child'] = len(detections) > 0
        
        if not detections:
            logger.info("No person detected in image")
            result['message'] = 'No person detected in the image'
            result['processed_image'] = encode_image_to_base64(frame)
            return result
        
        # ✅ Extract pose
        keypoints = pose_estimator.extract_keypoints(frame)
        result['pose_detected'] = keypoints is not None
        
        if keypoints is None:
            logger.warning("⚠️ No pose detected in image")
            result['message'] = 'Pose not detected. Ensure person is clearly visible.'
            annotated_frame = frame.copy()
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Person {det['confidence']:.2f}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            result['processed_image'] = encode_image_to_base64(annotated_frame)
            return result
        
        # ✅ Create synthetic sequence from single frame
        activities = ['walking', 'running', 'sitting', 'falling', 'climbing']
        best_activity = None
        best_confidence = 0.0
        all_predictions = {}
        
        for activity_type in activities:
            try:
                synthetic_seq = create_activity_specific_sequence(
                    keypoints, activity_type, Config.SEQUENCE_LENGTH
                )
                
                if synthetic_seq is not None:
                    pred_activity, confidence = activity_recognizer.predict_activity(synthetic_seq)
                    
                    if pred_activity is not None:
                        all_predictions[pred_activity] = confidence
                        
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_activity = pred_activity
            except Exception as e:
                logger.error(f"Error with {activity_type} sequence: {e}")
                continue
        
        # ✅ If no activity-specific sequence worked, try generic
        if best_activity is None:
            try:
                synthetic_seq = create_synthetic_sequence(keypoints, Config.SEQUENCE_LENGTH)
                if synthetic_seq is not None:
                    activity, confidence = activity_recognizer.predict_activity(synthetic_seq)
                    if activity is not None:
                        best_activity = activity
                        best_confidence = confidence
            except Exception as e:
                logger.error(f"Error with generic sequence: {e}")
        
        # ✅ Set results with proper confidence
        if best_activity is not None and best_confidence > 0.3:
            result['activity'] = best_activity
            result['confidence'] = best_confidence
            logger.info(f"🎯 Activity predicted: {best_activity} (confidence: {best_confidence:.2%})")
            
            # ✅ Check safety - CRITICAL for alert generation
            detection = detections[0]
            safety_result = safety_engine.check_safety(
                activity=best_activity,
                confidence=best_confidence,
                pose_keypoints=keypoints,
                bbox=detection['bbox'],
                frame_time=datetime.now()
            )
            
            result['safe'] = safety_result['safe']
            result['message'] = safety_result.get('message', 'All safe')
            result['severity'] = safety_result.get('severity', 'low')
            result['alert'] = safety_result.get('alert')
            
            # ✅ LOG for debugging
            logger.info(f"🔍 Safety check result: safe={result['safe']}, alert={result['alert']}")
        else:
            result['message'] = f'Activity not recognized with sufficient confidence (best: {best_confidence:.2%})'
            logger.warning(f"⚠️ No confident prediction: best confidence {best_confidence:.2%}")
        
        # ✅ Create annotated visualization
        annotated_frame = frame.copy()
        
        # Draw detections with confidence
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = (0, 255, 0) if result['safe'] else (0, 0, 255)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"Child {det['confidence']:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw pose
        annotated_frame = pose_estimator.draw_pose(annotated_frame)
        
        # Draw status with confidence
        recent_alerts = []
        if result.get('alert'):
            recent_alerts.append({
                'message': result['message'],
                'severity': result['severity']
            })
        
        extra_info = {'Pose': 'Detected' if keypoints is not None else 'Not Detected'}
        
        annotated_frame = visualizer.draw_status(
            annotated_frame,
            result['activity'],
            result['confidence'],
            result['safe'],
            recent_alerts,
            extra_info
        )
        
        result['processed_image'] = encode_image_to_base64(annotated_frame)
        
    except Exception as e:
        logger.error(f"Error processing single image: {e}")
        result['message'] = f"Error: {str(e)}"
        result['processed_image'] = encode_image_to_base64(frame)
    
    return result


def encode_image_to_base64(frame, quality=85):
    """Encode a frame to base64 string"""
    if frame is None:
        return None
    try:
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        return None


@app.route('/api/capture', methods=['POST'])
def capture_frame():
    """Capture and save current frame"""
    if not monitoring_active or camera is None:
        return jsonify({'error': 'Monitoring not active'}), 400
    
    try:
        ret, frame = camera.read()
        if not ret:
            return jsonify({'error': 'Failed to capture frame'}), 500
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"capture_{timestamp}.jpg"
        filepath = os.path.join('captures', filename)
        os.makedirs('captures', exist_ok=True)
        cv2.imwrite(filepath, frame)
        
        return jsonify({
            'filename': filename,
            'path': filepath,
            'url': f'/captures/{filename}'
        })
    except Exception as e:
        logger.error(f"Error capturing frame: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance')
def get_performance():
    """Get performance metrics"""
    try:
        health = perf_monitor.get_health_status()
        summary = perf_monitor.get_summary()
        history = perf_monitor.get_history(limit=50)
        
        return jsonify({
            'health': health,
            'summary': summary,
            'history': history
        })
    except Exception as e:
        logger.error(f"Error in get_performance: {e}")
        return jsonify({
            'health': {'status': 'unknown', 'warnings': ['Error fetching performance']},
            'summary': {},
            'history': {},
            'error': str(e)
        }), 200

@app.route('/api/settings', methods=['GET', 'POST'])
def manage_settings():
    """Get or update settings"""
    if request.method == 'GET':
        try:
            settings = {
                'enabled_methods': advanced_alert.config.get('enabled_methods', ['desktop']),
                'email': advanced_alert.config.get('email', {}),
                'telegram': advanced_alert.config.get('telegram', {}),
                'throttling': advanced_alert.config.get('throttling', {}),
                'yolo_model': Config.YOLO_MODEL,
                'confidence_threshold': Config.CONFIDENCE_THRESHOLD,
                'sequence_length': Config.SEQUENCE_LENGTH,
                'unsafe_activities': Config.UNSAFE_ACTIVITIES
            }
            return jsonify(settings)
        except Exception as e:
            logger.error(f"Error getting settings: {e}")
            return jsonify({'error': str(e)}), 500
    
    try:
        data = request.json
        if data:
            if 'enabled_methods' in data:
                advanced_alert.config['enabled_methods'] = data['enabled_methods']
            if 'email' in data:
                advanced_alert.config['email'].update(data['email'])
            if 'telegram' in data:
                advanced_alert.config['telegram'].update(data['telegram'])
            if 'throttling' in data:
                advanced_alert.config['throttling'].update(data['throttling'])
            
            with open('alert_config.json', 'w') as f:
                json.dump(advanced_alert.config, f, indent=2)
            advanced_alert.reload_config()
            
            return jsonify({'status': 'updated'})
        
        return jsonify({'error': 'Invalid request'}), 400
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/alerts')
def export_alerts():
    """Export alerts as CSV or JSON"""
    format = request.args.get('format', 'json')
    
    try:
        if format == 'json':
            return jsonify({'alerts': current_status.get('alerts', [])})
        elif format == 'csv':
            import csv
            import io
            
            output = io.StringIO()
            alerts = current_status.get('alerts', [])
            
            if alerts:
                fieldnames = ['id', 'timestamp', 'severity', 'activity', 'message', 'location', 'source']
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                for alert in alerts:
                    row = {k: alert.get(k, '') for k in fieldnames}
                    writer.writerow(row)
            
            return Response(
                output.getvalue(),
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename=alerts_{datetime.now().strftime("%Y%m%d")}.csv'}
            )
        
        return jsonify({'error': 'Invalid format'}), 400
    except Exception as e:
        logger.error(f"Error exporting alerts: {e}")
        return jsonify({'error': str(e)}), 500


# ==================== VIDEO STREAMING ====================

@app.route('/video_feed')
def video_feed():
    """Video streaming endpoint"""
    try:
        return Response(
            generate_video_feed(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e:
        logger.error(f"Error in video feed: {e}")
        return "Video feed unavailable", 500

def generate_video_feed():
    """Generate video feed for streaming"""
    global camera, monitoring_active
    
    while monitoring_active:
        try:
            if camera is None:
                break
            
            ret, frame = camera.read()
            if not ret:
                break
            
            processed_frame = process_frame_for_stream(frame)
            
            ret, jpeg = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + 
                       jpeg.tobytes() + b'\r\n\r\n')
            
            time.sleep(0.03)
        except Exception as e:
            logger.error(f"Error in generate_video_feed: {e}")
            break
    
    try:
        blank = np.zeros((Config.FRAME_HEIGHT, Config.FRAME_WIDTH, 3), dtype=np.uint8)
        ret, jpeg = cv2.imencode('.jpg', blank)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + 
                   jpeg.tobytes() + b'\r\n\r\n')
    except:
        pass


# ==================== PROCESSING FUNCTIONS ====================

def process_video():
    """Background video processing thread"""
    global camera, monitoring_active, keypoint_buffer, alerts_list, frame_count, current_status
    
    while monitoring_active and camera is not None:
        try:
            ret, frame = camera.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            frame_count += 1
            
            result = process_single_frame(frame)
            
            with processing_lock:
                current_status['activity'] = result.get('activity', 'None')
                current_status['confidence'] = result.get('confidence', 0.0)
                current_status['safe'] = result.get('safe', True)
                current_status['frame_count'] = frame_count
                
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > 0:
                    current_status['fps'] = frame_count / elapsed
            
            if not result.get('safe', True) and result.get('alert'):
                alert_info = {
                    'id': len(alerts_list) + 1,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'message': result.get('message', 'Unsafe behavior detected'),
                    'severity': result.get('severity', 'medium'),
                    'activity': result.get('activity', 'unknown'),
                    'confidence': result.get('confidence', 0.0),
                    'source': 'live_feed'
                }
                
                with processing_lock:
                    alerts_list.append(alert_info)
                    current_status['alerts'] = alerts_list
                
                socketio.emit('alert', alert_info)
                
                try:
                    advanced_alert.send_alert(alert_info)
                except Exception as e:
                    logger.error(f"Error sending advanced alert: {e}")
                
                logger.info(f"Alert generated: {alert_info['message']}")
            
            if frame_count % 5 == 0:
                try:
                    socketio.emit('status_update', {
                        'activity': current_status['activity'],
                        'confidence': current_status['confidence'],
                        'safe': current_status['safe'],
                        'alert_count': len(alerts_list),
                        'fps': current_status['fps']
                    })
                except Exception as e:
                    logger.error(f"Error emitting status update: {e}")
            
            if frame_count % 30 == 0:
                try:
                    perf_monitor.add_metric('frames_processed', 30)
                except Exception as e:
                    logger.error(f"Error updating performance metrics: {e}")
            
        except Exception as e:
            logger.error(f"Error in video processing: {e}")
            time.sleep(0.1)

def process_single_frame(frame):
    """Process a single frame for monitoring"""
    global keypoint_buffer
    
    result = {
        'activity': 'None',
        'confidence': 0.0,
        'safe': True,
        'message': 'All safe',
        'severity': 'low',
        'alert': None,
        'detections': []
    }
    
    try:
        detections = detector.detect(frame)
        result['detections'] = detections
        
        tracks = tracker.update(detections, frame)
        
        if not detections:
            return result
        
        detection = detections[0]
        bbox = detection['bbox']
        
        keypoints = pose_estimator.extract_keypoints(frame)
        
        if keypoints is not None:
            keypoints_flat = keypoints[:, :3].flatten()
            keypoint_buffer.append(keypoints_flat)
            
            if len(keypoint_buffer) > Config.SEQUENCE_LENGTH * 2:
                keypoint_buffer = keypoint_buffer[-Config.SEQUENCE_LENGTH:]
            
            if len(keypoint_buffer) >= Config.SEQUENCE_LENGTH:
                try:
                    activity, confidence = activity_recognizer.predict_activity(keypoint_buffer)
                    result['activity'] = activity or 'Unknown'
                    result['confidence'] = confidence
                except Exception as e:
                    logger.error(f"Error predicting activity: {e}")
                    result['activity'] = 'Error'
                    result['confidence'] = 0.0
            else:
                result['activity'] = 'Collecting data...'
                result['confidence'] = 0.0
        else:
            if keypoint_buffer:
                keypoint_buffer.append(np.zeros(33 * 3))
                if len(keypoint_buffer) > Config.SEQUENCE_LENGTH * 2:
                    keypoint_buffer = keypoint_buffer[-Config.SEQUENCE_LENGTH:]
        
        if result['activity'] not in ['None', 'Collecting data...', 'Error']:
            try:
                safety_result = safety_engine.check_safety(
                    activity=result['activity'],
                    confidence=result['confidence'],
                    pose_keypoints=keypoints if 'keypoints' in locals() else None,
                    bbox=bbox,
                    frame_time=datetime.now()
                )
                
                result['safe'] = safety_result['safe']
                if not result['safe']:
                    result['message'] = safety_result.get('message', 'Unsafe behavior detected')
                    result['severity'] = safety_result.get('severity', 'medium')
                    result['alert'] = safety_result.get('alert')
            except Exception as e:
                logger.error(f"Error checking safety: {e}")
        
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
    
    return result

def process_frame_for_stream(frame):
    """Process frame for streaming display"""
    try:
        result = process_single_frame(frame)
        
        annotated_frame = frame.copy()
        
        for det in result.get('detections', []):
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Child {det['confidence']:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        keypoints = pose_estimator.extract_keypoints(frame)
        if keypoints is not None:
            annotated_frame = pose_estimator.draw_pose(annotated_frame)
        
        annotated_frame = tracker.draw_tracks(annotated_frame)
        
        recent_alerts = current_status.get('alerts', [])[-3:]
        annotated_frame = visualizer.draw_status(
            annotated_frame,
            result['activity'],
            result['confidence'],
            result['safe'],
            recent_alerts
        )
        
        return annotated_frame
    except Exception as e:
        logger.error(f"Error in process_frame_for_stream: {e}")
        return frame


# ==================== SOCKETIO EVENTS ====================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {
        'status': 'connected',
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('start_monitoring')
def handle_start():
    """Handle start monitoring from socket"""
    try:
        global monitoring_active, camera
        
        if not monitoring_active:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                emit('error', {'message': 'Could not open camera'})
                return
            
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
            camera.set(cv2.CAP_PROP_FPS, Config.FPS)
            
            monitoring_active = True
            threading.Thread(target=process_video, daemon=True).start()
            
            emit('status_update', {'monitoring': True})
            logger.info("Monitoring started from socket")
        else:
            emit('status_update', {'monitoring': True, 'message': 'Already running'})
    except Exception as e:
        logger.error(f"Error in handle_start: {e}")
        emit('error', {'message': str(e)})

@socketio.on('stop_monitoring')
def handle_stop():
    """Handle stop monitoring from socket"""
    global monitoring_active, camera
    
    try:
        monitoring_active = False
        if camera:
            camera.release()
            camera = None
        
        emit('status_update', {'monitoring': False})
        logger.info("Monitoring stopped from socket")
    except Exception as e:
        logger.error(f"Error in handle_stop: {e}")
        emit('error', {'message': str(e)})

@socketio.on('get_status')
def handle_get_status():
    """Handle get status request"""
    try:
        response = get_status()
        if hasattr(response, 'get_json'):
            status = response.get_json()
            emit('status_update', status)
        else:
            emit('status_update', {
                'monitoring_active': monitoring_active,
                'activity': current_status.get('activity', 'None'),
                'safe': current_status.get('safe', True),
                'alert_count': len(alerts_list),
                'fps': current_status.get('fps', 0)
            })
    except Exception as e:
        logger.error(f"Error in handle_get_status: {e}")
        emit('error', {'message': str(e)})


# ==================== SERVE STATIC FILES ====================

@app.route('/captures/<filename>')
def serve_capture(filename):
    """Serve captured image"""
    filepath = os.path.join('captures', filename)
    if os.path.exists(filepath):
        return send_file(filepath, mimetype='image/jpeg')
    return jsonify({'error': 'File not found'}), 404

@app.route('/alerts/<filename>')
def serve_alert_image(filename):
    """Serve alert image"""
    filepath = os.path.join('alerts', filename)
    if os.path.exists(filepath):
        return send_file(filepath, mimetype='image/jpeg')
    return jsonify({'error': 'File not found'}), 404


# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


# ==================== MAIN ENTRY POINT ====================

def find_available_port(start_port=5000, max_port=5010):
    """Find an available port starting from start_port"""
    for port in range(start_port, max_port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    return start_port

def main():
    """Main entry point for the Flask application"""
    try:
        os.makedirs('static/uploads', exist_ok=True)
        os.makedirs('captures', exist_ok=True)
        os.makedirs('alerts', exist_ok=True)
        os.makedirs('saved_models', exist_ok=True)
        os.makedirs('data/activities', exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating directories: {e}")
    
    host = Config.FLASK_HOST
    port = find_available_port(Config.FLASK_PORT)
    debug = Config.FLASK_DEBUG
    
    logger.info(f"Starting Flask server on {host}:{port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"Open browser at: http://localhost:{port}")
    
    try:
        browser_thread = threading.Thread(target=open_brave_browser, daemon=True)
        browser_thread.start()
    except Exception as e:
        logger.error(f"Error starting browser: {e}")
    
    try:
        socketio.run(app, host=host, port=port, debug=debug)
    except Exception as e:
        logger.error(f"Error with socketio.run: {e}")
        logger.info("Falling back to app.run...")
        try:
            app.run(host=host, port=port, debug=debug)
        except Exception as e2:
            logger.error(f"Error with app.run: {e2}")
            port = find_available_port(port + 1)
            logger.info(f"Trying port {port}")
            app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    main()
