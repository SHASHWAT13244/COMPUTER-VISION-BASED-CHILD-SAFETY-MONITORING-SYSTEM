# flask_app.py
"""
Flask Web Application for Child Safety Monitoring System
Real-time pipeline: single-camera-reader producer + MJPEG consumers.
"""

import os
import sys
import cv2
import time
import json
import base64
import csv
import io
import threading
import queue
import logging
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import webbrowser
import socket
import atexit

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, Response, request, jsonify, send_file, url_for
from flask_cors import CORS
from flask_socketio import SocketIO, emit

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = Config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')


# =====================================================================
#  COMPONENTS
# =====================================================================
detector = ChildDetector(
    model_path=Config.YOLO_MODEL,
    conf_threshold=Config.CONFIDENCE_THRESHOLD,
    iou_threshold=Config.IOU_THRESHOLD,
)

pose_estimator = PoseEstimator(
    min_detection_confidence=Config.POSE_MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=Config.POSE_MIN_TRACKING_CONFIDENCE,
    model_complexity=Config.POSE_MODEL_COMPLEXITY,
)

activity_recognizer = ActivityRecognizer(
    sequence_length=Config.SEQUENCE_LENGTH,
    num_keypoints=33,
    num_classes=len(Config.ACTIVITY_CLASSES),
    hidden_size=Config.LSTM_HIDDEN_SIZE,
    num_layers=Config.LSTM_NUM_LAYERS,
    dropout=Config.LSTM_DROPOUT,
    bidirectional=Config.LSTM_BIDIRECTIONAL,
)

safety_engine   = SafetyEngine()
tracker         = PersonTracker(max_lost_frames=10, min_confidence=0.5)
alert_system    = AlertSystem(sound_enabled=False, display_enabled=True, log_enabled=True)
advanced_alert  = AdvancedAlertSystem('alert_config.json')
visualizer      = Visualizer(show_fps=True, show_info=True, show_activity=True)
perf_monitor    = PerformanceMonitor()

# Background pool for blocking I/O (SMTP, HTTP webhooks, etc.)
_alert_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix='alert')


def load_pretrained_model():
    for path in [
        os.path.join(Config.MODELS_DIR, 'activity_model.pth'),
        os.path.join(Config.MODELS_DIR, 'best_activity_model.pth'),
    ]:
        if os.path.exists(path):
            try:
                activity_recognizer.load_model(path)
                logger.info(f"Loaded pre-trained model from {path}")
                return True
            except Exception as e:
                logger.error(f"Error loading {path}: {e}")
    logger.info("No pre-trained model found. Using random weights.")
    return False


load_pretrained_model()
perf_monitor.start_monitoring()


# =====================================================================
#  CONSTANTS
# =====================================================================
MAX_ALERTS_KEPT  = 1000    # ring-buffer cap for _alerts_list
MAX_EMPTY_GRACE  = 5       # consecutive YOLO misses before clearing pose cache


# =====================================================================
#  SHARED STATE (single producer, many consumers)
# =====================================================================
_current_status = {
    'activity': 'None',
    'confidence': 0.0,
    'safe': True,
    'alerts': [],
    'fps': 0.0,
    'frame_count': 0,
}
_status_lock = threading.Lock()

_alerts_list = []
_alerts_lock = threading.Lock()

keypoint_buffer   = []              # guarded by _processing_lock
_processing_lock  = threading.Lock()

# Guards torch model inference across reader thread + Flask request threads
_recognizer_lock  = threading.Lock()

# Guards writes to alert_config.json
_config_write_lock = threading.Lock()

# Camera state
camera           = None
_reader_thread   = None
_reader_running  = False
_reader_lock     = threading.Lock()

# Per-MJPEG-consumer fan-out
_subscribers       = set()
_subscribers_lock  = threading.Lock()
_last_frame_for_capture = None
_last_frame_lock        = threading.Lock()

# Cached detections / keypoints between heavy-model runs
_last_boxes              = []
_last_keypoints          = None
_empty_detection_frames  = 0

_start_time       = datetime.now()
monitoring_active = False


# =====================================================================
#  FRAME PROCESSING (worker thread only)
# =====================================================================
def _empty_result():
    return {
        'activity': 'Collecting data...',
        'confidence': 0.0,
        'safe': True,
        'message': 'All safe',
        'severity': 'low',
        'alert': None,
        'detections': [],
    }


def process_single_frame(frame):
    """Run detection / pose / activity / safety on one frame.

    Uses frame-count-based scheduling to run YOLO and MediaPipe less
    often than every frame. Only called from the camera reader thread.
    """
    global _last_boxes, _last_keypoints, keypoint_buffer, _empty_detection_frames

    with _status_lock:
        fc = _current_status['frame_count']

    do_detect = (fc % Config.DETECT_EVERY_N_FRAMES == 0) or not _last_boxes
    do_pose   = (fc % Config.POSE_EVERY_N_FRAMES  == 0) or _last_keypoints is None

    # ---- Detection (YOLO) ----
    if do_detect:
        try:
            _last_boxes = detector.detect(frame)
        except Exception as e:
            logger.error(f"YOLO error: {e}")
            _last_boxes = []
    detections = _last_boxes

    # ---- Detection miss: grace period before wiping pose cache ----
    if not detections:
        _empty_detection_frames += 1
        if _empty_detection_frames >= MAX_EMPTY_GRACE:
            _last_keypoints = None
            with _processing_lock:
                keypoint_buffer.clear()
        return _empty_result()
    _empty_detection_frames = 0

    # ---- Pose (MediaPipe) ----
    if do_pose:
        try:
            _last_keypoints = pose_estimator.extract_keypoints(frame)
        except Exception as e:
            logger.error(f"Pose error: {e}")
            _last_keypoints = None
    keypoints = _last_keypoints

    result = _empty_result()
    result['detections'] = detections

    if keypoints is None:
        return result

    # ---- Activity buffer (guarded) ----
    with _processing_lock:
        keypoint_buffer.append(keypoints[:, :3].flatten())
        if len(keypoint_buffer) > Config.SEQUENCE_LENGTH:
            keypoint_buffer.pop(0)
        buffer_snapshot = list(keypoint_buffer)

    if len(buffer_snapshot) < Config.SEQUENCE_LENGTH:
        result['activity'] = 'Collecting data...'
        return result

    # ---- LSTM prediction (serialized) ----
    try:
        with _recognizer_lock:
            activity, confidence = activity_recognizer.predict_activity(buffer_snapshot)
    except Exception as e:
        logger.error(f"Activity predict error: {e}")
        return result

    if not activity:
        return result

    result['activity']   = activity
    result['confidence'] = float(confidence)

    # ---- Safety rules ----
    try:
        safety = safety_engine.check_safety(
            activity=activity,
            confidence=confidence,
            pose_keypoints=keypoints,
            bbox=detections[0]['bbox'],
            frame_time=datetime.now(),
        )
        result['safe']     = safety.get('safe', True)
        result['message']  = safety.get('message', 'All safe')
        result['severity'] = safety.get('severity', 'low')
        result['alert']    = safety.get('alert')
    except Exception as e:
        logger.error(f"Safety check error: {e}")

    return result


def annotate_frame(frame, result):
    """Draw boxes, pose, tracker IDs and status overlay on a copy of the frame."""
    annotated = frame.copy()

    safe = result.get('safe', True)
    for det in result.get('detections', []):
        x1, y1, x2, y2 = det['bbox']
        color = (0, 255, 0) if safe else (0, 0, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"Child {det.get('confidence', 0):.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    try:
        annotated = pose_estimator.draw_pose(annotated)
    except Exception:
        pass

    try:
        if result.get('detections'):
            annotated = tracker.draw_tracks(annotated)
    except Exception:
        pass

    try:
        visualizer.fps = _current_status.get('fps', 0.0)
        annotated = visualizer.draw_status(
            annotated,
            result.get('activity', 'None'),
            result.get('confidence', 0.0),
            result.get('safe', True),
            _current_status.get('alerts', [])[-3:],
        )
    except Exception as e:
        logger.debug(f"Status overlay error: {e}")

    return annotated


# =====================================================================
#  ALERTS: async delivery + bounded list
# =====================================================================
def _safe_send_alert(alert_info):
    try:
        advanced_alert.send_alert(alert_info)
    except Exception as e:
        logger.error(f"Advanced alert error: {e}")


def _append_alert(alert_info):
    """Append to bounded _alerts_list and return an atomic snapshot."""
    with _alerts_lock:
        _alerts_list.append(alert_info)
        if len(_alerts_list) > MAX_ALERTS_KEPT:
            del _alerts_list[:-MAX_ALERTS_KEPT]
        return list(_alerts_list)


def _dispatch_alert(alert_info):
    """Push socket + async delivery; must be called outside locks."""
    try:
        socketio.emit('alert', alert_info)
    except Exception:
        pass
    try:
        _alert_pool.submit(_safe_send_alert, alert_info)
    except Exception as e:
        logger.error(f"Alert submit failed: {e}")
    logger.info(f"🚨 Alert: {alert_info['message']}")


# =====================================================================
#  FRAME FAN-OUT (per-MJPEG-consumer)
# =====================================================================
def _publish_frame(frame):
    """Publish frame to every subscriber; keep last for /api/capture."""
    global _last_frame_for_capture

    with _last_frame_lock:
        _last_frame_for_capture = frame

    with _subscribers_lock:
        dead = []
        for q in _subscribers:
            try:
                q.put_nowait(frame)
            except queue.Full:
                # Drop the oldest for this subscriber, then retry once
                try:
                    q.get_nowait()
                    q.put_nowait(frame)
                except Exception:
                    dead.append(q)
        for q in dead:
            _subscribers.discard(q)


# =====================================================================
#  CAMERA READER THREAD  (the ONLY place cv2.VideoCapture is used)
# =====================================================================
def _open_camera():
    """Open camera with the lowest-latency backend available."""
    backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_V4L2
    cap = cv2.VideoCapture(Config.CAMERA_ID, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(Config.CAMERA_ID)   # fallback

    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          Config.FPS)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, Config.CAMERA_BUFFER_SIZE)
    except Exception:
        pass

    return cap


def camera_reader_loop(cap):
    """Single producer: grab → infer → annotate → publish latest frame."""
    global camera, _reader_running, monitoring_active
    global _last_boxes, _last_keypoints, keypoint_buffer, _empty_detection_frames

    camera = cap

    # Flush driver's internal buffer
    for _ in range(5):
        cap.grab()

    # Warm-up inference (forces lazy model init: YOLO + MediaPipe)
    ok, warm = cap.retrieve()
    if ok and warm is not None:
        try:
            detector.detect(warm)
            pose_estimator.extract_keypoints(warm)
            logger.info("Warm-up inference complete")
        except Exception as e:
            logger.warning(f"Warm-up inference failed: {e}")

    frame_times = []
    last_emit   = time.time()
    frame_idx   = 0
    target_dt   = 1.0 / max(1, Config.STREAM_MAX_FPS)

    try:
        while True:
            with _reader_lock:
                if not _reader_running:
                    break

            loop_start = time.perf_counter()

            # Drop stale frames in the OS/driver queue
            for _ in range(Config.GRAB_FLUSH_COUNT):
                if not cap.grab():
                    break

            ok, frame = cap.retrieve()
            if not ok or frame is None:
                time.sleep(0.005)
                continue

            frame_idx += 1
            with _status_lock:
                _current_status['frame_count'] += 1

            # Inference + drawing
            try:
                result = process_single_frame(frame)
                annotated = annotate_frame(frame, result)
            except Exception as e:
                logger.error(f"Pipeline error: {e}")
                annotated = frame
                result = _empty_result()

            # Publish
            _publish_frame(annotated)

            # ---- Status update ----
            now = time.time()
            frame_times.append(now)
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps = 0.0
            if len(frame_times) >= 2:
                span = frame_times[-1] - frame_times[0]
                if span > 0:
                    fps = (len(frame_times) - 1) / span

            alert_info = None
            with _status_lock:
                _current_status['activity']   = result.get('activity', 'None')
                _current_status['confidence'] = float(result.get('confidence', 0.0))
                _current_status['safe']       = result.get('safe', True)
                _current_status['fps']        = fps

                if result.get('alert') and not result.get('safe', True):
                    alert_info = {
                        'id': -1,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'message': result.get('message', 'Unsafe behaviour detected'),
                        'severity': result.get('severity', 'medium'),
                        'activity': result.get('activity', 'unknown'),
                        'confidence': float(result.get('confidence', 0.0)),
                        'source': 'live_feed',
                    }

            # Side-effects outside locks
            if alert_info is not None:
                alert_info['id'] = len(_alerts_list) + 1
                snapshot = _append_alert(alert_info)
                with _status_lock:
                    _current_status['alerts'] = snapshot
                _dispatch_alert(alert_info)

            # Broadcast status once per second
            if now - last_emit > 1.0:
                last_emit = now
                try:
                    socketio.emit('status_update', {
                        'activity': _current_status['activity'],
                        'confidence': _current_status['confidence'],
                        'safe': _current_status['safe'],
                        'alert_count': len(_alerts_list),
                        'fps': _current_status['fps'],
                    })
                except Exception:
                    pass
                perf_monitor.add_metric('fps', _current_status['fps'])
                perf_monitor.add_metric('frames_processed', frame_idx)

            # Pace the loop
            elapsed = time.perf_counter() - loop_start
            sleep_for = target_dt - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    finally:
        try:
            cap.release()
        except Exception:
            pass
        camera = None
        with _reader_lock:
            _reader_running   = False
            monitoring_active = False
        logger.info("Camera reader thread stopped")


def _start_reader():
    global _reader_thread, _reader_running, monitoring_active, _start_time
    global _last_boxes, _last_keypoints, keypoint_buffer, _empty_detection_frames
    global _last_frame_for_capture

    with _reader_lock:
        if _reader_running:
            return False

        cap = _open_camera()
        if cap is None:
            logger.error("Could not open camera")
            return False

        _reader_running   = True
        monitoring_active = True
        _start_time       = datetime.now()

        # Reset pipeline state (do NOT clear _subscribers — existing MJPEG
        # consumers must keep receiving once monitoring begins)
        _last_boxes     = []
        _last_keypoints = None
        _empty_detection_frames = 0
        with _processing_lock:
            keypoint_buffer.clear()
        with _last_frame_lock:
            _last_frame_for_capture = None

        _reader_thread = threading.Thread(
            target=camera_reader_loop, args=(cap,), daemon=True
        )
        _reader_thread.start()

    logger.info("Monitoring started")
    return True


def _stop_reader():
    global _reader_running, monitoring_active, camera, _reader_thread
    with _reader_lock:
        _reader_running   = False
        monitoring_active = False

    if _reader_thread:
        _reader_thread.join(timeout=3.0)
        _reader_thread = None

    if camera:
        try:
            camera.release()
        except Exception:
            pass
        camera = None
    logger.info("Monitoring stopped")
    return True


# =====================================================================
#  SYNTHETIC SEQUENCE HELPERS (used by image upload)
# =====================================================================
def create_synthetic_sequence(keypoints, seq_length=30):
    if keypoints is None:
        return None
    base = keypoints[:, :3].flatten() if keypoints.shape[1] >= 3 else keypoints.flatten()
    seq = []
    for i in range(seq_length):
        t = i / seq_length
        phase = t * 2 * np.pi
        variation = np.zeros_like(base)
        for j in range(len(base)):
            kp = j // 3
            c  = j % 3
            if kp <= 22:
                if c == 0:   variation[j] = 0.04 * np.sin(phase + kp * 0.3)
                elif c == 1: variation[j] = 0.03 * np.cos(phase + kp * 0.2)
                else:        variation[j] = 0.02 * np.sin(phase * 0.5 + kp * 0.1)
            else:
                if c == 0:   variation[j] = 0.02 * np.sin(phase + kp * 0.2)
                elif c == 1: variation[j] = 0.02 * np.cos(phase * 0.7 + kp * 0.15)
                else:        variation[j] = 0.01 * np.sin(phase * 0.3 + kp * 0.1)
        noise = np.random.normal(0, 0.005, len(base))
        seq.append(base + variation + noise)
    return np.array(seq)


def create_activity_specific_sequence(keypoints, activity_type='walking', seq_length=30):
    if keypoints is None:
        return None
    base = keypoints[:, :3].flatten() if keypoints.shape[1] >= 3 else keypoints.flatten()
    seq = []
    for i in range(seq_length):
        t = i / seq_length
        variation = np.zeros_like(base)
        for j in range(len(base)):
            kp = j // 3
            c  = j % 3
            if activity_type == 'walking':
                if c == 0:   variation[j] = 0.06 * np.sin(t * 4 * np.pi + kp * 0.2)
                elif c == 1: variation[j] = 0.04 * np.sin(t * 4 * np.pi + kp * 0.15)
                else:        variation[j] = 0.02 * np.sin(t * 2 * np.pi + kp * 0.1)
            elif activity_type == 'running':
                if c == 0:   variation[j] = 0.10 * np.sin(t * 6 * np.pi + kp * 0.3)
                elif c == 1: variation[j] = 0.08 * np.sin(t * 6 * np.pi + kp * 0.25)
                else:        variation[j] = 0.03 * np.sin(t * 3 * np.pi + kp * 0.15)
            elif activity_type == 'sitting':
                if c == 0:   variation[j] = 0.01 * np.sin(t * 2 * np.pi + kp * 0.1)
                elif c == 1: variation[j] = 0.01 * np.cos(t * 2 * np.pi + kp * 0.1)
                else:        variation[j] = 0.005 * np.sin(t * np.pi + kp * 0.05)
            elif activity_type == 'falling':
                if c == 1:   variation[j] = -0.15 * (t ** 2) + 0.05 * np.sin(t * 2 * np.pi + kp * 0.1)
                elif c == 0: variation[j] = 0.03 * np.sin(t * 4 * np.pi + kp * 0.2)
                else:        variation[j] = 0.01 * np.sin(t * np.pi + kp * 0.1)
            elif activity_type == 'climbing':
                if c == 1:   variation[j] = -0.08 * t + 0.04 * np.sin(t * 4 * np.pi + kp * 0.2)
                elif c == 0: variation[j] = 0.05 * np.sin(t * 3 * np.pi + kp * 0.3)
                else:        variation[j] = 0.02 * np.sin(t * 2 * np.pi + kp * 0.15)
            else:
                variation[j] = 0.04 * np.sin(t * 4 * np.pi)
        seq.append(base + variation + np.random.normal(0, 0.006, len(base)))
    return np.array(seq)


# =====================================================================
#  ROUTES
# =====================================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/alerts')
def alerts_page():
    return render_template('alerts.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')


# =====================================================================
#  API ROUTES
# =====================================================================
@app.route('/api/status')
def get_status():
    try:
        stats  = alert_system.get_alert_statistics()
        health = perf_monitor.get_health_status()

        with _status_lock:
            snap = {
                'activity':    _current_status['activity'],
                'confidence':  _current_status['confidence'],
                'safe':        _current_status['safe'],
                'fps':         _current_status['fps'],
                'frame_count': _current_status['frame_count'],
            }
        with _alerts_lock:
            alerts_tail = list(_alerts_list[-10:])
            alert_count = len(_alerts_list)
        with _reader_lock:
            is_active = monitoring_active
            started   = _start_time

        return jsonify({
            'monitoring_active': is_active,
            'activity':    snap['activity'],
            'confidence':  snap['confidence'],
            'safe':        snap['safe'],
            'alerts':      alerts_tail,
            'statistics':  stats,
            'fps':         snap['fps'],
            'frame_count': snap['frame_count'],
            'health':      health,
            'uptime':      (datetime.now() - started).total_seconds(),
            'alert_count': alert_count,
        })
    except Exception as e:
        logger.error(f"get_status error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/alerts')
def get_alerts():
    limit    = request.args.get('limit', 100, type=int)
    severity = request.args.get('severity', None)
    with _alerts_lock:
        alerts = list(_alerts_list)
    if severity:
        alerts = [a for a in alerts if a.get('severity') == severity]
    return jsonify({'alerts': alerts[-limit:], 'total': len(alerts)})


@app.route('/api/start', methods=['POST'])
def start_monitoring():
    try:
        started = _start_reader()
        if started:
            socketio.emit('status_update', {'monitoring': True})
            return jsonify({'status': 'started'})
        if monitoring_active:
            return jsonify({'status': 'already_running'})
        return jsonify({'status': 'error', 'error': 'Could not open camera'}), 500
    except Exception as e:
        logger.error(f"start error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stop', methods=['POST'])
def stop_monitoring():
    try:
        _stop_reader()
        socketio.emit('status_update', {'monitoring': False})
        return jsonify({'status': 'stopped'})
    except Exception as e:
        logger.error(f"stop error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/reset', methods=['POST'])
def reset_system():
    """Stop reader if active, wipe state, then restart if it was running."""
    try:
        was_running = monitoring_active
        if was_running:
            _stop_reader()

        global _last_boxes, _last_keypoints, _empty_detection_frames
        with _processing_lock:
            keypoint_buffer.clear()
            with _alerts_lock:
                _alerts_list.clear()
            with _status_lock:
                _current_status['alerts']     = []
                _current_status['activity']   = 'None'
                _current_status['confidence'] = 0.0
                _current_status['safe']       = True
            _last_boxes             = []
            _last_keypoints         = None
            _empty_detection_frames = 0

        activity_recognizer.reset_buffer()
        safety_engine.reset()
        tracker.reset()

        resumed = False
        if was_running:
            resumed = _start_reader()

        socketio.emit('status_update', {'reset': True})
        return jsonify({'status': 'reset', 'resumed': resumed})
    except Exception as e:
        logger.error(f"reset error: {e}")
        return jsonify({'error': str(e)}), 500


# =====================================================================
#  IMAGE UPLOAD
# =====================================================================
def _encode_b64(frame, quality=None):
    if frame is None:
        return None
    q = quality if quality is not None else Config.JPEG_QUALITY
    ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        return None
    return base64.b64encode(buf).decode('utf-8')


def process_single_image_enhanced(frame):
    result = {
        'activity': 'None', 'confidence': 0.0, 'safe': True,
        'message': 'All safe', 'severity': 'low', 'alert': None,
        'detections': [], 'has_child': False,
        'processed_image': None, 'pose_detected': False,
    }
    try:
        detections = detector.detect(frame)
        result['detections'] = detections
        result['has_child']  = len(detections) > 0
        if not detections:
            result['message'] = 'No person detected in the image'
            result['processed_image'] = _encode_b64(frame)
            return result

        keypoints = pose_estimator.extract_keypoints(frame)
        result['pose_detected'] = keypoints is not None

        if keypoints is None:
            result['message'] = 'Pose not detected. Ensure person is clearly visible.'
            vis = frame.copy()
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            result['processed_image'] = _encode_b64(vis)
            return result

        activities = ['walking', 'running', 'sitting', 'falling', 'climbing']
        best_activity, best_conf = None, 0.0
        for act in activities:
            seq = create_activity_specific_sequence(keypoints, act, Config.SEQUENCE_LENGTH)
            if seq is None:
                continue
            try:
                with _recognizer_lock:
                    pred, conf = activity_recognizer.predict_activity(seq)
                if pred is not None and conf > best_conf:
                    best_activity, best_conf = pred, conf
            except Exception as e:
                logger.debug(f"predict {act} failed: {e}")

        if best_activity is None:
            seq = create_synthetic_sequence(keypoints, Config.SEQUENCE_LENGTH)
            if seq is not None:
                try:
                    with _recognizer_lock:
                        pred, conf = activity_recognizer.predict_activity(seq)
                    if pred is not None:
                        best_activity, best_conf = pred, conf
                except Exception:
                    pass

        if best_activity is not None and best_conf > 0.3:
            result['activity']   = best_activity
            result['confidence'] = float(best_conf)
            safety = safety_engine.check_safety(
                activity=best_activity,
                confidence=best_conf,
                pose_keypoints=keypoints,
                bbox=detections[0]['bbox'],
                frame_time=datetime.now(),
            )
            result['safe']     = safety.get('safe', True)
            result['message']  = safety.get('message', 'All safe')
            result['severity'] = safety.get('severity', 'low')
            result['alert']    = safety.get('alert')
        else:
            result['message'] = f'Activity not recognized with sufficient confidence ({best_conf:.2%})'

        vis = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = (0, 255, 0) if result['safe'] else (0, 0, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        vis = pose_estimator.draw_pose(vis)
        result['processed_image'] = _encode_b64(vis)
    except Exception as e:
        logger.error(f"process_single_image_enhanced error: {e}")
        result['message'] = f"Error: {e}"
        result['processed_image'] = _encode_b64(frame)
    return result


@app.route('/api/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        data = file.read()
        arr  = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Invalid image'}), 400

        result = process_single_image_enhanced(frame)

        alert_info = None
        with _processing_lock:
            if result.get('activity') and result['activity'] != 'None':
                with _status_lock:
                    _current_status['activity']   = result['activity']
                    _current_status['confidence'] = result.get('confidence', 0.0)
                    _current_status['safe']       = result.get('safe', True)

                if not result.get('safe', True) and result.get('alert'):
                    alert_info = {
                        'id': len(_alerts_list) + 1,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'message': result.get('message', f'Unsafe activity: {result["activity"]}'),
                        'severity': result.get('severity', 'high'),
                        'activity': result.get('activity', 'unknown'),
                        'confidence': float(result.get('confidence', 0.0)),
                        'source': 'image_upload',
                    }
                    snapshot = _append_alert(alert_info)
                    with _status_lock:
                        _current_status['alerts'] = snapshot

        if alert_info is not None:
            _dispatch_alert(alert_info)

        with _status_lock:
            status_payload = {
                'activity':    _current_status.get('activity', 'None'),
                'confidence':  _current_status.get('confidence', 0.0),
                'safe':        _current_status.get('safe', True),
                'fps':         _current_status.get('fps', 0.0),
                'source':      'image_upload',
            }
        with _alerts_lock:
            status_payload['alert_count'] = len(_alerts_list)
        socketio.emit('status_update', status_payload)

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = f"uploaded_{ts}.jpg"
        os.makedirs('static/uploads', exist_ok=True)
        cv2.imwrite(os.path.join('static/uploads', fname), frame)

        return jsonify({
            'result': result,
            'image_url': f'/static/uploads/{fname}',
            'status_updated': True,
            'alert_generated': not result.get('safe', True),
        })
    except Exception as e:
        logger.error(f"upload error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/capture', methods=['POST'])
def capture_frame():
    if not monitoring_active:
        return jsonify({'error': 'Monitoring not active'}), 400
    with _last_frame_lock:
        frame = _last_frame_for_capture.copy() if _last_frame_for_capture is not None else None
    if frame is None:
        return jsonify({'error': 'No frame available'}), 500
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = f"capture_{ts}.jpg"
    os.makedirs('captures', exist_ok=True)
    path = os.path.join('captures', fname)
    cv2.imwrite(path, frame)
    return jsonify({'filename': fname, 'path': path, 'url': f'/captures/{fname}'})


@app.route('/api/performance')
def get_performance():
    try:
        return jsonify({
            'health':  perf_monitor.get_health_status(),
            'summary': perf_monitor.get_summary(),
            'history': perf_monitor.get_history(limit=50),
        })
    except Exception as e:
        logger.error(f"perf error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/settings', methods=['GET', 'POST'])
def manage_settings():
    if request.method == 'GET':
        try:
            return jsonify({
                'enabled_methods':      advanced_alert.config.get('enabled_methods', ['desktop']),
                'email':                advanced_alert.config.get('email', {}),
                'telegram':             advanced_alert.config.get('telegram', {}),
                'throttling':           advanced_alert.config.get('throttling', {}),
                'yolo_model':           Config.YOLO_MODEL,
                'confidence_threshold': Config.CONFIDENCE_THRESHOLD,
                'sequence_length':      Config.SEQUENCE_LENGTH,
                'unsafe_activities':    Config.UNSAFE_ACTIVITIES,
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    try:
        data = request.json or {}
        if 'enabled_methods' in data:
            advanced_alert.config['enabled_methods'] = data['enabled_methods']
        for key in ('email', 'telegram', 'throttling'):
            if key in data:
                advanced_alert.config.setdefault(key, {}).update(data[key])
        with _config_write_lock:
            with open('alert_config.json', 'w') as f:
                json.dump(advanced_alert.config, f, indent=2)
            advanced_alert.reload_config()
        return jsonify({'status': 'updated'})
    except Exception as e:
        logger.error(f"settings error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/alerts')
def export_alerts():
    fmt = request.args.get('format', 'json')
    try:
        with _alerts_lock:
            alerts = list(_alerts_list)
        if fmt == 'json':
            return jsonify({'alerts': alerts})
        if fmt == 'csv':
            out = io.StringIO()
            fields = ['id', 'timestamp', 'severity', 'activity', 'message', 'source']
            writer = csv.DictWriter(out, fieldnames=fields)
            writer.writeheader()
            for a in alerts:
                writer.writerow({k: a.get(k, '') for k in fields})
            return Response(
                out.getvalue(),
                mimetype='text/csv',
                headers={'Content-Disposition':
                         f'attachment; filename=alerts_{datetime.now().strftime("%Y%m%d")}.csv'},
            )
        return jsonify({'error': 'Invalid format'}), 400
    except Exception as e:
        logger.error(f"export error: {e}")
        return jsonify({'error': str(e)}), 500


# =====================================================================
#  VIDEO STREAM (MJPEG consumer — no inference here)
# =====================================================================
@app.route('/video_feed')
def video_feed():
    resp = Response(
        generate_video_feed(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )
    resp.headers['Cache-Control']     = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma']            = 'no-cache'
    resp.headers['X-Accel-Buffering'] = 'no'
    resp.headers['Connection']        = 'close'
    return resp


def generate_video_feed():
    """Each consumer gets its own bounded queue; producer fans out."""
    placeholder = np.zeros((Config.FRAME_HEIGHT, Config.FRAME_WIDTH, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Stream idle - press Start",
                (30, Config.FRAME_HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)

    q = queue.Queue(maxsize=1)
    with _subscribers_lock:
        _subscribers.add(q)

    last_placeholder_time = 0.0
    try:
        while True:
            try:
                frame = q.get(timeout=1.0)
            except queue.Empty:
                if not monitoring_active:
                    now = time.time()
                    if now - last_placeholder_time > 0.5:
                        last_placeholder_time = now
                        ok, jpeg = cv2.imencode(
                            '.jpg', placeholder,
                            [cv2.IMWRITE_JPEG_QUALITY, 60]
                        )
                        if ok:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n'
                                   b'Content-Length: ' + str(len(jpeg)).encode() + b'\r\n\r\n'
                                   + jpeg.tobytes() + b'\r\n')
                continue

            ok, jpeg = cv2.imencode(
                '.jpg', frame,
                [cv2.IMWRITE_JPEG_QUALITY, Config.JPEG_QUALITY]
            )
            if not ok:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(jpeg)).encode() + b'\r\n\r\n'
                   + jpeg.tobytes() + b'\r\n')
    finally:
        with _subscribers_lock:
            _subscribers.discard(q)


# =====================================================================
#  STATIC FILE SERVING
# =====================================================================
@app.route('/captures/<filename>')
def serve_capture(filename):
    path = os.path.join('captures', filename)
    if os.path.exists(path):
        return send_file(path, mimetype='image/jpeg')
    return jsonify({'error': 'File not found'}), 404


@app.route('/alerts/<filename>')
def serve_alert_image(filename):
    path = os.path.join('alerts', filename)
    if os.path.exists(path):
        return send_file(path, mimetype='image/jpeg')
    return jsonify({'error': 'File not found'}), 404


# =====================================================================
#  SOCKETIO EVENTS
# =====================================================================
@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'status': 'connected', 'timestamp': datetime.now().isoformat()})


@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")


@socketio.on('start_monitoring')
def handle_start():
    try:
        ok = _start_reader()
        emit('status_update', {'monitoring': bool(ok or monitoring_active)})
    except Exception as e:
        logger.error(f"socket start error: {e}")
        emit('error', {'message': str(e)})


@socketio.on('stop_monitoring')
def handle_stop():
    try:
        _stop_reader()
        emit('status_update', {'monitoring': False})
    except Exception as e:
        logger.error(f"socket stop error: {e}")
        emit('error', {'message': str(e)})


@socketio.on('get_status')
def handle_get_status():
    try:
        with _status_lock:
            snap = {
                'activity':   _current_status['activity'],
                'confidence': _current_status['confidence'],
                'safe':       _current_status['safe'],
                'fps':        _current_status['fps'],
            }
        with _alerts_lock:
            alert_count = len(_alerts_list)
        emit('status_update', {
            'monitoring_active': monitoring_active,
            'activity':   snap['activity'],
            'confidence': snap['confidence'],
            'safe':       snap['safe'],
            'alert_count': alert_count,
            'fps':        snap['fps'],
        })
    except Exception as e:
        emit('error', {'message': str(e)})


# =====================================================================
#  ERROR HANDLERS
# =====================================================================
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


# =====================================================================
#  BROWSER AUTO-OPEN
# =====================================================================
def open_brave_browser(port):
    time.sleep(2)
    url = f'http://localhost:{port}'
    brave_paths = [
        r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
        r"C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\BraveSoftware\Brave-Browser\Application\brave.exe"),
        "brave.exe",
    ]
    for p in brave_paths:
        try:
            if os.path.exists(p):
                webbrowser.register('brave', None, webbrowser.GenericBrowser(p))
                webbrowser.get('brave').open(url)
                logger.info(f"Brave opened at {url}")
                return
        except Exception:
            continue
    logger.warning("Brave not found, using default browser")
    webbrowser.open(url)


def find_available_port(start_port=5000, max_port=5010):
    for port in range(start_port, max_port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    return start_port


# =====================================================================
#  CLEANUP
# =====================================================================
def _shutdown():
    try:
        _stop_reader()
        perf_monitor.stop_monitoring()
    except Exception:
        pass
    try:
        _alert_pool.shutdown(wait=False)
    except Exception:
        pass

atexit.register(_shutdown)


# =====================================================================
#  MAIN
# =====================================================================
def main():
    for d in ('static/uploads', 'captures', 'alerts', 'saved_models', 'data/activities'):
        os.makedirs(d, exist_ok=True)

    host  = Config.FLASK_HOST
    port  = find_available_port(Config.FLASK_PORT)
    debug = False            # MUST be False — reloader spawns a second process
    logger.info(f"Starting on {host}:{port} (debug={debug})")

    threading.Thread(target=open_brave_browser, args=(port,), daemon=True).start()

    try:
        socketio.run(
            app,
            host=host,
            port=port,
            debug=debug,
            use_reloader=False,
            allow_unsafe_werkzeug=True,
        )
    except Exception as e:
        logger.error(f"socketio.run failed: {e}; falling back to app.run")
        try:
            app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)
        except Exception as e2:
            logger.error(f"app.run failed: {e2}")
            port = find_available_port(port + 1)
            logger.info(f"Retrying on port {port}")
            app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)


if __name__ == '__main__':
    main()
