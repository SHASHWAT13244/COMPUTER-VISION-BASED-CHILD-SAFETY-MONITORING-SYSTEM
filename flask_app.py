# flask_app.py
from flask import Flask, render_template, Response, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
from PIL import Image
import io
import json
import os
import time
from datetime import datetime
import threading
import queue
import eventlet

from config import Config
from models.detector import ChildDetector
from models.pose_estimator import PoseEstimator
from models.activity_recognizer import ActivityRecognizer
from models.safety_engine import SafetyEngine
from utils.alert import AlertSystem
from utils.visualization import Visualizer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'child-safety-monitoring-secret-key'
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
    'alerts': []
}

# Initialize components
detector = ChildDetector(model_path=Config.YOLO_MODEL, conf_threshold=Config.CONFIDENCE_THRESHOLD)
pose_estimator = PoseEstimator()
activity_recognizer = ActivityRecognizer(
    sequence_length=Config.SEQUENCE_LENGTH,
    num_keypoints=33,
    num_classes=len(Config.ACTIVITY_CLASSES)
)
safety_engine = SafetyEngine()
alert_system = AlertSystem(sound_enabled=False, display_enabled=True, log_enabled=True)
visualizer = Visualizer()

# Load pre-trained model if exists
model_path = os.path.join(Config.MODELS_DIR, 'activity_model.pth')
if os.path.exists(model_path):
    try:
        activity_recognizer.load_model(model_path)
        print(f"Loaded pre-trained model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")

# Global variables for processing
keypoint_buffer = []
alerts_list = []
frame_count = 0
start_time = time.time()

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

@app.route('/api/status')
def get_status():
    """Get current system status"""
    stats = alert_system.get_alert_statistics()
    return jsonify({
        'monitoring_active': monitoring_active,
        'activity': current_status['activity'],
        'confidence': current_status['confidence'],
        'safe': current_status['safe'],
        'alerts': current_status['alerts'][-10:],
        'statistics': stats,
        'fps': get_fps()
    })

@app.route('/api/alerts')
def get_alerts():
    """Get all alerts"""
    return jsonify({
        'alerts': current_status['alerts'][-50:],
        'total': len(current_status['alerts'])
    })

@app.route('/api/start', methods=['POST'])
def start_monitoring():
    """Start monitoring"""
    global monitoring_active, camera
    
    if not monitoring_active:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            return jsonify({'error': 'Could not open camera'}), 500
        
        monitoring_active = True
        threading.Thread(target=process_video, daemon=True).start()
        socketio.emit('status_update', {'monitoring': True})
        return jsonify({'status': 'started'})
    
    return jsonify({'status': 'already_running'})

@app.route('/api/stop', methods=['POST'])
def stop_monitoring():
    """Stop monitoring"""
    global monitoring_active, camera
    
    monitoring_active = False
    if camera:
        camera.release()
        camera = None
    
    socketio.emit('status_update', {'monitoring': False})
    return jsonify({'status': 'stopped'})

@app.route('/api/reset', methods=['POST'])
def reset_system():
    """Reset system state"""
    global keypoint_buffer, alerts_list, current_status
    
    keypoint_buffer = []
    alerts_list = []
    current_status['alerts'] = []
    current_status['activity'] = 'None'
    current_status['confidence'] = 0.0
    current_status['safe'] = True
    
    socketio.emit('status_update', {'reset': True})
    return jsonify({'status': 'reset'})

@app.route('/api/upload', methods=['POST'])
def upload_image():
    """Upload and process an image"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    # Read image
    image_bytes = file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return jsonify({'error': 'Invalid image'}), 400
    
    # Process the image
    result = process_single_image(frame)
    
    # Save uploaded image
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"uploaded_{timestamp}.jpg"
    filepath = os.path.join('static/uploads', filename)
    os.makedirs('static/uploads', exist_ok=True)
    cv2.imwrite(filepath, frame)
    
    return jsonify({
        'result': result,
        'image_url': f'/static/uploads/{filename}'
    })

@app.route('/api/capture', methods=['POST'])
def capture_frame():
    """Capture and save current frame"""
    if not monitoring_active or camera is None:
        return jsonify({'error': 'Monitoring not active'}), 400
    
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

@app.route('/captures/<filename>')
def serve_capture(filename):
    """Serve captured image"""
    filepath = os.path.join('captures', filename)
    if os.path.exists(filepath):
        return send_file(filepath, mimetype='image/jpeg')
    return jsonify({'error': 'File not found'}), 404

@app.route('/video_feed')
def video_feed():
    """Video streaming endpoint"""
    return Response(generate_video_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_video_feed():
    """Generate video feed for streaming"""
    global camera, monitoring_active
    
    while monitoring_active:
        if camera is None:
            break
        
        ret, frame = camera.read()
        if not ret:
            break
        
        # Process frame
        processed_frame = process_frame_for_stream(frame)
        
        # Encode as JPEG
        ret, jpeg = cv2.imencode('.jpg', processed_frame)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + 
                   jpeg.tobytes() + b'\r\n\r\n')
        
        time.sleep(0.03)
    
    # Send empty frame when stopped
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    ret, jpeg = cv2.imencode('.jpg', blank)
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + 
           jpeg.tobytes() + b'\r\n\r\n')

def process_video():
    """Background video processing thread"""
    global camera, monitoring_active, keypoint_buffer, alerts_list, frame_count
    
    while monitoring_active and camera is not None:
        try:
            ret, frame = camera.read()
            if not ret:
                continue
            
            frame_count += 1
            
            # Process frame
            result = process_single_frame(frame)
            
            # Update current status
            current_status['activity'] = result['activity']
            current_status['confidence'] = result['confidence']
            current_status['safe'] = result['safe']
            
            # Handle alerts
            if not result['safe'] and result.get('alert'):
                alert_info = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'message': result['message'],
                    'severity': result['severity'],
                    'activity': result['activity']
                }
                current_status['alerts'].append(alert_info)
                alerts_list.append(alert_info)
                
                # Emit alert via SocketIO
                socketio.emit('alert', alert_info)
                
                # Log alert
                alert_system.generate_alert(
                    message=result['message'],
                    severity=result['severity'],
                    activity=result['activity']
                )
            
            # Emit status update
            if frame_count % 5 == 0:  # Reduce frequency
                socketio.emit('status_update', {
                    'activity': result['activity'],
                    'confidence': result['confidence'],
                    'safe': result['safe'],
                    'alert_count': len(current_status['alerts'])
                })
            
            # Clear buffer if too large
            if len(keypoint_buffer) > 100:
                keypoint_buffer = keypoint_buffer[-50:]
            
        except Exception as e:
            print(f"Error in video processing: {e}")
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
        'alert': None
    }
    
    # Detect children
    detections = detector.detect(frame)
    
    if not detections:
        return result
    
    # Process first detection
    detection = detections[0]
    bbox = detection['bbox']
    
    # Extract pose
    keypoints = pose_estimator.extract_keypoints(frame)
    
    if keypoints is not None:
        # Add to buffer
        keypoints_flat = keypoints[:, :3].flatten()
        keypoint_buffer.append(keypoints_flat)
        
        # Keep buffer size
        if len(keypoint_buffer) > Config.SEQUENCE_LENGTH * 2:
            keypoint_buffer = keypoint_buffer[-Config.SEQUENCE_LENGTH:]
        
        # Predict activity
        if len(keypoint_buffer) >= Config.SEQUENCE_LENGTH:
            activity, confidence = activity_recognizer.predict_activity(keypoint_buffer)
            result['activity'] = activity or 'Unknown'
            result['confidence'] = confidence
        else:
            result['activity'] = 'Collecting data...'
            result['confidence'] = 0.0
    else:
        # No pose detected
        if keypoint_buffer:
            # Add zeros to maintain sequence
            keypoint_buffer.append(np.zeros(33 * 3))
            if len(keypoint_buffer) > Config.SEQUENCE_LENGTH * 2:
                keypoint_buffer = keypoint_buffer[-Config.SEQUENCE_LENGTH:]
    
    # Check safety
    if result['activity'] != 'Collecting data...' and result['activity'] != 'None':
        safety_result = safety_engine.check_safety(
            activity=result['activity'],
            confidence=result['confidence'],
            pose_keypoints=keypoints,
            bbox=bbox
        )
        
        result['safe'] = safety_result['safe']
        if not result['safe']:
            result['message'] = safety_result.get('message', 'Unsafe behavior detected')
            result['severity'] = safety_result.get('severity', 'high')
            result['alert'] = safety_result.get('alert')
    
    return result

def process_single_image(frame):
    """Process a single image"""
    result = process_single_frame(frame)
    
    # Add visualization details
    result['has_child'] = len(detector.detect(frame)) > 0
    
    # Get processed image with annotations
    annotated_frame = frame.copy()
    
    # Draw detections
    annotated_frame, detections = detector.detect_and_draw(annotated_frame)
    
    # Draw pose
    if len(detections) > 0:
        annotated_frame = pose_estimator.draw_pose(annotated_frame)
    
    # Draw status
    annotated_frame = visualizer.draw_status(
        annotated_frame,
        result['activity'],
        result['confidence'],
        result['safe'],
        []
    )
    
    # Convert to base64 for display
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    result['processed_image'] = image_base64
    
    return result

def process_frame_for_stream(frame):
    """Process frame for streaming display"""
    result = process_single_frame(frame)
    
    # Draw annotations
    annotated_frame = frame.copy()
    
    # Detect and draw
    annotated_frame, detections = detector.detect_and_draw(annotated_frame)
    
    # Draw pose
    if detections:
        annotated_frame = pose_estimator.draw_pose(annotated_frame)
    
    # Draw status
    annotated_frame = visualizer.draw_status(
        annotated_frame,
        result['activity'],
        result['confidence'],
        result['safe'],
        current_status['alerts'][-3:] if current_status['alerts'] else []
    )
    
    return annotated_frame

def get_fps():
    """Calculate FPS"""
    global frame_count, start_time
    elapsed = time.time() - start_time
    if elapsed > 0:
        fps = frame_count / elapsed
        return round(fps, 1)
    return 0

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('connected', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('start_monitoring')
def handle_start():
    """Handle start monitoring from socket"""
    start_monitoring()

@socketio.on('stop_monitoring')
def handle_stop():
    """Handle stop monitoring from socket"""
    stop_monitoring()

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('captures', exist_ok=True)
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
