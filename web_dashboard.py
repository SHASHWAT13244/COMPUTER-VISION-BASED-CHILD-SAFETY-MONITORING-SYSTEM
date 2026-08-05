"""
Web Dashboard for Child Safety Monitoring System
"""

from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import base64
import json
import threading
import time
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class WebDashboard:
    """Web dashboard for remote monitoring"""
    
    def __init__(self, monitoring_system):
        """
        Initialize web dashboard
        
        Args:
            monitoring_system: Reference to the main monitoring system
        """
        self.system = monitoring_system
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Setup routes
        self.setup_routes()
        self.setup_socket_events()
        
        # State
        self.clients = set()
        self.last_frame = None
        self.frame_lock = threading.Lock()
        
        logger.info("Web Dashboard initialized")
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main dashboard page"""
            return render_template('dashboard.html')
        
        @self.app.route('/video_feed')
        def video_feed():
            """Video feed endpoint"""
            return Response(self.generate_frames(), 
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @self.app.route('/api/status')
        def get_status():
            """Get system status"""
            if self.system:
                return jsonify({
                    'status': 'running',
                    'frame_count': self.system.frame_count,
                    'activity_buffer': self.system.activity_buffer[-10:],
                    'alert_history': self.system.alert_system.get_alert_history(limit=10)
                })
            return jsonify({'status': 'not_initialized'})
        
        @self.app.route('/api/alert', methods=['POST'])
        def send_alert():
            """Send alert via API"""
            data = request.json
            if self.system:
                success = self.system.alert_system.send_alert(
                    alert_type=data.get('type', 'api'),
                    message=data.get('message', 'Alert from API'),
                    severity=data.get('severity', 'medium')
                )
                return jsonify({'success': success})
            return jsonify({'success': False, 'error': 'System not initialized'})
    
    def setup_socket_events(self):
        """Setup SocketIO events"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            self.clients.add(request.sid)
            logger.info(f"Client connected: {request.sid}")
            emit('connection_response', {'status': 'connected'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            if request.sid in self.clients:
                self.clients.remove(request.sid)
            logger.info(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('get_alert_history')
        def handle_get_alerts():
            """Send alert history to client"""
            if self.system:
                history = self.system.alert_system.get_alert_history(limit=20)
                emit('alert_history', history)
    
    def generate_frames(self):
        """Generate MJPEG frames for streaming"""
        while True:
            with self.frame_lock:
                if self.last_frame is not None:
                    # Convert frame to JPEG
                    ret, jpeg = cv2.imencode('.jpg', self.last_frame)
                    if ret:
                        frame_data = jpeg.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + 
                               frame_data + b'\r\n')
            time.sleep(0.033)  # ~30 fps
    
    def update_frame(self, frame):
        """Update the latest frame for streaming"""
        with self.frame_lock:
            self.last_frame = frame
    
    def broadcast_alert(self, alert):
        """Broadcast alert to all connected clients"""
        self.socketio.emit('new_alert', alert)
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the web dashboard"""
        logger.info(f"Starting web dashboard on http://{host}:{port}")
        self.socketio.run(self.app, host=host, port=port, debug=debug)
    
    def create_dashboard_html(self):
        """Create the dashboard HTML template"""
        html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Child Safety Monitoring Dashboard</title>
    <script src="https://cdn.socket.io/4.5.0/socket.io.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: Arial, sans-serif; 
            background: #1a1a2e; 
            color: white;
            padding: 20px;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
        }
        h1 { 
            text-align: center; 
            margin-bottom: 20px;
            color: #00d4ff;
        }
        .dashboard { 
            display: grid; 
            grid-template-columns: 2fr 1fr; 
            gap: 20px; 
        }
        .video-container { 
            background: #16213e; 
            border-radius: 10px; 
            padding: 10px;
        }
        .video-container img { 
            width: 100%; 
            border-radius: 5px; 
        }
        .sidebar { 
            background: #16213e; 
            border-radius: 10px; 
            padding: 20px;
        }
        .status-indicator {
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 15px;
            text-align: center;
            font-weight: bold;
        }
        .status-safe { background: #2ecc71; color: #1a1a2e; }
        .status-warning { background: #f1c40f; color: #1a1a2e; }
        .status-danger { background: #e74c3c; color: white; }
        
        .alert-list {
            max-height: 300px;
            overflow-y: auto;
        }
        .alert-item {
            padding: 8px 12px;
            margin: 5px 0;
            border-radius: 5px;
            background: #1a1a2e;
            border-left: 4px solid #3498db;
        }
        .alert-high { border-left-color: #e74c3c; }
        .alert-medium { border-left-color: #f1c40f; }
        .alert-low { border-left-color: #2ecc71; }
        .alert-time { font-size: 11px; color: #888; }
        
        .stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin: 15px 0;
        }
        .stat-card {
            background: #1a1a2e;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #00d4ff;
        }
        .stat-label {
            font-size: 12px;
            color: #888;
            margin-top: 5px;
        }
        .controls {
            display: flex;
            gap: 10px;
            margin: 15px 0;
            flex-wrap: wrap;
        }
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: transform 0.2s;
        }
        .btn:hover { transform: scale(1.05); }
        .btn-primary { background: #3498db; color: white; }
        .btn-danger { background: #e74c3c; color: white; }
        .btn-success { background: #2ecc71; color: white; }
        .btn-warning { background: #f1c40f; color: #1a1a2e; }
        
        .activity-history {
            margin-top: 15px;
        }
        .activity-item {
            display: inline-block;
            padding: 4px 12px;
            margin: 2px;
            border-radius: 12px;
            font-size: 12px;
            background: #2c3e50;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        .pulse { animation: pulse 2s infinite; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚸 Child Safety Monitoring System</h1>
        
        <div class="dashboard">
            <div class="video-container">
                <img id="video_feed" src="/video_feed" alt="Video Feed">
                <div style="display: flex; justify-content: space-between; padding: 10px 0;">
                    <span>FPS: <span id="fps_display">0</span></span>
                    <span>Frame: <span id="frame_display">0</span></span>
                    <span id="status_display" class="pulse" style="color: #2ecc71;">● SAFE</span>
                </div>
            </div>
            
            <div class="sidebar">
                <div id="status_indicator" class="status-indicator status-safe">🟢 SYSTEM SAFE</div>
                
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-value" id="activity_count">0</div>
                        <div class="stat-label">Current Activity</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="alert_count">0</div>
                        <div class="stat-label">Total Alerts</div>
                    </div>
                </div>
                
                <div class="controls">
                    <button class="btn btn-primary" onclick="testAlert()">🔔 Test Alert</button>
                    <button class="btn btn-warning" onclick="toggleRecording()">⏺ Toggle Record</button>
                    <button class="btn btn-danger" onclick="resetSystem()">🔄 Reset</button>
                </div>
                
                <div class="activity-history" id="activity_history">
                    <h3>Activity History</h3>
                    <div id="activity_container"></div>
                </div>
                
                <div style="margin-top: 15px;">
                    <h3>Recent Alerts</h3>
                    <div class="alert-list" id="alert_list"></div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const socket = io();
        
        // Alert container
        const alertList = document.getElementById('alert_list');
        const activityContainer = document.getElementById('activity_container');
        
        // Socket events
        socket.on('new_alert', function(alert) {
            addAlert(alert);
        });
        
        socket.on('alert_history', function(alerts) {
            alertList.innerHTML = '';
            alerts.forEach(alert => addAlert(alert));
        });
        
        function addAlert(alert) {
            const div = document.createElement('div');
            div.className = `alert-item alert-${alert.severity}`;
            div.innerHTML = `
                <strong>${alert.type.toUpperCase()}</strong> - ${alert.message}
                <br><span class="alert-time">${new Date(alert.timestamp).toLocaleString()}</span>
            `;
            alertList.insertBefore(div, alertList.firstChild);
            
            if (alertList.children.length > 20) {
                alertList.removeChild(alertList.lastChild);
            }
        }
        
        // Update status
        function updateStatus(status) {
            const indicator = document.getElementById('status_indicator');
            const display = document.getElementById('status_display');
            
            if (status === 'danger') {
                indicator.className = 'status-indicator status-danger';
                indicator.innerHTML = '🔴 DANGER - Immediate Action Required';
                display.textContent = '● DANGER';
                display.style.color = '#e74c3c';
            } else if (status === 'warning') {
                indicator.className = 'status-indicator status-warning';
                indicator.innerHTML = '🟡 WARNING - Caution Advised';
                display.textContent = '● WARNING';
                display.style.color = '#f1c40f';
            } else {
                indicator.className = 'status-indicator status-safe';
                indicator.innerHTML = '🟢 SYSTEM SAFE';
                display.textContent = '● SAFE';
                display.style.color = '#2ecc71';
            }
        }
        
        // Activity history
        function addActivity(activity) {
            const span = document.createElement('span');
            span.className = 'activity-item';
            span.textContent = activity;
            activityContainer.appendChild(span);
            
            if (activityContainer.children.length > 30) {
                activityContainer.removeChild(activityContainer.firstChild);
            }
        }
        
        // Controls
        function testAlert() {
            fetch('/api/alert', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    type: 'test',
                    message: 'Test alert from dashboard',
                    severity: 'medium'
                })
            });
        }
        
        function toggleRecording() {
            // Implement recording toggle
            const btn = event.target;
            if (btn.textContent.includes('⏺')) {
                btn.textContent = '⏹ Stop Recording';
                btn.className = 'btn btn-danger';
            } else {
                btn.textContent = '⏺ Toggle Record';
                btn.className = 'btn btn-warning';
            }
        }
        
        function resetSystem() {
            if (confirm('Reset the system?')) {
                alertList.innerHTML = '';
                activityContainer.innerHTML = '';
                updateStatus('safe');
            }
        }
        
        // Fetch status periodically
        setInterval(() => {
            fetch('/api/status')
                .then(res => res.json())
                .then(data => {
                    document.getElementById('frame_display').textContent = data.frame_count || 0;
                    
                    // Update activity
                    if (data.activity_buffer && data.activity_buffer.length > 0) {
                        const lastActivity = data.activity_buffer[data.activity_buffer.length - 1];
                        document.getElementById('activity_count').textContent = lastActivity || 'unknown';
                        addActivity(lastActivity || 'unknown');
                    }
                    
                    // Update alert count
                    if (data.alert_history) {
                        document.getElementById('alert_count').textContent = data.alert_history.length;
                    }
                })
                .catch(err => console.error('Error fetching status:', err));
        }, 2000);
        
        // Update FPS
        let frameCount = 0;
        setInterval(() => {
            document.getElementById('fps_display').textContent = frameCount;
            frameCount = 0;
        }, 1000);
        
        // Track frames (simplified)
        const videoFeed = document.getElementById('video_feed');
        videoFeed.addEventListener('load', () => {
            frameCount++;
        });
        
        console.log('Dashboard loaded successfully');
    </script>
</body>
</html>
"""
        return html
    
    def save_dashboard_template(self):
        """Save dashboard HTML template to file"""
        template_dir = Path(__file__).resolve().parent / 'templates'
        template_dir.mkdir(exist_ok=True)
        
        template_path = template_dir / 'dashboard.html'
        with open(template_path, 'w') as f:
            f.write(self.create_dashboard_html())
        
        logger.info(f"Dashboard template saved to {template_path}")
        return template_path