👶 Child Safety Monitoring System
A Computer Vision-Based Real-Time Child Safety Monitoring System that uses Deep Learning to detect and analyze children's activities through a webcam, generating alerts for unsafe behaviors.

📋 Table of Contents
Project Overview

Team Members

Quick Start

Project Structure

How It Works

Configuration

Training

Alert System

Performance

Technologies Used

Future Scope

License

🎯 Project Overview
This system monitors children in real-time using computer vision and deep learning. It can detect children, estimate their pose, recognize activities, and generate alerts for unsafe behaviors.

✨ Key Features
Feature	Description
🎥 Real-time Monitoring	Live video feed with annotations
👤 Child Detection	YOLOv8-based person detection
🦴 Pose Estimation	MediaPipe-based 33-keypoint pose extraction
🏃 Activity Recognition	LSTM-based activity classification
⚠️ Safety Engine	Rule-based unsafe behavior detection
🔔 Alert System	Multi-channel notifications
📊 Dashboard	Web-based monitoring dashboard
📸 Image Analysis	Upload and analyze single images
💾 Recording	Save video recordings and capture frames
🎯 Detected Activities
Activity	Status	Description
🚶 Walking	✅ Safe	Normal walking behavior
🏃 Running	⚠️ Caution	Running - potential injury risk
🪑 Sitting	✅ Safe	Sitting position
💫 Falling	🔴 Unsafe	Fall detected - immediate attention required
🧗 Climbing	🔴 Unsafe	Climbing on unsafe surfaces
👥 Team Members
Name	Roll Number	Role
Arpit Ojha	-	Team Member
Aashutosh Vaish	-	Team Member
Shivansh Mishra	303302223199	Team Member
Shashwat Khandelwal	303302223197	Team Member
👨‍🏫 Project Guide
Mrs. Poonam Gupta
Assistant Professor, Department of Computer Science & Engineering
SSIPMT, Raipur

📚 Project Details
Attribute	Value
Semester	7th
Batch	2023-27
Session	July-Dec 2026
Institution	SSIPMT, Raipur
🚀 Quick Start
📋 Prerequisites
Python 3.10 or higher

Webcam

8GB+ RAM (recommended)

GPU (optional, for better performance)

💻 Installation
bash
# Clone or download the project
cd child_safety_monitoring

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download YOLO model (automatically downloaded on first run)
▶️ Running the System
bash
# Web Interface (Recommended)
python run.py --mode web

# Console Monitoring
python run.py --mode monitor

# Train the model
python run.py --mode train

# Prepare training data
python run.py --mode prepare

# Generate synthetic data
python run.py --mode synthetic

# Run tests
python run.py --mode test
🌐 Access Web Interface
Open your browser and navigate to: http://localhost:5000

📁 Project Structure
bash
child_safety_monitoring/
│
├── app.py                          # Main console application
├── flask_app.py                    # Flask web application
├── config.py                       # Configuration settings
├── data_preparation.py             # Data preparation script
├── data_augmentation.py            # Data augmentation
├── requirements.txt                # Python dependencies
├── run.py                          # Launcher script
├── setup.py                        # Package setup
├── alert_config.example.json       # Example alert config
├── .gitignore                      # Git ignore file
├── README.md                       # Documentation
│
├── models/                         # ML Models
│   ├── __init__.py
│   ├── detector.py                 # YOLOv8 child detection
│   ├── pose_estimator.py           # MediaPipe pose estimation
│   ├── activity_recognizer.py      # LSTM activity recognition
│   ├── safety_engine.py            # Safety rule engine
│   └── tracker.py                  # Multi-person tracking
│
├── utils/                          # Utilities
│   ├── __init__.py
│   ├── alert.py                    # Basic alert system
│   ├── alert_advanced.py           # Advanced alert with email/SMS
│   ├── visualization.py            # Visualization utilities
│   └── performance_monitor.py      # Performance monitoring
│
├── static/                         # Web static files
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   ├── script.js
│   │   ├── dashboard.js
│   │   └── alerts.js
│   └── uploads/                    # Uploaded images
│
├── templates/                      # HTML templates
│   ├── index.html                  # Home page
│   ├── dashboard.html              # Dashboard page
│   ├── alerts.html                 # Alerts page
│   ├── about.html                  # About page
│   └── settings.html               # Settings page
│
├── tests/                          # Tests
│   ├── __init__.py
│   └── test_system.py              # System tests
│
├── saved_models/                   # Trained models
├── data/
│   └── activities/                 # Training videos
│       ├── walking/
│       ├── running/
│       ├── sitting/
│       ├── falling/
│       └── climbing/
│
├── captures/                       # Captured frames
├── alerts/                         # Alert images
└── recordings/                     # Recorded videos
🧠 How It Works
📊 Processing Pipeline






🔄 Step-by-Step Flow
1. 📸 Detection (YOLOv8)
Detects persons in the frame

Returns bounding boxes with confidence scores

2. 🦴 Pose Estimation (MediaPipe)
Extracts 33 body keypoints (x, y, z, visibility)

Tracks body position and movement

3. 🧠 Activity Recognition (LSTM)
Buffers 30 frames of keypoint sequences

Classifies activity: walking, running, sitting, falling, climbing

4. ⚠️ Safety Engine
Checks if activity is unsafe

Additional pose-based fall detection

Zone violation detection

5. 🔔 Alert System
Generates alerts for unsafe behaviors

Multi-channel notifications

Throttling to prevent spam

🎮 Keyboard Shortcuts
Key	Action
q	Quit monitoring
s	Save current frame
r	Reset system state
v	Toggle visualization info
🔧 Configuration
Edit config.py to customize system parameters:

Model Settings
python
YOLO_MODEL = 'yolov8n.pt'        # YOLO model variant
CONFIDENCE_THRESHOLD = 0.5       # Detection threshold
SEQUENCE_LENGTH = 30              # Frames per sequence
Camera Settings
python
FRAME_WIDTH = 640                # Processing width
FRAME_HEIGHT = 480               # Processing height
FPS = 30                         # Camera FPS
Safety Rules
python
UNSAFE_ACTIVITIES = ['falling', 'climbing']
FALL_DETECTION_THRESHOLD = 0.6
Alert Settings
python
ALERT_COOLDOWN_SECONDS = 5
MAX_ALERTS_PER_MINUTE = 10
📊 Training
📁 Prepare Training Data
Add videos to data/activities/[activity_name]/

Extract features:

bash
python data_preparation.py --process
🎲 Generate Synthetic Data
bash
python data_preparation.py --synthetic 200
🏋️ Train Model
bash
python run.py --mode train
🔔 Alert System
📝 Configuration
Create alert_config.json (copy from alert_config.example.json):

json
{
    "enabled_methods": ["desktop", "email", "telegram"],
    "email": {
        "smtp_server": "smtp.gmail.com",
        "sender_email": "your_email@gmail.com",
        "sender_password": "your_app_password",
        "recipient_emails": ["recipient@email.com"]
    },
    "telegram": {
        "bot_token": "your_bot_token",
        "chat_ids": ["chat_id"]
    },
    "throttling": {
        "max_alerts_per_minute": 10,
        "cooldown_seconds": 5
    }
}
📧 Supported Methods
Method	Description
💻 Desktop	Console notification with sound
📧 Email	SMTP email alerts
📱 SMS	Twilio SMS alerts
📨 Telegram	Telegram bot messages
🔗 Webhook	HTTP callbacks
🧪 Testing
Run Test Suite
bash
python run.py --mode test
Manual Testing
bash
python -m pytest tests/ -v
📈 Performance
💻 System Requirements
Component	Minimum	Recommended
CPU	Intel i5	Intel i7 / AMD Ryzen 7
RAM	8GB	16GB
GPU	None	NVIDIA GTX 1060+
Storage	2GB	10GB
📊 Performance Metrics
Operation	CPU Only	With GPU
Detection	10-15 FPS	25-30 FPS
Pose Estimation	15-20 FPS	30+ FPS
Activity Recognition	20-25 FPS	30+ FPS
Overall	8-12 FPS	20-25 FPS
🛠️ Technologies Used
Technology	Purpose
Python 3.10+	Core programming language
OpenCV	Video processing and computer vision
YOLOv8	Child/person detection
MediaPipe	Pose estimation
PyTorch	Deep learning framework for LSTM
Flask	Web interface framework
Socket.IO	Real-time updates
Bootstrap	Frontend styling
Chart.js	Dashboard charts
🔮 Future Scope
□ Mobile application for remote monitoring
□ Cloud-based storage for recordings
□ Multi-camera support
□ Facial recognition for child identification
□ Voice alerts
□ Integration with smart home devices
□ Real-time notification via SMS/WhatsApp
□ Advanced analytics and reporting
📝 License
This project is developed as part of the 7th Semester Project at SSIPMT, Raipur.

👥 Team Members
Arpit Ojha

Aashutosh Vaish

Shivansh Mishra (303302223199)

Shashwat Khandelwal (303302223197)

👨‍🏫 Project Guide
Mrs. Poonam Gupta, Assistant Professor, CSE

🙏 Acknowledgments
YOLOv8 by Ultralytics

MediaPipe by Google

PyTorch by Meta

Flask for web framework

📚 References
Ultralytics YOLOv8 Documentation

MediaPipe Pose Estimation Guide

PyTorch LSTM Documentation

Flask Web Framework Documentation

📞 Contact
For questions or support, please contact the project team.

<div align="center"> <p> <b>SSIPMT, Raipur | 7th Semester | 2023-27</b><br> <i>Computer Vision-Based Child Safety Monitoring System</i> </p> <br> <p> <a href="#"><img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python"></a> <a href="#"><img src="https://img.shields.io/badge/OpenCV-4.8.1-green.svg" alt="OpenCV"></a> <a href="#"><img src="https://img.shields.io/badge/PyTorch-2.1.0-red.svg" alt="PyTorch"></a> <a href="#"><img src="https://img.shields.io/badge/Flask-2.3.3-lightgrey.svg" alt="Flask"></a> <a href="#"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a> </p> <p> ⭐ If you find this project useful, please star it on GitHub! ⭐ </p> </div>
