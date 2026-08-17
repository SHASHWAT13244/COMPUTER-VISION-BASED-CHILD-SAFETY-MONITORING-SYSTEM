# 👶 Computer Vision-Based Child Safety Monitoring System

> **A real-time AI-powered child safety monitoring system using computer vision and deep learning to detect activities, identify potentially unsafe behavior, and generate alerts.**

**SSIPMT, Raipur | 7th Semester Project | Batch 2023–27 | Session: July–December 2026**

---

## 📌 Table of Contents

* [Overview](#-overview)
* [Key Features](#-key-features)
* [Detected Activities](#-detected-activities)
* [System Architecture](#-system-architecture)
* [How It Works](#-how-it-works)
* [Project Structure](#-project-structure)
* [Technologies Used](#-technologies-used)
* [Requirements](#-requirements)
* [Installation](#-installation)
* [Running the Application](#-running-the-application)
* [Web Interface](#-web-interface)
* [Configuration](#-configuration)
* [Training](#-training)
* [Alert System](#-alert-system)
* [Testing](#-testing)
* [Performance](#-performance)
* [Keyboard Shortcuts](#-keyboard-shortcuts)
* [Team Members](#-team-members)
* [Project Guide](#-project-guide)
* [Future Scope](#-future-scope)
* [Acknowledgments](#-acknowledgments)
* [References](#-references)
* [License](#-license)

---

## 🎯 Overview

The **Computer Vision-Based Child Safety Monitoring System** is an AI-powered application designed to monitor children's activities in real time using a webcam.

The system combines:

* 🎥 Real-time video processing
* 👤 Person/child detection
* 🦴 Human pose estimation
* 🧠 Deep-learning-based activity recognition
* ⚠️ Rule-based safety analysis
* 🔔 Multi-channel alerts
* 📊 Web-based monitoring
* 📸 Image analysis
* 💾 Video recording and frame capture

The primary objective is to identify potentially unsafe activities such as **falling** and **climbing** and notify the responsible person as quickly as possible.

> **Note:** The detection pipeline uses a YOLO-based person detector. If child-specific detection is required, the detector can be fine-tuned on a child-specific dataset.

---

## ✨ Key Features

| Feature                  | Description                                                 |
| ------------------------ | ----------------------------------------------------------- |
| 🎥 Real-Time Monitoring  | Processes live webcam footage and displays annotated frames |
| 👤 Person Detection      | YOLOv8-based detection of people in the scene               |
| 🦴 Pose Estimation       | MediaPipe extracts 33 body landmarks                        |
| 🏃 Activity Recognition  | LSTM classifies sequences of body movements                 |
| ⚠️ Safety Engine         | Applies rules to identify potentially unsafe behavior       |
| 🔔 Alert System          | Supports desktop, email, Telegram, SMS, and webhook alerts  |
| 📊 Dashboard             | Web-based interface for monitoring activities and alerts    |
| 📸 Image Analysis        | Upload and analyze individual images                        |
| 💾 Recording             | Save monitoring videos and important frames                 |
| 👥 Multi-Person Tracking | Tracks multiple people within the camera view               |
| 🚦 Alert Throttling      | Prevents excessive repeated notifications                   |

---

## 🏃 Detected Activities

| Activity    | Status     | Description                                  |
| ----------- | ---------- | -------------------------------------------- |
| 🚶 Walking  | 🟢 Safe    | Normal walking behavior                      |
| 🏃 Running  | 🟡 Caution | Running may increase the risk of injury      |
| 🪑 Sitting  | 🟢 Safe    | Normal sitting position                      |
| 💫 Falling  | 🔴 Unsafe  | Potential fall requiring immediate attention |
| 🧗 Climbing | 🔴 Unsafe  | Potentially dangerous climbing behavior      |

---

## 🏗️ System Architecture

```text
                  ┌─────────────────────┐
                  │   Webcam / Image    │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   OpenCV Processing │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │     YOLOv8          │
                  │  Person Detection   │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   MediaPipe Pose    │
                  │  33 Body Keypoints  │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   Frame Sequence    │
                  │    Buffer (30)      │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   LSTM Activity     │
                  │    Recognition      │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │    Safety Engine    │
                  │ Rules + Pose Check  │
                  └──────────┬──────────┘
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
             ┌─────────────┐   ┌─────────────┐
             │ Safe/Caution│   │ Unsafe Event│
             └─────────────┘   └──────┬──────┘
                                       │
                                       ▼
                              ┌────────────────┐
                              │  Alert System  │
                              └───────┬────────┘
                                      │
                 ┌────────────┬───────┼────────┬──────────┐
                 ▼            ▼       ▼        ▼          ▼
              Desktop      Email   Telegram   SMS      Webhook
```

---

## 🧠 How It Works

### 1. 📸 Frame Acquisition

The webcam continuously captures video frames using OpenCV.

The frames are resized and preprocessed before being passed to the detection pipeline.

### 2. 👤 Person Detection

YOLOv8 identifies people within each frame and provides:

* Bounding boxes
* Confidence scores
* Object classes

### 3. 🦴 Pose Estimation

MediaPipe Pose extracts **33 body landmarks** for each detected person.

Each landmark contains:

* `x` coordinate
* `y` coordinate
* `z` coordinate
* Visibility score

These landmarks provide the information required for activity recognition.

### 4. 🧠 Activity Recognition

A sequence of pose information is collected over multiple frames.

The default configuration uses:

```text
Sequence Length = 30 frames
```

The LSTM model analyzes the temporal movement pattern and classifies it as:

```text
Walking
Running
Sitting
Falling
Climbing
```

### 5. ⚠️ Safety Analysis

The Safety Engine combines activity predictions with additional rules.

Examples include:

* Fall detection
* Unsafe climbing detection
* Zone violations
* Activity confidence checks
* Alert cooldowns

### 6. 🔔 Alert Generation

When an unsafe event is detected, the system generates an alert.

Depending on configuration, alerts can be delivered through:

* Desktop notification
* Email
* Telegram
* SMS
* Webhook

Alert throttling is implemented to reduce repeated notifications.

---

## 📁 Project Structure

```text
child_safety_monitoring/
│
├── app.py
├── flask_app.py
├── config.py
├── data_preparation.py
├── data_augmentation.py
├── requirements.txt
├── run.py
├── setup.py
├── alert_config.example.json
├── README.md
├── .gitignore
│
├── models/
│   ├── __init__.py
│   ├── detector.py
│   ├── pose_estimator.py
│   ├── activity_recognizer.py
│   ├── safety_engine.py
│   └── tracker.py
│
├── utils/
│   ├── __init__.py
│   ├── alert.py
│   ├── alert_advanced.py
│   ├── visualization.py
│   └── performance_monitor.py
│
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   ├── script.js
│   │   ├── dashboard.js
│   │   └── alerts.js
│   └── uploads/
│
├── templates/
│   ├── index.html
│   ├── dashboard.html
│   ├── alerts.html
│   ├── about.html
│   └── settings.html
│
├── tests/
│   ├── __init__.py
│   └── test_system.py
│
├── saved_models/
│
├── data/
│   └── activities/
│       ├── walking/
│       ├── running/
│       ├── sitting/
│       ├── falling/
│       └── climbing/
│
├── captures/
├── alerts/
└── recordings/
```

---

## 🛠️ Technologies Used

| Technology       | Purpose                              |
| ---------------- | ------------------------------------ |
| **Python 3.10+** | Core programming language            |
| **OpenCV**       | Video processing and computer vision |
| **YOLOv8**       | Person detection                     |
| **MediaPipe**    | Human pose estimation                |
| **PyTorch**      | LSTM activity recognition            |
| **Flask**        | Web application framework            |
| **Socket.IO**    | Real-time communication              |
| **Bootstrap**    | Frontend UI                          |
| **Chart.js**     | Dashboard visualization              |

---

## 💻 Requirements

### Minimum Requirements

| Component | Minimum                    |
| --------- | -------------------------- |
| CPU       | Intel Core i5 / equivalent |
| RAM       | 8 GB                       |
| GPU       | Not required               |
| Storage   | 2 GB                       |
| Python    | 3.10+                      |
| Camera    | Webcam                     |

### Recommended Requirements

| Component | Recommended                 |
| --------- | --------------------------- |
| CPU       | Intel Core i7 / AMD Ryzen 7 |
| RAM       | 16 GB                       |
| GPU       | NVIDIA GTX 1060 or better   |
| Storage   | 10 GB                       |
| Camera    | HD Webcam                   |

A GPU is optional but can significantly improve real-time performance.

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd child_safety_monitoring
```

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Connect a Webcam

Connect a webcam to your computer and make sure it is accessible by OpenCV.

The YOLO model will be downloaded automatically on its first use if configured accordingly.

---

## ▶️ Running the Application

### 🌐 Web Interface

The recommended way to run the project:

```bash
python run.py --mode web
```

Then open:

```text
http://localhost:5000
```

### 🖥️ Console Monitoring

```bash
python run.py --mode monitor
```

### 🏋️ Train the Activity Model

```bash
python run.py --mode train
```

### 📦 Prepare Training Data

```bash
python run.py --mode prepare
```

### 🎲 Generate Synthetic Data

```bash
python run.py --mode synthetic
```

### 🧪 Run Tests

```bash
python run.py --mode test
```

---

## 🌐 Web Interface

The Flask-based web application provides a centralized monitoring dashboard.

### Main Pages

| Page         | Purpose                             |
| ------------ | ----------------------------------- |
| 🏠 Home      | Main monitoring interface           |
| 📊 Dashboard | Activity and performance statistics |
| 🚨 Alerts    | View detected safety alerts         |
| ℹ️ About     | Project information                 |
| ⚙️ Settings  | Configure monitoring parameters     |

The dashboard can display real-time information such as:

* Current activity
* Detection status
* Confidence score
* FPS
* Number of detected people
* Safety status
* Recent alerts

---

## 🔧 Configuration

The main configuration can be modified in:

```text
config.py
```

### Model Settings

```python
YOLO_MODEL = "yolov8n.pt"

CONFIDENCE_THRESHOLD = 0.5

SEQUENCE_LENGTH = 30
```

### Camera Settings

```python
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
```

### Safety Rules

```python
UNSAFE_ACTIVITIES = [
    "falling",
    "climbing"
]

FALL_DETECTION_THRESHOLD = 0.6
```

### Alert Settings

```python
ALERT_COOLDOWN_SECONDS = 5

MAX_ALERTS_PER_MINUTE = 10
```

---

## 🏋️ Training

The activity recognition model can be trained using activity videos.

### Dataset Structure

Place training videos into the appropriate directories:

```text
data/
└── activities/
    ├── walking/
    ├── running/
    ├── sitting/
    ├── falling/
    └── climbing/
```

### Prepare Training Data

```bash
python data_preparation.py --process
```

This process extracts pose/keypoint information from the videos and prepares the data for model training.

### Generate Synthetic Data

```bash
python data_preparation.py --synthetic 200
```

### Train the Model

```bash
python run.py --mode train
```

Trained models are stored in:

```text
saved_models/
```

---

## 🔔 Alert System

The system supports multiple notification methods.

### Supported Methods

| Method      | Description                        |
| ----------- | ---------------------------------- |
| 💻 Desktop  | Local notification and sound       |
| 📧 Email    | SMTP-based email notifications     |
| 📱 SMS      | SMS through a supported provider   |
| 📨 Telegram | Telegram bot notifications         |
| 🔗 Webhook  | HTTP callback for external systems |

---

## ⚙️ Alert Configuration

Create:

```text
alert_config.json
```

using:

```text
alert_config.example.json
```

as a template.

Example:

```json
{
    "enabled_methods": [
        "desktop",
        "email",
        "telegram"
    ],
    "email": {
        "smtp_server": "smtp.gmail.com",
        "sender_email": "your_email@gmail.com",
        "sender_password": "your_app_password",
        "recipient_emails": [
            "recipient@email.com"
        ]
    },
    "telegram": {
        "bot_token": "your_bot_token",
        "chat_ids": [
            "chat_id"
        ]
    },
    "throttling": {
        "max_alerts_per_minute": 10,
        "cooldown_seconds": 5
    }
}
```

### 🔐 Security Recommendation

**Never commit `alert_config.json` containing passwords, API keys, bot tokens, or other credentials to GitHub.**

Add it to `.gitignore`:

```gitignore
alert_config.json
.env
*.key
*.pem
```

For production deployments, environment variables or a secure secret-management system should be preferred.

---

## 🧪 Testing

### Run the Complete Test Suite

```bash
python run.py --mode test
```

### Run Pytest Directly

```bash
python -m pytest tests/ -v
```

The test suite is located in:

```text
tests/test_system.py
```

---

## 📊 Performance

The following are approximate target/observed performance figures and can vary depending on hardware, camera resolution, model configuration, and workload.

| Operation            |  CPU Only |  With GPU |
| -------------------- | --------: | --------: |
| Detection            | 10–15 FPS | 25–30 FPS |
| Pose Estimation      | 15–20 FPS |   30+ FPS |
| Activity Recognition | 20–25 FPS |   30+ FPS |
| Overall Pipeline     |  8–12 FPS | 20–25 FPS |

Actual performance should be benchmarked on the target deployment hardware.

---

## ⌨️ Keyboard Shortcuts

| Key | Action                           |
| --- | -------------------------------- |
| `Q` | Quit monitoring                  |
| `S` | Save current frame               |
| `R` | Reset system state               |
| `V` | Toggle visualization information |

---

## 🔮 Future Scope

The project can be extended with the following features:

* [ ] 📱 Mobile application for remote monitoring
* [ ] ☁️ Cloud-based recording storage
* [ ] 📹 Multi-camera support
* [ ] 🧑‍🧒 Child-specific identification
* [ ] 🔊 Voice-based safety alerts
* [ ] 🏠 Smart-home integration
* [ ] 📱 WhatsApp/SMS notification integration
* [ ] 📈 Advanced analytics and reporting
* [ ] 🗺️ Configurable danger zones
* [ ] 👥 Improved multi-person tracking
* [ ] 🤖 More advanced activity recognition models
* [ ] 🔒 Privacy-preserving local processing

---

## 👥 Team Members

| Name                    | Roll Number    | Role        |
| ----------------------- | -------------- | ----------- |
| **Aashutosh Vaish**     | `303302223004` | Team Member |
| **Arpit Ojha**          | `303302223049` | Team Member |
| **Shivansh Mishra**     | `303302223199` | Team Member |
| **Shashwat Khandelwal** | `303302223197` | Team Leader |

---

## 👨‍🏫 Project Guide

**Mrs. Poonam Gupta**

Assistant Professor
Department of Computer Science & Engineering
**SSIPMT, Raipur**

---

## 📚 Project Details

| Attribute   | Details                                              |
| ----------- | ---------------------------------------------------- |
| Project     | Computer Vision-Based Child Safety Monitoring System |
| Semester    | 7th Semester                                         |
| Batch       | 2023–27                                              |
| Session     | July–December 2026                                   |
| Institution | SSIPMT, Raipur                                       |
| Department  | Computer Science & Engineering                       |

---

## 🙏 Acknowledgments

We would like to thank the open-source communities and technologies that made this project possible:

* **Ultralytics YOLO** — Object detection
* **Google MediaPipe** — Pose estimation
* **PyTorch** — Deep learning and LSTM implementation
* **OpenCV** — Computer vision and video processing
* **Flask** — Web application framework
* **Bootstrap** — Frontend development
* **Chart.js** — Data visualization

---

## 📖 References

* Ultralytics YOLO documentation
* MediaPipe Pose documentation
* PyTorch LSTM documentation
* Flask documentation
* OpenCV documentation

---

## 🔒 Privacy & Safety Considerations

Because this system processes video of children, privacy should be treated as a core design requirement.

Recommended practices include:

* Process video locally whenever possible.
* Avoid unnecessary cloud uploads.
* Protect stored recordings and captured images.
* Do not expose monitoring endpoints publicly without authentication.
* Keep notification credentials outside source control.
* Delete recordings and alert images according to an appropriate retention policy.
* Obtain appropriate consent before monitoring or storing children's video.

The system is intended as an **assistive monitoring tool**, not as a replacement for responsible adult supervision.

---

## 📄 License

This project is developed as part of the **7th Semester Project at SSIPMT, Raipur**.



---

## ⭐ Project

**Computer Vision-Based Child Safety Monitoring System**

> Detect. Analyze. Alert. Protect.

<div align="center">

**SSIPMT, Raipur | Department of Computer Science & Engineering**

**7th Semester | Batch 2023–27 | Session July–December 2026**

</div>
