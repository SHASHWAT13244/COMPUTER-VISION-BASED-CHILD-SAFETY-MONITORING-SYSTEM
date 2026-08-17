# 👶 Computer Vision-Based Child Safety Monitoring System

> A real-time AI-powered child safety monitoring system using computer vision and deep learning to detect activities, identify potentially unsafe behavior, and generate alerts.

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
* [Model Evaluation](#-model-evaluation)
* [Alert System](#-alert-system)
* [Performance](#-performance)
* [Keyboard Shortcuts](#-keyboard-shortcuts)
* [Team Members](#-team-members)
* [Project Guide](#-project-guide)
* [Future Scope](#-future-scope)
* [Acknowledgments](#-acknowledgments)
* [References](#-references)
* [Privacy & Safety](#-privacy--safety)
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
* 👥 Multi-person tracking

The primary objective is to identify potentially unsafe activities such as **falling** and **climbing** and notify the responsible person as quickly as possible.

> **Note:** This system is intended as an assistive monitoring tool and is not a replacement for responsible adult supervision.

---

## ✨ Key Features

| Feature                       | Description                                                 |
| ----------------------------- | ----------------------------------------------------------- |
| 🎥 **Real-Time Monitoring**   | Processes live webcam footage and displays annotated frames |
| 👤 **Person Detection**       | YOLOv8-based detection of people in the scene               |
| 🦴 **Pose Estimation**        | MediaPipe extracts 33 body landmarks                        |
| 🏃 **Activity Recognition**   | LSTM classifies sequences of body movements                 |
| ⚠️ **Safety Engine**          | Applies rules to identify potentially unsafe behavior       |
| 🔔 **Alert System**           | Supports desktop, email, Telegram, SMS, and webhook alerts  |
| 📊 **Dashboard**              | Web-based interface for monitoring activities and alerts    |
| 📸 **Image Analysis**         | Upload and analyze individual images                        |
| 💾 **Recording**              | Saves monitoring videos and important frames                |
| 👥 **Multi-Person Tracking**  | Tracks multiple people within the camera view               |
| 🚦 **Alert Throttling**       | Prevents excessive repeated notifications                   |
| 📈 **Performance Evaluation** | Provides Accuracy, Precision, Recall, and F1-Score metrics  |

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
│  OpenCV Processing  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│       YOLOv8        │
│   Person Detection  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    MediaPipe Pose   │
│   33 Body Keypoints │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    Frame Sequence   │
│     Buffer (30)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   LSTM Activity     │
│     Recognition     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    Safety Engine    │
│ Rules + Pose Check  │
└──────────┬──────────┘
           │
      ┌────┴────┐
      ▼         ▼
┌───────────┐ ┌────────────────┐
│   Safe /  │ │  Unsafe Event  │
│  Caution  │ └───────┬────────┘
└───────────┘         │
                      ▼
              ┌────────────────┐
              │  Alert System  │
              └───────┬────────┘
                      │
       ┌──────────────┼──────────────┐
       ▼              ▼              ▼
   Desktop          Email        Telegram
       │              │              │
       └──────────────┼──────────────┘
                      │
                 ┌────┴────┐
                 ▼         ▼
                SMS     Webhook
```

---

## 🧠 How It Works

### 1. 📸 Frame Acquisition

The webcam continuously captures video frames using OpenCV. The frames are resized and preprocessed before being passed to the detection pipeline.

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

### 4. 🧠 Activity Recognition

A sequence of pose information is collected over multiple frames. By default, the sequence contains **30 frames**.

The LSTM model analyzes the temporal movement pattern and classifies the activity.

Supported activities include:

```text
Walking
Running
Sitting
Falling
Climbing
```

### 5. ⚠️ Safety Analysis

The Safety Engine combines activity predictions with additional rules, including:

* Fall detection
* Unsafe climbing detection
* Zone violations
* Activity confidence checks
* Alert cooldowns

### 6. 🔔 Alert Generation

When an unsafe event is detected, the system generates an alert.

Depending on configuration, alerts can be delivered through:

* 💻 Desktop notification
* 📧 Email
* 📱 SMS
* 📨 Telegram
* 🔗 Webhook

---

## 📁 Project Structure

```text
child_safety_monitoring/
│
├── app.py                    # Main console application
├── flask_app.py              # Flask web application
├── config.py                 # Configuration settings
├── data_preparation.py       # Data preparation script
├── data_augmentation.py      # Data augmentation utilities
├── evaluate_model.py         # Model performance evaluation
├── run_evaluation.py         # Quick launcher for evaluation
├── view_training_data.py     # View training data utility
├── requirements.txt          # Python dependencies
├── run.py                    # Launcher script
├── alert_config.json         # Alert configuration
├── README.md                 # Project documentation
│
├── models/
│   ├── detector.py           # YOLOv8 person detection
│   ├── pose_estimator.py     # MediaPipe pose estimation
│   ├── activity_recognizer.py# LSTM activity recognition
│   ├── safety_engine.py      # Safety rule engine
│   └── tracker.py            # Multi-person tracking
│
├── utils/
│   ├── alert.py              # Basic alert system
│   ├── alert_advanced.py     # Advanced email/SMS alerts
│   ├── visualization.py      # Visualization utilities
│   └── performance_monitor.py# Performance monitoring
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       ├── script.js
│       ├── dashboard.js
│       └── alerts.js
│
├── templates/
│   ├── index.html
│   ├── dashboard.html
│   ├── alerts.html
│   ├── about.html
│   └── settings.html
│
├── saved_models/
│   └── activity_model.pth
│
├── data/
│   └── activities/
│       ├── walking/
│       ├── running/
│       ├── sitting/
│       ├── falling/
│       └── climbing/
│
├── captures/                 # Captured frames
├── alerts/                   # Alert images
├── recordings/               # Recorded videos
├── tests/                    # System tests
│
├── evaluation_results.json   # Model evaluation metrics
└── evaluation_plots.png      # Performance visualization
```

---

## 🛠️ Technologies Used

| Technology       | Purpose                               |
| ---------------- | ------------------------------------- |
| **Python 3.10+** | Core programming language             |
| **OpenCV**       | Video processing and computer vision  |
| **YOLOv8**       | Person detection                      |
| **MediaPipe**    | Human pose estimation                 |
| **PyTorch**      | Deep learning and LSTM implementation |
| **Flask**        | Web application framework             |
| **Socket.IO**    | Real-time communication               |
| **Bootstrap**    | Frontend UI                           |
| **Chart.js**     | Dashboard visualization               |

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

---

## ▶️ Running the Application

### 🌐 Web Interface

The web interface is the recommended way to use the system.

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

### 📊 Evaluate Model Performance

```bash
python run_evaluation.py
```

### 🧪 Run Tests

```bash
python run.py --mode test
```

### 📁 View Training Data

```bash
python view_training_data.py
```

---

## 🌐 Web Interface

The Flask-based web application provides a centralized monitoring dashboard.

### Main Pages

| Page             | Purpose                             |
| ---------------- | ----------------------------------- |
| 🏠 **Home**      | Main monitoring interface           |
| 📊 **Dashboard** | Activity and performance statistics |
| 🚨 **Alerts**    | View detected safety alerts         |
| ℹ️ **About**     | Project information                 |
| ⚙️ **Settings**  | Configure monitoring parameters     |

---

## 🔧 Configuration

The main configuration can be modified in `config.py`.

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
UNSAFE_ACTIVITIES = ["falling", "climbing"]
FALL_DETECTION_THRESHOLD = 0.6
```

### Alert Settings

```python
ALERT_COOLDOWN_SECONDS = 5
MAX_ALERTS_PER_MINUTE = 10
```

---

## 🏋️ Training

### Dataset Structure

Place training data into the appropriate activity directories:

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

## 📊 Model Evaluation

### Run Evaluation

```bash
python run_evaluation.py
```

### Evaluation Metrics

Based on the current evaluation documentation:

| Metric                |  Value |
| --------------------- | -----: |
| **Accuracy**          | 92.86% |
| **Precision (Macro)** | 95.00% |
| **Recall (Macro)**    | 93.75% |
| **F1-Score (Macro)**  | 93.65% |

### Per-Class Performance

| Activity | Precision | Recall | F1-Score | Support |
| -------- | --------: | -----: | -------: | ------: |
| Walking  |    1.0000 | 1.0000 |   1.0000 |       2 |
| Running  |    1.0000 | 0.7500 |   0.8571 |       4 |
| Sitting  |    1.0000 | 1.0000 |   1.0000 |       4 |
| Climbing |    0.8000 | 1.0000 |   0.8889 |       4 |

> **Evaluation note:** The reported evaluation currently lists 14 test samples, while the project defines five activities including `falling`. Before using these results in a final presentation/report, verify that the evaluation dataset and per-class metrics include all intended classes.

### Generated Files

The evaluation process generates:

```text
evaluation_results.json
evaluation_plots.png
```

* `evaluation_results.json` — Stores evaluation metrics in JSON format.
* `evaluation_plots.png` — Performance visualization dashboard.

---

## 🔔 Alert System

The system supports multiple notification methods.

| Method          | Description                        |
| --------------- | ---------------------------------- |
| 💻 **Desktop**  | Local notification and sound       |
| 📧 **Email**    | SMTP-based email notifications     |
| 📱 **SMS**      | SMS through a supported provider   |
| 📨 **Telegram** | Telegram bot notifications         |
| 🔗 **Webhook**  | HTTP callback for external systems |

### Alert Configuration

Create `alert_config.json` using `alert_config.example.json` as a template.

> ⚠️ **Security Note:** Never commit `alert_config.json` containing passwords, API keys, or bot tokens to GitHub.

Add sensitive configuration files to `.gitignore`:

```gitignore
alert_config.json
.env
*.key
*.pem
```

---

## 📊 Performance

The following performance figures are the current project targets/measurements documented for the system.

| Operation            |     CPU Only |      With GPU |
| -------------------- | -----------: | ------------: |
| Detection            |    10–15 FPS |     25–30 FPS |
| Pose Estimation      |    15–20 FPS |       30+ FPS |
| Activity Recognition |    20–25 FPS |       30+ FPS |
| **Overall Pipeline** | **8–12 FPS** | **20–25 FPS** |

Actual performance depends on:

* CPU/GPU model
* Camera resolution
* Input frame rate
* Number of people detected
* Model configuration
* Background processing
* Operating system

---

## ⌨️ Keyboard Shortcuts

| Key | Action                           |
| --- | -------------------------------- |
| `Q` | Quit monitoring                  |
| `S` | Save current frame               |
| `R` | Reset system state               |
| `V` | Toggle visualization information |

---

## 👥 Team Members

| Name                    | Roll Number    | Role        |
| ----------------------- | -------------- | ----------- |
| **Shashwat Khandelwal** | `303302223197` | Team Leader |
| **Shivansh Mishra**     | `303302223199` | Team Member |
| **Arpit Ojha**          | `303302223004` | Team Member |
| **Aashutosh Vaish**     | `303302223049` | Team Member |

---

## 👨‍🏫 Project Guide

**Mrs. Poonam Gupta**

Assistant Professor
Department of Computer Science & Engineering
**SSIPMT, Raipur**

---

## 📚 Project Details

| Attribute       | Details                                              |
| --------------- | ---------------------------------------------------- |
| **Project**     | Computer Vision-Based Child Safety Monitoring System |
| **Semester**    | 7th Semester                                         |
| **Batch**       | 2023–27                                              |
| **Session**     | July–December 2026                                   |
| **Institution** | SSIPMT, Raipur                                       |
| **Department**  | Computer Science & Engineering                       |

---

## 🔮 Future Scope

* 📱 Mobile application for remote monitoring
* ☁️ Cloud-based recording storage
* 📹 Multi-camera support
* 🧑‍🧒 Child-specific identification
* 🔊 Voice-based safety alerts
* 🏠 Smart-home integration
* 📱 WhatsApp/SMS notification integration
* 📈 Advanced analytics and reporting
* 🗺️ Configurable danger zones
* 👥 Improved multi-person tracking
* 🤖 More advanced activity recognition models
* 🔒 Privacy-preserving local processing

---

## 🙏 Acknowledgments

We would like to acknowledge the following technologies and projects that support the development of this system:

* **Ultralytics YOLO** — Object detection
* **Google MediaPipe** — Pose estimation
* **PyTorch** — Deep learning and LSTM implementation
* **OpenCV** — Computer vision and video processing
* **Flask** — Web application framework
* **Bootstrap** — Frontend development
* **Chart.js** — Data visualization

---

## 📖 References

1. Jocher, G., Chaurasia, A., & Qiu, J. (2023). **Ultralytics YOLOv8**.
   [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

2. Lugaresi, C., et al. (2019). **MediaPipe: A Framework for Building Perception Pipelines**. arXiv:1906.08172.

3. Hochreiter, S., & Schmidhuber, J. (1997). **Long Short-Term Memory**. *Neural Computation, 9(8)*, 1735–1780.

4. Tapia, E. M., et al. (2004). **Activity Recognition in the Home**. In *Pervasive Computing*.

5. Lin, T. Y., et al. (2014). **Microsoft COCO: Common Objects in Context**. In *ECCV*.

---

## 🔒 Privacy & Safety Considerations

Because this system processes video involving children, privacy and security should be treated as important design requirements.

* 🔐 Process video locally whenever possible.
* ☁️ Avoid unnecessary cloud uploads.
* 🔒 Protect stored recordings and captured images.
* 🌐 Do not expose monitoring endpoints publicly without authentication.
* 🔑 Keep notification credentials outside source control.
* 🗑️ Delete recordings and alert images according to an appropriate retention policy.
* 👨‍👩‍👧 Obtain appropriate consent before monitoring or storing children's video.

> **Important:** The system is intended as an **assistive monitoring tool**, not as a replacement for responsible adult supervision.

---

## 📄 License

This project is developed as part of the **7th Semester Project at SSIPMT, Raipur**.

---

<div align="center">

**SSIPMT, Raipur | Department of Computer Science & Engineering**

**7th Semester | Batch 2023–27 | Session July–December 2026**

</div>

