"""
Setup script for Child Safety Monitoring System
"""

from setuptools import setup, find_packages

setup(
    name="child-safety-monitoring",
    version="1.0.0",
    description="Computer Vision-Based Child Safety Monitoring System",
    author="Shashwat Khandelwal, Shivansh Mishra",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "opencv-python==4.8.1.78",
        "ultralytics==8.0.200",
        "mediapipe==0.10.7",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "Flask>=2.3.0",
        "flask-socketio>=5.3.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.65.0",
        "requests>=2.31.0",
        "Pillow>=10.0.0",
        "python-socketio>=5.9.0",
        "eventlet>=0.33.0",
    ],
    entry_points={
        "console_scripts": [
            "child-monitor=run_monitoring:main",
            "train-activity-model=train_model:main",
        ],
    },
    python_requires=">=3.8",
)