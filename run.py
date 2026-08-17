#!/usr/bin/env python
# run.py - Launcher for Child Safety Monitoring System

import os
import sys
import subprocess
import webbrowser
import time
import argparse
import platform

def print_banner():
    """Print application banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   👶 CHILD SAFETY MONITORING SYSTEM                         ║
    ║   Computer Vision-Based Real-Time Activity & Posture        ║
    ║   Recognition System                                        ║
    ║                                                               ║
    ║   Version: 1.0.0                                            ║
    ║   Semester: 7th | Batch: 2023-27                          ║
    ║                                                               ║
    ║   📍 SSIPMT, Raipur                                         ║
    ║   👨‍🏫 Guide: Mrs. Poonam Gupta                              ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_environment():
    """Check if environment is set up correctly"""
    print("\n🔍 Checking environment...")
    
    # Check Python version
    python_version = platform.python_version()
    print(f"   Python: {python_version}")
    if python_version < '3.10':
        print("   ⚠️  Python 3.10+ recommended")
    
    # Check requirements
    try:
        import cv2
        print("   ✅ OpenCV")
    except ImportError:
        print("   ❌ OpenCV not installed")
        return False
    
    try:
        import torch
        cuda = " (CUDA available)" if torch.cuda.is_available() else ""
        print(f"   ✅ PyTorch{cuda}")
    except ImportError:
        print("   ❌ PyTorch not installed")
        return False
    
    try:
        import mediapipe
        print("   ✅ MediaPipe")
    except ImportError:
        print("   ❌ MediaPipe not installed")
        return False
    
    try:
        import ultralytics
        print("   ✅ Ultralytics YOLO")
    except ImportError:
        print("   ❌ Ultralytics not installed")
        return False
    
    # Check directories
    dirs = ['saved_models', 'data/activities', 'captures', 'alerts', 'static/uploads']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("   ✅ Directories created")
    
    print("   ✅ Environment check passed")
    return True

def run_command(cmd, cwd=None):
    """Run a command and display output"""
    try:
        process = subprocess.Popen(
            cmd,
            shell=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        return process.returncode == 0
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def run_monitoring():
    """Run the monitoring system"""
    print("\n🔍 Starting monitoring system...")
    print("   Press 'q' to quit")
    print("   Press 's' to save frame")
    print("   Press 'r' to reset")
    print()
    
    run_command("python app.py")

def run_flask():
    """Run the Flask web application"""
    print("\n🌐 Starting web interface...")
    print("   Opening browser at http://localhost:5000")
    print("   Press Ctrl+C to stop")
    print()
    
    # Open browser after delay
    def open_browser():
        time.sleep(2)
        webbrowser.open("http://localhost:5000")
    
    import threading
    threading.Thread(target=open_browser, daemon=True).start()
    
    run_command("python flask_app.py")

def run_training():
    """Run training"""
    print("\n🏋️ Training activity recognition model...")
    
    # Check if data exists
    if not os.path.exists('data/training_data.npz'):
        print("   ⚠️  Training data not found")
        print("   Please run data preparation first:")
        print("   python data_preparation.py --create-dirs")
        print("   python data_preparation.py --process")
        return
    
    # Use a Python script file instead of inline command to avoid quoting issues
    train_script = """
import sys
sys.path.append('.')
from app import ChildSafetyMonitor
m = ChildSafetyMonitor()
m.train_from_data()
"""
    # Write to a temporary file
    with open('temp_train.py', 'w') as f:
        f.write(train_script)
    
    run_command("python temp_train.py")
    
    # Clean up
    if os.path.exists('temp_train.py'):
        os.remove('temp_train.py')

def run_data_prep():
    """Run data preparation"""
    print("\n📁 Preparing training data...")
    
    # Check if directories exist
    activities = ['walking', 'running', 'sitting', 'falling', 'climbing']
    has_videos = False
    
    for activity in activities:
        activity_dir = f'data/activities/{activity}'
        if os.path.exists(activity_dir):
            videos = [f for f in os.listdir(activity_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
            if videos:
                has_videos = True
                print(f"   Found {len(videos)} videos in {activity}")
    
    if not has_videos:
        print("   ⚠️  No videos found in data/activities/")
        print("   Creating directory structure...")
        run_command("python data_preparation.py --create-dirs")
        print("\n   Please add videos to the created directories")
        print("   Then run: python data_preparation.py --process")
        return
    
    run_command("python data_preparation.py --process")

def run_tests():
    """Run tests"""
    print("\n🧪 Running system tests...")
    run_command("python -m pytest tests/ -v --tb=short")

def run_synthetic():
    """Generate synthetic data"""
    print("\n📊 Generating synthetic training data...")
    run_command("python data_preparation.py --synthetic 200")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Child Safety Monitoring System Launcher"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["monitor", "web", "train", "prepare", "synthetic", "test"],
        default="web",
        help="Run mode: monitor, web, train, prepare, synthetic, test"
    )
    parser.add_argument(
        "--no-banner", "-nb",
        action="store_true",
        help="Suppress banner display"
    )
    parser.add_argument(
        "--check", "-c",
        action="store_true",
        help="Check environment only"
    )
    
    args = parser.parse_args()
    
    if not args.no_banner:
        print_banner()
    
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed")
        print("Please install dependencies: pip install -r requirements.txt")
        sys.exit(1)
    
    if args.check:
        print("\n✅ Environment is ready")
        return
    
    # Run selected mode
    mode_handlers = {
        "monitor": run_monitoring,
        "web": run_flask,
        "train": run_training,
        "prepare": run_data_prep,
        "synthetic": run_synthetic,
        "test": run_tests
    }
    
    handler = mode_handlers.get(args.mode)
    if handler:
        handler()
    else:
        print("❌ Invalid mode. Use --help for options")

if __name__ == "__main__":
    main()
