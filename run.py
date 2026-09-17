#!/usr/bin/env python
# run.py - Launcher for Child Safety Monitoring System

import os
import sys
import subprocess
import webbrowser
import shlex
import argparse
import platform
import tempfile


def print_banner():
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
    print("\n🔍 Checking environment...")

    python_version = platform.python_version()
    print(f"   Python: {python_version}")
    if python_version < '3.10':
        print("   ⚠️  Python 3.10+ recommended")

    try:
        import cv2  # noqa: F401
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
        import mediapipe  # noqa: F401
        print("   ✅ MediaPipe")
    except ImportError:
        print("   ❌ MediaPipe not installed")
        return False

    try:
        import ultralytics  # noqa: F401
        print("   ✅ Ultralytics YOLO")
    except ImportError:
        print("   ❌ Ultralytics not installed")
        return False

    dirs = ['saved_models', 'data/activities', 'captures', 'alerts', 'static/uploads']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("   ✅ Directories created")

    print("   ✅ Environment check passed")
    return True


def run_command(cmd, cwd=None):
    """Run a command and display output. cmd may be str or list."""
    try:
        args = shlex.split(cmd) if isinstance(cmd, str) else cmd
        process = subprocess.Popen(
            args,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
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
    print("\n🔍 Starting monitoring system...")
    print("   Press 'q' to quit")
    print("   Press 's' to save frame")
    print("   Press 'r' to reset")
    print()
    run_command([sys.executable, "app.py"])


def run_flask():
    print("\n🌐 Starting web interface...")
    print("   Browser will open automatically on the correct port")
    print("   Press Ctrl+C to stop")
    print()
    run_command([sys.executable, "flask_app.py"])


def run_training():
    print("\n🏋️ Training activity recognition model...")

    if not os.path.exists('data/training_data.npz'):
        print("   ⚠️  Training data not found")
        print("   Please run data preparation first:")
        print("   python data_preparation.py --create-dirs")
        print("   python data_preparation.py --process")
        return

    train_script = """
import sys
sys.path.append('.')
from app import ChildSafetyMonitor
m = ChildSafetyMonitor()
m.train_from_data()
"""

    fd, tmp_path = tempfile.mkstemp(suffix='.py', prefix='train_')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(train_script)
        run_command([sys.executable, tmp_path])
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def run_data_prep():
    print("\n📁 Preparing training data...")

    activities = ['walking', 'running', 'sitting', 'falling', 'climbing']
    has_videos = False

    for activity in activities:
        activity_dir = f'data/activities/{activity}'
        if os.path.exists(activity_dir):
            videos = [f for f in os.listdir(activity_dir)
                      if f.endswith(('.mp4', '.avi', '.mov'))]
            if videos:
                has_videos = True
                print(f"   Found {len(videos)} videos in {activity}")

    if not has_videos:
        print("   ⚠️  No videos found in data/activities/")
        print("   Creating directory structure...")
        run_command([sys.executable, "data_preparation.py", "--create-dirs"])
        print("\n   Please add videos to the created directories")
        print("   Then run: python data_preparation.py --process")
        return

    run_command([sys.executable, "data_preparation.py", "--process"])


def run_tests():
    print("\n🧪 Running system tests...")
    run_command([sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"])


def run_synthetic():
    print("\n📊 Generating synthetic training data...")
    run_command([sys.executable, "data_preparation.py", "--synthetic", "200"])


def main():
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

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if not check_environment():
        print("\n❌ Environment check failed")
        print("Please install dependencies: pip install -r requirements.txt")
        sys.exit(1)

    if args.check:
        print("\n✅ Environment is ready")
        return

    mode_handlers = {
        "monitor": run_monitoring,
        "web": run_flask,
        "train": run_training,
        "prepare": run_data_prep,
        "synthetic": run_synthetic,
        "test": run_tests,
    }

    handler = mode_handlers.get(args.mode)
    if handler:
        handler()
    else:
        print("❌ Invalid mode. Use --help for options")


if __name__ == "__main__":
    main()
