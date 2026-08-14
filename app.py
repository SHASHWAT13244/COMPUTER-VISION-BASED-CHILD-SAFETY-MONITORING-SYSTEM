# app.py
"""
Main Application for Child Safety Monitoring System
Runs the real-time monitoring pipeline with webcam input
"""

import cv2
import time
import numpy as np
from datetime import datetime
import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from models.detector import ChildDetector
from models.pose_estimator import PoseEstimator
from models.activity_recognizer import ActivityRecognizer
from models.safety_engine import SafetyEngine
from models.tracker import PersonTracker
from utils.alert import AlertSystem
from utils.visualization import Visualizer
from utils.performance_monitor import PerformanceMonitor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChildSafetyMonitor:
    """
    Main application class for child safety monitoring
    """
    
    def __init__(self, config=None):
        """
        Initialize the child safety monitoring system
        
        Args:
            config: Configuration object (uses Config if None)
        """
        self.config = config or Config()
        
        # Initialize components
        logger.info("Initializing Child Safety Monitoring System...")
        
        # Detection
        self.detector = ChildDetector(
            model_path=self.config.YOLO_MODEL,
            conf_threshold=self.config.CONFIDENCE_THRESHOLD,
            iou_threshold=self.config.IOU_THRESHOLD
        )
        
        # Pose estimation
        self.pose_estimator = PoseEstimator(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        # Activity recognition
        self.activity_recognizer = ActivityRecognizer(
            sequence_length=self.config.SEQUENCE_LENGTH,
            num_keypoints=33,
            num_classes=len(self.config.ACTIVITY_CLASSES),
            hidden_size=128,
            num_layers=2,
            dropout=0.2
        )
        
        # Safety engine
        self.safety_engine = SafetyEngine()
        
        # Tracker
        self.tracker = PersonTracker(
            max_lost_frames=10,
            min_confidence=0.5
        )
        
        # Alert system
        self.alert_system = AlertSystem(
            sound_enabled=self.config.ALERT_SOUND,
            display_enabled=True,
            log_enabled=self.config.ALERT_LOG
        )
        
        # Visualizer
        self.visualizer = Visualizer(
            show_fps=True,
            show_info=True
        )
        
        # Performance monitor
        self.perf_monitor = PerformanceMonitor()
        self.perf_monitor.start_monitoring()
        
        # State variables
        self.running = False
        self.cap = None
        self.keypoint_buffer = []
        self.alerts = []
        self.current_activity = None
        self.current_confidence = 0.0
        self.is_safe = True
        self.frame_count = 0
        self.start_time = None
        self.fps = 0
        
        # Load pre-trained model if exists
        self._load_pretrained_model()
        
        logger.info("System initialized successfully!")
        logger.info(f"Activity classes: {self.config.ACTIVITY_CLASSES}")
        logger.info(f"Unsafe activities: {self.config.UNSAFE_ACTIVITIES}")
    
    def _load_pretrained_model(self):
        """Load pre-trained activity recognition model if available"""
        model_paths = [
            os.path.join(self.config.MODELS_DIR, 'activity_model.pth'),
            os.path.join(self.config.MODELS_DIR, 'best_activity_model.pth'),
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                try:
                    self.activity_recognizer.load_model(path)
                    logger.info(f"Loaded pre-trained model from {path}")
                    return
                except Exception as e:
                    logger.error(f"Error loading model from {path}: {e}")
        
        logger.info("No pre-trained model found. Using random weights.")
        logger.info("You can train the model using training data.")
    
    def start(self, camera_id=0, save_video=False, output_path=None):
        """
        Start the monitoring system
        
        Args:
            camera_id: Camera device ID
            save_video: Whether to save the output video
            output_path: Path to save the output video
        """
        logger.info(f"Starting camera (ID: {camera_id})...")
        
        # Open camera
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            logger.error("Could not open camera.")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.FPS)
        
        # Video writer for saving
        self.video_writer = None
        if save_video:
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f"recordings/monitoring_{timestamp}.mp4"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                output_path, fourcc, self.config.FPS,
                (self.config.FRAME_WIDTH, self.config.FRAME_HEIGHT)
            )
            logger.info(f"Saving video to: {output_path}")
        
        self.running = True
        self.start_time = time.time()
        logger.info("Monitoring started! Press 'q' to quit.")
        
        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("\nInterrupted by user")
        finally:
            self.cleanup()
        
        return True
    
    def _main_loop(self):
        """Main processing loop"""
        while self.running:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to grab frame.")
                break
            
            self.frame_count += 1
            frame_start_time = time.time()
            
            # Process frame
            processed_frame, result = self._process_frame(frame)
            
            # Save video if enabled
            if self.video_writer is not None:
                self.video_writer.write(processed_frame)
            
            # Update FPS
            elapsed = time.time() - frame_start_time
            if elapsed > 0:
                current_fps = 1.0 / elapsed
                self.fps = 0.9 * self.fps + 0.1 * current_fps if self.fps > 0 else current_fps
                self.perf_monitor.add_metric('fps', self.fps)
                self.perf_monitor.add_metric('inference_time', elapsed * 1000)
            
            # Display
            cv2.imshow('Child Safety Monitoring', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
                break
            elif key == ord('s'):
                self._save_frame(processed_frame)
            elif key == ord('r'):
                self._reset_state()
            elif key == ord('v'):
                self._toggle_visualization()
    
    def _process_frame(self, frame):
        """
        Process a single frame
        
        Args:
            frame: Input frame
            
        Returns:
            processed_frame: Frame with annotations
            result: Processing results
        """
        result = {
            'activity': 'None',
            'confidence': 0.0,
            'safe': True,
            'message': 'All safe',
            'severity': 'low',
            'alert': None,
            'detections': [],
            'tracks': {}
        }
        
        # Detect children
        detections = self.detector.detect(frame)
        result['detections'] = detections
        
        # Update tracker
        tracks = self.tracker.update(detections, frame)
        result['tracks'] = tracks
        
        # Process frame
        annotated_frame = frame.copy()
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Process each tracked person
        for track_id, track in tracks.items():
            bbox = track['bbox']
            x1, y1, x2, y2 = bbox
            
            # Extract person region
            person_frame = frame[y1:y2, x1:x2]
            if person_frame.size == 0:
                continue
            
            # Extract pose
            keypoints = self.pose_estimator.extract_keypoints(frame)
            
            if keypoints is not None:
                # Draw pose
                annotated_frame = self.pose_estimator.draw_pose(annotated_frame)
                
                # Add to buffer
                keypoints_flat = keypoints[:, :3].flatten()
                self.keypoint_buffer.append(keypoints_flat)
                
                # Keep buffer size manageable
                if len(self.keypoint_buffer) > self.config.SEQUENCE_LENGTH * 2:
                    self.keypoint_buffer = self.keypoint_buffer[-self.config.SEQUENCE_LENGTH:]
                
                # Predict activity
                if len(self.keypoint_buffer) >= self.config.SEQUENCE_LENGTH:
                    activity, confidence = self.activity_recognizer.predict_activity(
                        self.keypoint_buffer
                    )
                    self.current_activity = activity
                    self.current_confidence = confidence
                    result['activity'] = activity or 'Unknown'
                    result['confidence'] = confidence
                else:
                    result['activity'] = 'Collecting data...'
                    result['confidence'] = 0.0
            else:
                # No pose detected, maintain previous state
                result['activity'] = self.current_activity or 'Unknown'
                result['confidence'] = self.current_confidence
                
                # Add zeros to maintain sequence
                if len(self.keypoint_buffer) > 0:
                    self.keypoint_buffer.append(np.zeros(33 * 3))
                    if len(self.keypoint_buffer) > self.config.SEQUENCE_LENGTH * 2:
                        self.keypoint_buffer = self.keypoint_buffer[-self.config.SEQUENCE_LENGTH:]
        
        # Check safety
        if result['activity'] and result['activity'] not in ['None', 'Collecting data...']:
            safety_result = self.safety_engine.check_safety(
                activity=result['activity'],
                confidence=result['confidence'],
                pose_keypoints=keypoints if 'keypoints' in locals() else None,
                bbox=bbox if 'bbox' in locals() else None,
                frame_time=datetime.now()
            )
            
            result['safe'] = safety_result['safe']
            result['message'] = safety_result.get('message', 'All safe')
            result['severity'] = safety_result.get('severity', 'low')
            result['alert'] = safety_result.get('alert')
            
            # Generate alert if unsafe
            if not result['safe'] and safety_result.get('alert_generated', False):
                alert_info = self.alert_system.generate_alert(
                    message=result['message'],
                    severity=result['severity'],
                    activity=result['activity'],
                    bbox=bbox if 'bbox' in locals() else None
                )
                self.alerts.append(alert_info)
                result['alert_info'] = alert_info
        
        # Draw status
        annotated_frame = self.visualizer.draw_status(
            annotated_frame,
            result['activity'],
            result['confidence'],
            result['safe'],
            self.alerts[-5:] if self.alerts else []
        )
        
        # Draw tracking
        annotated_frame = self.tracker.draw_tracks(annotated_frame)
        
        # Update FPS display
        self.visualizer.fps = self.fps
        
        # Add performance overlay
        if self.frame_count % 30 == 0:
            self.perf_monitor.add_metric('frame_processed', 1)
        
        return annotated_frame, result
    
    def _save_frame(self, frame):
        """Save current frame"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs('captures', exist_ok=True)
        filename = f"captures/capture_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        logger.info(f"Frame saved as {filename}")
    
    def _reset_state(self):
        """Reset the system state"""
        self.keypoint_buffer = []
        self.alerts = []
        self.current_activity = None
        self.current_confidence = 0.0
        self.is_safe = True
        self.safety_engine.reset()
        self.tracker.reset()
        self.activity_recognizer.reset_buffer()
        logger.info("System state reset")
    
    def _toggle_visualization(self):
        """Toggle visualization mode"""
        self.visualizer.show_info = not self.visualizer.show_info
        logger.info(f"Visualization info: {self.visualizer.show_info}")
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up...")
        
        self.running = False
        
        if self.cap:
            self.cap.release()
        
        if self.video_writer:
            self.video_writer.release()
        
        cv2.destroyAllWindows()
        
        self.perf_monitor.stop_monitoring()
        
        logger.info("System stopped.")
        
        # Print summary
        stats = self.alert_system.get_alert_statistics()
        logger.info("\n=== Session Summary ===")
        logger.info(f"Total frames processed: {self.frame_count}")
        logger.info(f"Average FPS: {self.fps:.1f}")
        logger.info(f"Total alerts generated: {stats['total']}")
        if stats['by_severity']:
            logger.info(f"Alerts by severity: {stats['by_severity']}")
        if stats['by_activity']:
            logger.info(f"Alerts by activity: {stats['by_activity']}")
        
        # Performance summary
        summary = self.perf_monitor.get_summary()
        if summary:
            logger.info("\n=== Performance Summary ===")
            if 'cpu_usage' in summary:
                logger.info(f"CPU Usage: {summary['cpu_usage']['average']:.1f}%")
            if 'memory_usage' in summary:
                logger.info(f"Memory Usage: {summary['memory_usage']['average']:.1f}%")
    
    def train_from_data(self, data_path=None, save_model=True):
        """
        Train the activity recognition model from data
        
        Args:
            data_path: Path to training data (.npz file)
            save_model: Whether to save the trained model
            
        Returns:
            training_history: Dictionary with training history
        """
        if data_path is None:
            data_path = os.path.join(self.config.DATA_DIR, 'training_data.npz')
        
        if not os.path.exists(data_path):
            logger.error(f"Training data not found at {data_path}")
            logger.info("Please prepare training data first using data_preparation.py")
            return None
        
        logger.info(f"Loading training data from {data_path}")
        data = np.load(data_path)
        X = data['X']
        y = data['y']
        
        logger.info(f"Data shape: {X.shape}, labels: {y.shape}")
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Validation set: {X_val.shape[0]} samples")
        
        # Train model
        save_path = os.path.join(self.config.MODELS_DIR, 'activity_model.pth') if save_model else None
        
        history = self.activity_recognizer.train_model(
            X_train, y_train, X_val, y_val,
            epochs=self.config.EPOCHS,
            batch_size=self.config.BATCH_SIZE,
            learning_rate=self.config.LEARNING_RATE,
            save_path=save_path
        )
        
        logger.info("\nTraining completed!")
        logger.info(f"Final training accuracy: {history['accuracy'][-1]:.2f}%")
        logger.info(f"Final validation accuracy: {history['val_accuracy'][-1]:.2f}%")
        
        if save_path:
            logger.info(f"Model saved to {save_path}")
        
        return history


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Child Safety Monitoring System'
    )
    parser.add_argument(
        '--camera', '-c',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )
    parser.add_argument(
        '--save-video', '-s',
        action='store_true',
        help='Save output video'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output video path'
    )
    parser.add_argument(
        '--train', '-t',
        action='store_true',
        help='Train the model before starting'
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        help='Path to training data'
    )
    parser.add_argument(
        '--config', '-cfg',
        type=str,
        help='Path to config file'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = Config()
    if args.config:
        # Load custom config
        import json
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            for key, value in custom_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Create monitor
    monitor = ChildSafetyMonitor(config)
    
    # Train if requested
    if args.train:
        monitor.train_from_data(args.data)
        return
    
    # Start monitoring
    monitor.start(
        camera_id=args.camera,
        save_video=args.save_video,
        output_path=args.output
    )


if __name__ == "__main__":
    main()