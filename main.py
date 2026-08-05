"""
Main application for Child Safety Monitoring System
"""

import sys
import time
import cv2
import numpy as np
import logging
from pathlib import Path
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

# Import modules
from modules import (
    ObjectDetector,
    PoseEstimator,
    ActivityRecognizer,
    SafetyEngine,
    AlertSystem
)
from modules.video_processor import VideoProcessor
from utils.visualization import Visualizer
import config

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.SYSTEM_SETTINGS['log_level']),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.SYSTEM_SETTINGS['log_file']),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ChildSafetyMonitoringSystem:
    """Main application class for child safety monitoring"""
    
    def __init__(self, config_module=None):
        """
        Initialize the monitoring system
        
        Args:
            config_module: Configuration module or dictionary
        """
        self.config = config_module or config
        
        # Initialize components
        self.setup_components()
        
        # Initialize video processor
        self.video_processor = VideoProcessor(
            source=self.config.VIDEO_SETTINGS['camera_index'],
            width=self.config.VIDEO_SETTINGS['frame_width'],
            height=self.config.VIDEO_SETTINGS['frame_height'],
            fps=self.config.VIDEO_SETTINGS['fps']
        )
        
        # Initialize visualizer
        self.visualizer = Visualizer()
        
        # State variables
        self.running = False
        self.frame_count = 0
        self.activity_buffer = []
        self.fps_tracker = []
        
        logger.info("Child Safety Monitoring System initialized")
    
    def setup_components(self):
        """Setup all monitoring components"""
        try:
            # Object Detector
            self.object_detector = ObjectDetector(
                model_path=str(config.YOLO_MODEL_PATH),
                confidence_threshold=config.CONFIDENCE_THRESHOLD,
                device='cuda' if config.SYSTEM_SETTINGS['use_gpu'] else 'cpu'
            )
            logger.info("Object Detector initialized")
            
            # Pose Estimator
            self.pose_estimator = PoseEstimator(
                detection_confidence=config.POSE_DETECTION_CONFIDENCE,
                tracking_confidence=config.POSE_TRACKING_CONFIDENCE
            )
            logger.info("Pose Estimator initialized")
            
            # Activity Recognizer
            self.activity_recognizer = ActivityRecognizer(
                sequence_length=config.SEQUENCE_LENGTH,
                features_per_frame=config.FEATURES_PER_FRAME,
                num_classes=len(config.ACTIVITY_CLASSES)
            )
            self.activity_recognizer.initialize_model()
            logger.info("Activity Recognizer initialized")
            
            # Safety Engine
            self.safety_engine = SafetyEngine(config=config.SAFETY_RULES)
            logger.info("Safety Engine initialized")
            
            # Alert System
            self.alert_system = AlertSystem(config=config.ALERT_CONFIG)
            logger.info("Alert System initialized")
            
        except Exception as e:
            logger.error(f"Failed to setup components: {e}")
            raise
    
    def process_frame(self, frame):
        """
        Process a single frame
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame with annotations
        """
        if frame is None:
            return None
        
        self.frame_count += 1
        
        # Start timing for FPS calculation
        start_time = time.time()
        
        # 1. Object Detection
        detections, detection_frame = self.object_detector.detect_and_draw(frame)
        
        # 2. Pose Estimation
        keypoints, pose_frame = self.pose_estimator.extract_keypoints(frame)
        
        # 3. Activity Recognition (if we have a person detected)
        activity = 'unknown'
        activity_confidence = 0
        
        if len(detections) > 0 and np.any(keypoints != 0):
            # Update activity buffer
            sequence = self.activity_recognizer.update_buffer(keypoints)
            
            if sequence is not None:
                try:
                    activity, activity_confidence = self.activity_recognizer.predict(sequence)
                    self.activity_buffer.append((time.time(), activity))
                    
                    # Keep buffer size manageable
                    if len(self.activity_buffer) > 100:
                        self.activity_buffer = self.activity_buffer[-50:]
                    
                except Exception as e:
                    logger.debug(f"Activity prediction failed: {e}")
        
        # 4. Safety Checking
        alerts = []
        status = 'safe'
        
        if len(detections) > 0:
            # Get the first detection for safety checking
            bbox = detections[0]['bbox']
            frame_shape = frame.shape
            
            # Update safety engine buffer
            if np.any(keypoints != 0):
                self.safety_engine.update_buffer(keypoints)
            
            # Check safety rules
            alerts = self.safety_engine.check_safety(
                keypoints=keypoints,
                activity=activity,
                bbox=bbox,
                frame_shape=frame_shape
            )
            
            # Determine overall status
            if alerts:
                high_severity = any(alert.get('severity') == 'high' for alert in alerts)
                medium_severity = any(alert.get('severity') == 'medium' for alert in alerts)
                
                if high_severity:
                    status = 'danger'
                elif medium_severity:
                    status = 'warning'
                else:
                    status = 'warning'
            
            # Send alerts for new events
            for alert in alerts:
                # Check if this alert was already sent
                alert_key = f"{alert['type']}_{alert['message']}"
                if not hasattr(self, '_sent_alerts'):
                    self._sent_alerts = []
                
                if alert_key not in self._sent_alerts[-10:]:
                    self.alert_system.send_alert(
                        alert_type=alert['type'],
                        message=alert['message'],
                        severity=alert['severity']
                    )
                    self._sent_alerts.append(alert_key)
        
        # 5. Visualization
        # Combine all annotations
        annotated_frame = self.visualizer.draw_combined(
            frame=frame,
            detections=detections,
            keypoints=keypoints,
            alerts=alerts,
            activity=activity if activity != 'unknown' else None,
            status=status
        )
        
        # Calculate FPS
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            fps = 1.0 / elapsed_time
            self.fps_tracker.append(fps)
            
            # Keep FPS buffer manageable
            if len(self.fps_tracker) > 30:
                self.fps_tracker = self.fps_tracker[-30:]
            
            avg_fps = np.mean(self.fps_tracker)
            cv2.putText(annotated_frame, f"FPS: {avg_fps:.1f}", 
                       (10, annotated_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Record if enabled
        if self.config.VIDEO_SETTINGS['record_output'] and self.video_processor.is_recording:
            self.video_processor.write_frame(annotated_frame)
        
        return annotated_frame
    
    def run(self):
        """Run the monitoring system"""
        try:
            # Open video source
            if not self.video_processor.open():
                logger.error("Failed to open video source")
                return
            
            # Start recording if enabled
            if self.config.VIDEO_SETTINGS['record_output']:
                self.video_processor.start_recording(
                    self.config.VIDEO_SETTINGS['output_video_path']
                )
            
            self.running = True
            logger.info("Monitoring started. Press 'q' to quit.")
            
            # Initialize alert tracking
            self._sent_alerts = []
            
            while self.running:
                # Read frame
                frame = self.video_processor.read_frame()
                
                if frame is None:
                    logger.info("End of video stream")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                if processed_frame is not None:
                    # Display
                    cv2.imshow('Child Safety Monitoring System', processed_frame)
                
                # Check for exit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Stopping monitoring...")
                    break
                elif key == ord('r'):
                    # Toggle recording
                    if self.video_processor.is_recording:
                        self.video_processor.stop_recording()
                        logger.info("Recording stopped")
                    else:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        path = config.OUTPUT_DIR / f"monitoring_{timestamp}.avi"
                        self.video_processor.start_recording(path)
                        logger.info(f"Recording started: {path}")
                elif key == ord('s'):
                    # Save screenshot
                    if processed_frame is not None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        path = config.OUTPUT_DIR / f"screenshot_{timestamp}.jpg"
                        cv2.imwrite(str(path), processed_frame)
                        logger.info(f"Screenshot saved: {path}")
                elif key == ord('a'):
                    # Send test alert
                    self.alert_system.send_test_alert()
                    logger.info("Test alert sent")
            
        except KeyboardInterrupt:
            logger.info("Monitoring interrupted by user")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        
        # Stop recording
        self.video_processor.stop_recording()
        
        # Close video processor
        self.video_processor.close()
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
        
        logger.info("System shutdown complete")
    
    def test_model(self):
        """Test the system with sample data"""
        logger.info("Running system test...")
        
        # Test object detection
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections, _ = self.object_detector.detect_and_draw(test_frame)
        logger.info(f"Detection test: {len(detections)} detections")
        
        # Test pose estimation
        keypoints, _ = self.pose_estimator.extract_keypoints(test_frame)
        logger.info(f"Pose estimation test: {np.sum(keypoints != 0)} keypoints detected")
        
        # Test activity recognition
        try:
            # Generate sample data
            X, y = self.activity_recognizer.generate_sample_data(num_samples=100)
            logger.info(f"Activity recognition test: Generated {len(X)} samples")
        except Exception as e:
            logger.error(f"Activity recognition test failed: {e}")
        
        # Test safety engine
        safety_status = self.safety_engine.get_safety_status()
        logger.info(f"Safety engine test: {safety_status}")
        
        # Test alert system
        success = self.alert_system.send_test_alert()
        logger.info(f"Alert system test: {'Success' if success else 'Failed'}")
        
        logger.info("System test completed")
        return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Child Safety Monitoring System')
    parser.add_argument('--source', type=str, default='0',
                       help='Video source: camera index or file path')
    parser.add_argument('--test', action='store_true',
                       help='Run system test')
    parser.add_argument('--no-record', action='store_true',
                       help='Disable video recording')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    if args.source:
        try:
            config.VIDEO_SETTINGS['camera_index'] = int(args.source)
        except ValueError:
            config.VIDEO_SETTINGS['camera_index'] = args.source
    
    if args.no_record:
        config.VIDEO_SETTINGS['record_output'] = False
    
    if args.debug:
        config.SYSTEM_SETTINGS['debug_mode'] = True
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run system
    system = ChildSafetyMonitoringSystem()
    
    if args.test:
        system.test_model()
    else:
        system.run()

if __name__ == "__main__":
    main()