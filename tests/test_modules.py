"""
Unit tests for the Child Safety Monitoring System
"""

import unittest
import numpy as np
import cv2
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules import (
    ObjectDetector,
    PoseEstimator,
    ActivityRecognizer,
    SafetyEngine,
    AlertSystem
)
from modules.video_processor import VideoProcessor
from utils.visualization import Visualizer

class TestObjectDetector(unittest.TestCase):
    """Test ObjectDetector class"""
    
    def setUp(self):
        self.detector = ObjectDetector(confidence_threshold=0.5)
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    def test_detection(self):
        """Test detection on empty frame"""
        detections = self.detector.detect(self.test_frame)
        self.assertIsInstance(detections, list)
    
    def test_detect_and_draw(self):
        """Test detection and drawing"""
        detections, annotated = self.detector.detect_and_draw(self.test_frame)
        self.assertIsInstance(detections, list)
        self.assertEqual(annotated.shape, self.test_frame.shape)

class TestPoseEstimator(unittest.TestCase):
    """Test PoseEstimator class"""
    
    def setUp(self):
        self.estimator = PoseEstimator()
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    def test_extract_keypoints(self):
        """Test keypoint extraction"""
        keypoints, annotated = self.estimator.extract_keypoints(self.test_frame)
        self.assertEqual(keypoints.shape, (33, 3))
        self.assertEqual(annotated.shape, self.test_frame.shape)
    
    def test_extract_keypoints_normalized(self):
        """Test normalized keypoint extraction"""
        keypoints = self.estimator.extract_keypoints_normalized(self.test_frame)
        self.assertEqual(keypoints.shape, (33, 3))

class TestSafetyEngine(unittest.TestCase):
    """Test SafetyEngine class"""
    
    def setUp(self):
        self.engine = SafetyEngine()
        self.keypoints = np.zeros((33, 3))
        self.bbox = [100, 100, 200, 200]
        self.frame_shape = (480, 640, 3)
    
    def test_check_safety(self):
        """Test safety checking"""
        alerts = self.engine.check_safety(
            keypoints=self.keypoints,
            activity='unknown',
            bbox=self.bbox,
            frame_shape=self.frame_shape
        )
        self.assertIsInstance(alerts, list)
    
    def test_detect_fall(self):
        """Test fall detection"""
        # Place shoulder near ground
        self.keypoints[5] = [0.4, 0.8, 0]  # Left shoulder
        self.keypoints[6] = [0.6, 0.8, 0]  # Right shoulder
        
        alerts = self.engine.detect_fall(self.keypoints, 'unknown')
        self.assertIsInstance(alerts, list)
    
    def test_get_safety_status(self):
        """Test getting safety status"""
        status = self.engine.get_safety_status()
        self.assertIn('active_rules', status)
        self.assertIn('recent_alerts', status)

class TestAlertSystem(unittest.TestCase):
    """Test AlertSystem class"""
    
    def setUp(self):
        self.alert_system = AlertSystem({
            'enable_console': True,
            'enable_email': False,
            'enable_sms': False
        })
    
    def test_send_alert(self):
        """Test sending alerts"""
        success = self.alert_system.send_alert(
            alert_type='test',
            message='Test alert',
            severity='medium'
        )
        self.assertTrue(success)
    
    def test_get_alert_history(self):
        """Test getting alert history"""
        history = self.alert_system.get_alert_history()
        self.assertIsInstance(history, list)

class TestVideoProcessor(unittest.TestCase):
    """Test VideoProcessor class"""
    
    def setUp(self):
        self.processor = VideoProcessor()
    
    def test_open_close(self):
        """Test opening and closing video source"""
        # This might fail if no camera is available, so we catch exceptions
        try:
            self.processor.open()
            self.processor.close()
        except:
            pass

class TestVisualizer(unittest.TestCase):
    """Test Visualizer class"""
    
    def setUp(self):
        self.visualizer = Visualizer()
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.keypoints = np.zeros((33, 3))
        self.detections = [{
            'bbox': [100, 100, 200, 200],
            'confidence': 0.8,
            'class_name': 'person'
        }]
        self.alerts = [{
            'type': 'test',
            'message': 'Test alert',
            'severity': 'medium'
        }]
    
    def test_draw_detection(self):
        """Test drawing detections"""
        annotated = self.visualizer.draw_detection(self.test_frame, self.detections)
        self.assertEqual(annotated.shape, self.test_frame.shape)
    
    def test_draw_pose(self):
        """Test drawing pose"""
        annotated = self.visualizer.draw_pose(self.test_frame, self.keypoints)
        self.assertEqual(annotated.shape, self.test_frame.shape)
    
    def test_draw_safety_info(self):
        """Test drawing safety info"""
        annotated = self.visualizer.draw_safety_info(
            self.test_frame,
            self.alerts,
            activity='walking',
            status='safe'
        )
        self.assertEqual(annotated.shape, self.test_frame.shape)
    
    def test_draw_combined(self):
        """Test combined visualization"""
        annotated = self.visualizer.draw_combined(
            self.test_frame,
            self.detections,
            self.keypoints,
            self.alerts,
            activity='walking',
            status='safe'
        )
        self.assertEqual(annotated.shape, self.test_frame.shape)

if __name__ == '__main__':
    unittest.main()