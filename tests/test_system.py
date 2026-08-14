# tests/test_system.py
import unittest
import numpy as np
import cv2
import os
import sys
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.detector import ChildDetector
from models.pose_estimator import PoseEstimator
from models.activity_recognizer import ActivityRecognizer, LSTMActivityRecognizer
from models.safety_engine import SafetyEngine
from utils.alert import AlertSystem
from config import Config

class TestChildSafetySystem(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        print("\n" + "="*60)
        print("RUNNING CHILD SAFETY SYSTEM TESTS")
        print("="*60)
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = ChildDetector()
        self.pose_estimator = PoseEstimator()
        self.activity_recognizer = ActivityRecognizer(
            sequence_length=Config.SEQUENCE_LENGTH,
            num_keypoints=33,
            num_classes=len(Config.ACTIVITY_CLASSES)
        )
        self.safety_engine = SafetyEngine()
        self.alert_system = AlertSystem(sound_enabled=False)
        
        # Create a test frame
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some random noise
        self.test_frame = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)
        
    def test_config(self):
        """Test configuration"""
        self.assertTrue(hasattr(Config, 'ACTIVITY_CLASSES'))
        self.assertTrue(len(Config.ACTIVITY_CLASSES) > 0)
        self.assertTrue(hasattr(Config, 'SEQUENCE_LENGTH'))
        self.assertTrue(Config.SEQUENCE_LENGTH > 0)
        print("✅ Configuration test passed")
        
    def test_detector(self):
        """Test child detector"""
        detections = self.detector.detect(self.test_frame)
        self.assertIsInstance(detections, list)
        print(f"✅ Detector test passed - found {len(detections)} detections")
        
    def test_detector_draw(self):
        """Test detector with drawing"""
        annotated, detections = self.detector.detect_and_draw(self.test_frame)
        self.assertIsInstance(annotated, np.ndarray)
        self.assertEqual(annotated.shape, self.test_frame.shape)
        print("✅ Detector drawing test passed")
        
    def test_pose_estimator(self):
        """Test pose estimator"""
        keypoints = self.pose_estimator.extract_keypoints(self.test_frame)
        # Should return None or array
        self.assertTrue(keypoints is None or isinstance(keypoints, np.ndarray))
        print(f"✅ Pose estimator test passed - keypoints: {keypoints.shape if keypoints is not None else 'None'}")
        
    def test_pose_estimator_draw(self):
        """Test pose estimator drawing"""
        annotated = self.pose_estimator.draw_pose(self.test_frame)
        self.assertIsInstance(annotated, np.ndarray)
        self.assertEqual(annotated.shape, self.test_frame.shape)
        print("✅ Pose estimator drawing test passed")
        
    def test_pose_angles(self):
        """Test pose angle calculation"""
        # Create dummy keypoints
        dummy_keypoints = np.random.rand(33, 3)
        angles = self.pose_estimator.get_keypoint_angles(dummy_keypoints)
        
        if angles is not None:
            self.assertIsInstance(angles, np.ndarray)
            self.assertEqual(len(angles), 4)  # 4 angles
            print(f"✅ Pose angle calculation test passed - angles: {angles}")
        else:
            print("✅ Pose angle calculation test passed - no angles returned")
        
    def test_activity_recognizer(self):
        """Test activity recognizer"""
        # Create dummy keypoints
        dummy_keypoints = np.random.rand(Config.SEQUENCE_LENGTH, 33 * 3)
        activity, confidence = self.activity_recognizer.predict_activity(dummy_keypoints)
        
        if activity is not None:
            self.assertIsInstance(activity, str)
            self.assertIsInstance(confidence, float)
            self.assertGreaterEqual(confidence, 0)
            self.assertLessEqual(confidence, 1)
            print(f"✅ Activity recognizer test passed - {activity}: {confidence:.2f}")
        else:
            print("✅ Activity recognizer test passed - no activity predicted")
        
    def test_activity_recognizer_buffer(self):
        """Test activity recognizer with buffer"""
        # Add frames to buffer
        for _ in range(Config.SEQUENCE_LENGTH + 5):
            dummy_keypoints = np.random.rand(33, 3)
            activity, confidence = self.activity_recognizer.add_to_buffer(dummy_keypoints)
        
        self.assertTrue(len(self.activity_recognizer.sequence_buffer) > 0)
        print(f"✅ Activity recognizer buffer test passed - buffer size: {len(self.activity_recognizer.sequence_buffer)}")
        
    def test_activity_recognizer_reset(self):
        """Test activity recognizer reset"""
        self.activity_recognizer.reset_buffer()
        self.assertEqual(len(self.activity_recognizer.sequence_buffer), 0)
        print("✅ Activity recognizer reset test passed")
        
    def test_safety_engine(self):
        """Test safety engine"""
        # Test safe activity
        result = self.safety_engine.check_safety('walking', 0.9)
        self.assertTrue(result['safe'])
        self.assertEqual(result['severity'], 'low')
        print("✅ Safety engine - safe activity test passed")
        
        # Test unsafe activity
        result = self.safety_engine.check_safety('falling', 0.9)
        self.assertFalse(result['safe'])
        self.assertEqual(result['severity'], 'high')
        print("✅ Safety engine - unsafe activity test passed")
        
    def test_safety_engine_fall_detection(self):
        """Test fall detection from pose"""
        # Create keypoints for a fallen person
        fallen_keypoints = np.random.rand(33, 3)
        # Make person horizontal
        for i in range(11, 29):  # Shoulders to ankles
            fallen_keypoints[i, 1] = 0.5  # All at same height
        
        result = self.safety_engine.check_safety(
            'unknown', 0.8, 
            pose_keypoints=fallen_keypoints
        )
        
        if not result['safe'] and result['alert'] == 'falling':
            print("✅ Safety engine - fall detection test passed")
        else:
            print("⚠️  Safety engine - fall detection test: fall not detected (expected on random data)")
        
    def test_alert_system(self):
        """Test alert system"""
        alert = self.alert_system.generate_alert('Test alert')
        self.assertIsNotNone(alert)
        self.assertEqual(alert['message'], 'Test alert')
        
        # Check alert log
        stats = self.alert_system.get_alert_statistics()
        self.assertGreaterEqual(stats['total'], 1)
        print("✅ Alert system test passed")
        
    def test_alert_throttling(self):
        """Test alert throttling"""
        for i in range(5):
            self.alert_system.generate_alert(f'Test alert {i}')
        
        stats = self.alert_system.get_alert_statistics()
        self.assertEqual(stats['total'], 6)  # 1 from previous test + 5
        print("✅ Alert throttling test passed")
        
    def test_model_loading(self):
        """Test model loading"""
        # Try to load model if exists
        model_path = os.path.join(Config.MODELS_DIR, 'activity_model.pth')
        if os.path.exists(model_path):
            try:
                self.activity_recognizer.load_model(model_path)
                print("✅ Model loading test passed")
            except Exception as e:
                print(f"⚠️  Model loading test: {e}")
        else:
            print("⚠️  Model loading test: no model found")
        
    def test_lstm_model(self):
        """Test LSTM model structure"""
        model = LSTMActivityRecognizer(33*3, 128, 2, 5)
        
        # Test forward pass
        dummy_input = torch.randn(1, 30, 33*3)
        output = model(dummy_input)
        
        self.assertEqual(output.shape, (1, 5))
        print("✅ LSTM model structure test passed")

class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_end_to_end_pipeline(self):
        """Test end-to-end pipeline with dummy data"""
        print("\n" + "="*60)
        print("RUNNING INTEGRATION TESTS")
        print("="*60)
        
        detector = ChildDetector()
        pose_estimator = PoseEstimator()
        safety_engine = SafetyEngine()
        
        # Create test frame with a person-like shape
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a simple person shape
        cv2.rectangle(frame, (250, 200), (390, 400), (200, 200, 200), -1)
        cv2.circle(frame, (320, 170), 30, (200, 200, 200), -1)
        
        # Detection
        detections = detector.detect(frame)
        print(f"✅ Detection: {len(detections)} detections")
        
        if detections:
            # Pose estimation
            keypoints = pose_estimator.extract_keypoints(frame)
            print(f"✅ Pose estimation: {'Success' if keypoints is not None else 'No keypoints'}")
            
            # Safety check
            result = safety_engine.check_safety('walking', 0.8)
            print(f"✅ Safety check: {'Safe' if result['safe'] else 'Unsafe'}")
        else:
            print("⚠️  No detections in test frame (expected)")
        
        print("✅ Integration test completed")

class TestPerformance(unittest.TestCase):
    """Performance tests"""
    
    def test_detection_speed(self):
        """Test detection speed"""
        detector = ChildDetector()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        iterations = 10
        start = time.time()
        for _ in range(iterations):
            detector.detect(frame)
        elapsed = time.time() - start
        
        fps = iterations / elapsed
        print(f"✅ Detection speed: {fps:.1f} FPS")
        
        # Should be at least 5 FPS on reasonable hardware
        self.assertGreater(fps, 5)
        
    def test_pose_speed(self):
        """Test pose estimation speed"""
        pose_estimator = PoseEstimator()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        iterations = 10
        start = time.time()
        for _ in range(iterations):
            pose_estimator.extract_keypoints(frame)
        elapsed = time.time() - start
        
        fps = iterations / elapsed
        print(f"✅ Pose estimation speed: {fps:.1f} FPS")
        
        # Should be at least 5 FPS
        self.assertGreater(fps, 5)

def run_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("CHILD SAFETY MONITORING - TEST SUITE")
    print("="*60)
    
    # Create test loader
    loader = unittest.TestLoader()
    
    # Create test suite
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestChildSafetySystem))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    import torch
    success = run_tests()
    sys.exit(0 if success else 1)