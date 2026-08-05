"""
Pose Estimation module using MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PoseEstimator:
    """MediaPipe-based pose estimation for keypoint extraction"""
    
    def __init__(self, detection_confidence=0.5, tracking_confidence=0.5):
        """
        Initialize the pose estimator
        
        Args:
            detection_confidence: Minimum confidence for detection
            tracking_confidence: Minimum confidence for tracking
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        
        # Define keypoint indices
        self.KEYPOINT_INDICES = {
            'nose': 0,
            'left_eye': 1, 'right_eye': 2,
            'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6,
            'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10,
            'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14,
            'left_ankle': 15, 'right_ankle': 16,
            'left_heel': 17, 'right_heel': 18,
            'left_foot_index': 19, 'right_foot_index': 20,
            'left_shoulder_hip_mid': 21, 'right_shoulder_hip_mid': 22,
            'left_hip_knee_mid': 23, 'right_hip_knee_mid': 24,
            'left_knee_ankle_mid': 25, 'right_knee_ankle_mid': 26,
            'left_ankle_heel_mid': 27, 'right_ankle_heel_mid': 28,
            'left_heel_foot_mid': 29, 'right_heel_foot_mid': 30,
            'left_shoulder_elbow_mid': 31, 'right_shoulder_elbow_mid': 32
        }
    
    def extract_keypoints(self, frame, draw=True):
        """
        Extract pose keypoints from a frame
        
        Args:
            frame: Input image (RGB or BGR)
            draw: Whether to draw pose landmarks on frame
            
        Returns:
            Keypoints array (33x3) and annotated frame
        """
        # Convert BGR to RGB if needed
        if frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        keypoints = np.zeros((33, 3))  # 33 landmarks * (x, y, z)
        annotated_frame = frame.copy() if draw else frame
        
        if results.pose_landmarks:
            # Extract keypoints
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                h, w, _ = frame.shape
                keypoints[idx] = [landmark.x * w, landmark.y * h, landmark.z * w]
            
            # Draw landmarks if requested
            if draw:
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
            
            return keypoints, annotated_frame
        
        return keypoints, annotated_frame
    
    def extract_keypoints_normalized(self, frame):
        """
        Extract normalized keypoints (relative to image size)
        
        Args:
            frame: Input image
            
        Returns:
            Normalized keypoints array (33x3) with values in [0, 1]
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        keypoints = np.zeros((33, 3))
        
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                keypoints[idx] = [landmark.x, landmark.y, landmark.z]
        
        return keypoints
    
    def get_keypoint_angles(self, keypoints):
        """
        Calculate angles between keypoints for pose analysis
        
        Args:
            keypoints: Array of keypoints (33x3)
            
        Returns:
            Dictionary of angles
        """
        angles = {}
        
        # Helper function to calculate angle between three points
        def calculate_angle(p1, p2, p3):
            v1 = p1 - p2
            v2 = p3 - p2
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            return np.degrees(angle)
        
        # Calculate joint angles
        # Left arm
        if np.all(keypoints[self.KEYPOINT_INDICES['left_shoulder']] != 0):
            angles['left_elbow'] = calculate_angle(
                keypoints[self.KEYPOINT_INDICES['left_shoulder']],
                keypoints[self.KEYPOINT_INDICES['left_elbow']],
                keypoints[self.KEYPOINT_INDICES['left_wrist']]
            )
        
        # Right arm
        if np.all(keypoints[self.KEYPOINT_INDICES['right_shoulder']] != 0):
            angles['right_elbow'] = calculate_angle(
                keypoints[self.KEYPOINT_INDICES['right_shoulder']],
                keypoints[self.KEYPOINT_INDICES['right_elbow']],
                keypoints[self.KEYPOINT_INDICES['right_wrist']]
            )
        
        # Left leg
        if np.all(keypoints[self.KEYPOINT_INDICES['left_hip']] != 0):
            angles['left_knee'] = calculate_angle(
                keypoints[self.KEYPOINT_INDICES['left_hip']],
                keypoints[self.KEYPOINT_INDICES['left_knee']],
                keypoints[self.KEYPOINT_INDICES['left_ankle']]
            )
        
        # Right leg
        if np.all(keypoints[self.KEYPOINT_INDICES['right_hip']] != 0):
            angles['right_knee'] = calculate_angle(
                keypoints[self.KEYPOINT_INDICES['right_hip']],
                keypoints[self.KEYPOINT_INDICES['right_knee']],
                keypoints[self.KEYPOINT_INDICES['right_ankle']]
            )
        
        # Shoulder angle
        if np.all(keypoints[self.KEYPOINT_INDICES['left_shoulder']] != 0) and \
           np.all(keypoints[self.KEYPOINT_INDICES['right_shoulder']] != 0):
            left_shoulder = keypoints[self.KEYPOINT_INDICES['left_shoulder']]
            right_shoulder = keypoints[self.KEYPOINT_INDICES['right_shoulder']]
            # Calculate orientation (simplified)
            angles['shoulder_orientation'] = np.degrees(
                np.arctan2(right_shoulder[1] - left_shoulder[1],
                          right_shoulder[0] - left_shoulder[0])
            )
        
        return angles