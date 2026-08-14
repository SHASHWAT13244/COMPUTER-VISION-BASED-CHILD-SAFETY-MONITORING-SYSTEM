# models/pose_estimator.py
"""
Pose Estimator using MediaPipe
Extracts 33 body keypoints for activity recognition
"""

import cv2
import mediapipe as mp
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PoseEstimator:
    """
    Pose estimation using MediaPipe
    Extracts 33 body keypoints and calculates angles
    """
    
    # MediaPipe pose keypoint indices
    KEYPOINT_INDICES = {
        'nose': 0,
        'left_eye_inner': 1,
        'left_eye': 2,
        'left_eye_outer': 3,
        'right_eye_inner': 4,
        'right_eye': 5,
        'right_eye_outer': 6,
        'left_ear': 7,
        'right_ear': 8,
        'mouth_left': 9,
        'mouth_right': 10,
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
        'left_pinky': 17,
        'right_pinky': 18,
        'left_index': 19,
        'right_index': 20,
        'left_thumb': 21,
        'right_thumb': 22,
        'left_hip': 23,
        'right_hip': 24,
        'left_knee': 25,
        'right_knee': 26,
        'left_ankle': 27,
        'right_ankle': 28,
        'left_heel': 29,
        'right_heel': 30,
        'left_foot_index': 31,
        'right_foot_index': 32,
    }
    
    # Key connections for drawing
    POSE_CONNECTIONS = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Shoulders to wrists
        (11, 23), (12, 24), (23, 24),  # Shoulders to hips
        (23, 25), (25, 27), (27, 29), (29, 31),  # Left leg
        (24, 26), (26, 28), (28, 30), (30, 32),  # Right leg
        (11, 21), (12, 22),  # Shoulders to thumbs
        (15, 17), (15, 19), (15, 21),  # Left hand
        (16, 18), (16, 20), (16, 22),  # Right hand
        (0, 1), (1, 2), (2, 3), (3, 7),  # Face
        (0, 4), (4, 5), (5, 6), (6, 8),  # Face
        (9, 10),  # Mouth
        (7, 9), (8, 10),  # Mouth to ears
    ]
    
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5, 
                 model_complexity=1):
        """
        Initialize MediaPipe Pose estimator
        
        Args:
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            model_complexity: 0, 1, or 2 (higher = more accurate but slower)
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            enable_segmentation=False
        )
        
        self.num_keypoints = 33
        self.keypoint_names = list(self.KEYPOINT_INDICES.keys())
        
    def extract_keypoints(self, frame):
        """
        Extract pose keypoints from frame
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            numpy array of keypoints (33 x 4) with x, y, z, visibility
            or None if no pose detected
        """
        if frame is None or frame.size == 0:
            return None
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.pose.process(rgb_frame)
            
            if not results.pose_landmarks:
                return None
            
            # Extract keypoints (x, y, z, visibility)
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                keypoints.append([
                    landmark.x,
                    landmark.y,
                    landmark.z,
                    landmark.visibility
                ])
            
            return np.array(keypoints)
            
        except Exception as e:
            logger.error(f"Error in pose estimation: {e}")
            return None
    
    def extract_keypoints_batch(self, frames):
        """Extract keypoints from multiple frames"""
        keypoints_list = []
        for frame in frames:
            kp = self.extract_keypoints(frame)
            keypoints_list.append(kp)
        return keypoints_list
    
    def draw_pose(self, frame, keypoints=None, draw_connections=True, 
                  draw_landmarks=True, landmark_color=(0, 255, 0), 
                  connection_color=(255, 0, 0)):
        """
        Draw pose landmarks on frame
        
        Args:
            frame: Input image (BGR format)
            keypoints: Optional keypoints to draw (if None, uses MediaPipe)
            draw_connections: Whether to draw connections between landmarks
            draw_landmarks: Whether to draw individual landmarks
            landmark_color: Color for landmarks
            connection_color: Color for connections
            
        Returns:
            annotated_frame: Frame with pose drawn
        """
        annotated_frame = frame.copy()
        
        try:
            # Use MediaPipe's built-in drawing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                if draw_landmarks:
                    # Draw landmarks with custom style
                    for idx, landmark in enumerate(results.pose_landmarks.landmark):
                        h, w = frame.shape[:2]
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(annotated_frame, (cx, cy), 3, landmark_color, -1)
                        
                        # Draw keypoint index
                        cv2.putText(annotated_frame, str(idx), (cx+5, cy-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                
                if draw_connections:
                    # Draw connections
                    for connection in self.POSE_CONNECTIONS:
                        idx1, idx2 = connection
                        if idx1 < len(results.pose_landmarks.landmark) and \
                           idx2 < len(results.pose_landmarks.landmark):
                            lm1 = results.pose_landmarks.landmark[idx1]
                            lm2 = results.pose_landmarks.landmark[idx2]
                            h, w = frame.shape[:2]
                            x1, y1 = int(lm1.x * w), int(lm1.y * h)
                            x2, y2 = int(lm2.x * w), int(lm2.y * h)
                            cv2.line(annotated_frame, (x1, y1), (x2, y2), connection_color, 2)
                
            return annotated_frame
            
        except Exception as e:
            logger.error(f"Error drawing pose: {e}")
            return annotated_frame
    
    def get_keypoint_angles(self, keypoints):
        """
        Calculate joint angles from keypoints for activity recognition
        
        Args:
            keypoints: numpy array of keypoints (33 x 4)
            
        Returns:
            numpy array of angles
        """
        if keypoints is None or len(keypoints) < 33:
            return None
        
        angles = []
        
        try:
            # Get relevant joints
            left_shoulder = keypoints[11]
            right_shoulder = keypoints[12]
            left_elbow = keypoints[13]
            right_elbow = keypoints[14]
            left_wrist = keypoints[15]
            right_wrist = keypoints[16]
            left_hip = keypoints[23]
            right_hip = keypoints[24]
            left_knee = keypoints[25]
            right_knee = keypoints[26]
            left_ankle = keypoints[27]
            right_ankle = keypoints[28]
            
            # Calculate angles
            # Left elbow angle
            angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
            angles.append(angle)
            
            # Right elbow angle
            angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
            angles.append(angle)
            
            # Left shoulder angle
            angle = self._calculate_angle(left_elbow, left_shoulder, left_hip)
            angles.append(angle)
            
            # Right shoulder angle
            angle = self._calculate_angle(right_elbow, right_shoulder, right_hip)
            angles.append(angle)
            
            # Left hip angle
            angle = self._calculate_angle(left_shoulder, left_hip, left_knee)
            angles.append(angle)
            
            # Right hip angle
            angle = self._calculate_angle(right_shoulder, right_hip, right_knee)
            angles.append(angle)
            
            # Left knee angle
            angle = self._calculate_angle(left_hip, left_knee, left_ankle)
            angles.append(angle)
            
            # Right knee angle
            angle = self._calculate_angle(right_hip, right_knee, right_ankle)
            angles.append(angle)
            
            # Shoulder angle (between shoulders)
            angle = self._calculate_angle(left_shoulder, 
                                         (left_shoulder + right_shoulder) / 2, 
                                         right_shoulder)
            angles.append(angle)
            
            # Hip angle (between hips)
            angle = self._calculate_angle(left_hip, 
                                         (left_hip + right_hip) / 2, 
                                         right_hip)
            angles.append(angle)
            
            return np.array(angles)
            
        except Exception as e:
            logger.error(f"Error calculating angles: {e}")
            return None
    
    def _calculate_angle(self, a, b, c):
        """
        Calculate angle between three points
        
        Args:
            a, b, c: Points as (x, y) or keypoint arrays
            
        Returns:
            angle in degrees
        """
        try:
            # Extract coordinates
            if hasattr(a, '__len__') and len(a) >= 2:
                a_coord = np.array([a[0], a[1]])
            else:
                a_coord = np.array(a)
                
            if hasattr(b, '__len__') and len(b) >= 2:
                b_coord = np.array([b[0], b[1]])
            else:
                b_coord = np.array(b)
                
            if hasattr(c, '__len__') and len(c) >= 2:
                c_coord = np.array([c[0], c[1]])
            else:
                c_coord = np.array(c)
            
            # Calculate vectors
            ba = a_coord - b_coord
            bc = c_coord - b_coord
            
            # Calculate angle
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            angle = np.arccos(cosine_angle)
            
            return np.degrees(angle)
            
        except Exception as e:
            logger.error(f"Error calculating angle: {e}")
            return 0.0
    
    def get_keypoint_distances(self, keypoints):
        """Calculate distances between keypoints"""
        if keypoints is None or len(keypoints) < 33:
            return None
        
        distances = []
        
        try:
            # Define keypoint pairs
            pairs = [
                (11, 12),  # Shoulders
                (11, 13),  # Left shoulder to elbow
                (12, 14),  # Right shoulder to elbow
                (13, 15),  # Left elbow to wrist
                (14, 16),  # Right elbow to wrist
                (11, 23),  # Left shoulder to hip
                (12, 24),  # Right shoulder to hip
                (23, 25),  # Left hip to knee
                (24, 26),  # Right hip to knee
                (25, 27),  # Left knee to ankle
                (26, 28),  # Right knee to ankle
            ]
            
            for idx1, idx2 in pairs:
                if idx1 < len(keypoints) and idx2 < len(keypoints):
                    p1 = keypoints[idx1][:2]  # x, y
                    p2 = keypoints[idx2][:2]
                    distance = np.linalg.norm(p1 - p2)
                    distances.append(distance)
            
            return np.array(distances)
            
        except Exception as e:
            logger.error(f"Error calculating distances: {e}")
            return None
    
    def get_pose_center(self, keypoints):
        """Get center of pose"""
        if keypoints is None or len(keypoints) < 33:
            return None
        
        try:
            # Use shoulders and hips to estimate center
            left_shoulder = keypoints[11][:2]
            right_shoulder = keypoints[12][:2]
            left_hip = keypoints[23][:2]
            right_hip = keypoints[24][:2]
            
            center = (left_shoulder + right_shoulder + left_hip + right_hip) / 4
            return center
            
        except Exception as e:
            logger.error(f"Error getting pose center: {e}")
            return None
    
    def is_pose_visible(self, keypoints, threshold=0.5):
        """Check if pose is visible"""
        if keypoints is None:
            return False
        
        try:
            # Check visibility of keypoints
            visible_count = np.sum(keypoints[:, 3] > threshold)
            return visible_count > 10  # At least 10 visible keypoints
            
        except Exception:
            return False