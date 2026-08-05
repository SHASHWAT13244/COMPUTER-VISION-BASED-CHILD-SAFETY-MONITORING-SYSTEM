"""
Safety Engine module for detecting unsafe events
"""

import numpy as np
import logging
from collections import deque

logger = logging.getLogger(__name__)

class SafetyEngine:
    """Safety rule engine for detecting unsafe behaviors"""
    
    def __init__(self, config=None):
        """
        Initialize the safety engine
        
        Args:
            config: Configuration dictionary for safety rules
        """
        self.config = config or {}
        self.alert_history = deque(maxlen=10)  # Store recent alerts
        self.event_buffer = deque(maxlen=30)   # Buffer for sequence analysis
        
        # Initialize safety rules
        self.rules = {
            'fall_detection': self.config.get('fall_detection', True),
            'running_detection': self.config.get('running_detection', True),
            'boundary_detection': self.config.get('boundary_detection', True),
            'climbing_detection': self.config.get('climbing_detection', True),
            'restricted_zones': self.config.get('restricted_zones', [])
        }
        
        # Thresholds
        self.thresholds = {
            'fall_angle': 30,  # Degrees
            'running_speed': 0.8,  # Relative speed
            'climbing_angle': 60,  # Degrees
            'boundary_margin': 0.1  # Normalized distance from boundary
        }
    
    def check_safety(self, keypoints, activity, bbox=None, frame_shape=None):
        """
        Check for safety violations
        
        Args:
            keypoints: Array of keypoints (33x3)
            activity: Current activity class
            bbox: Bounding box of the person
            frame_shape: Shape of the frame (height, width)
            
        Returns:
            List of safety alerts
        """
        alerts = []
        
        # Check each safety rule
        if self.rules['fall_detection']:
            fall_alerts = self.detect_fall(keypoints, activity)
            alerts.extend(fall_alerts)
        
        if self.rules['running_detection']:
            running_alerts = self.detect_unsafe_running(keypoints, activity)
            alerts.extend(running_alerts)
        
        if self.rules['boundary_detection'] and bbox is not None and frame_shape is not None:
            boundary_alerts = self.detect_boundary_violation(bbox, frame_shape)
            alerts.extend(boundary_alerts)
        
        if self.rules['climbing_detection']:
            climbing_alerts = self.detect_climbing(keypoints, activity)
            alerts.extend(climbing_alerts)
        
        # Filter duplicate alerts
        unique_alerts = self.filter_alerts(alerts)
        
        return unique_alerts
    
    def detect_fall(self, keypoints, activity):
        """
        Detect fall events based on keypoints and activity
        
        Args:
            keypoints: Array of keypoints
            activity: Current activity class
            
        Returns:
            List of fall alerts
        """
        alerts = []
        
        # Check if activity is already classified as fall
        if activity == 'falling':
            alerts.append({
                'type': 'fall',
                'severity': 'high',
                'message': 'Fall detected based on activity classification'
            })
            return alerts
        
        # Check keypoint-based fall detection
        try:
            # Get shoulder and hip keypoints
            left_shoulder = keypoints[5]  # Index for left shoulder
            right_shoulder = keypoints[6]  # Index for right shoulder
            left_hip = keypoints[11]  # Index for left hip
            right_hip = keypoints[12]  # Index for right hip
            
            # Check if keypoints are detected
            if np.all(left_shoulder != 0) and np.all(right_shoulder != 0):
                # Calculate shoulder center
                shoulder_center = (left_shoulder + right_shoulder) / 2
                
                # Check if shoulders are near the ground (low y-coordinate)
                if shoulder_center[1] > 0.7:  # Normalized y-coordinate
                    alerts.append({
                        'type': 'fall',
                        'severity': 'high',
                        'message': 'Fall detected - shoulders near ground level'
                    })
            
            # Check if activity is running and suddenly stops (potential fall)
            if activity == 'running' and len(self.event_buffer) > 0:
                previous_activities = list(self.event_buffer)[-5:]
                if 'running' in previous_activities and len(previous_activities) > 3:
                    # If was running and now activity is something else (except running)
                    if activity != 'running':
                        alerts.append({
                            'type': 'fall',
                            'severity': 'medium',
                            'message': 'Potential fall - sudden stop from running'
                        })
        
        except Exception as e:
            logger.debug(f"Error in fall detection: {e}")
        
        return alerts
    
    def detect_unsafe_running(self, keypoints, activity):
        """
        Detect unsafe running in restricted areas
        
        Args:
            keypoints: Array of keypoints
            activity: Current activity class
            
        Returns:
            List of running alerts
        """
        alerts = []
        
        if activity == 'running':
            # Check speed based on keypoint movement
            if len(self.event_buffer) >= 5:
                # Calculate average displacement of keypoints
                recent_keypoints = list(self.event_buffer)[-5:]
                displacements = []
                
                for i in range(1, len(recent_keypoints)):
                    prev_kp = recent_keypoints[i-1]
                    curr_kp = recent_keypoints[i]
                    
                    # Calculate movement of keypoints
                    movement = np.mean(np.linalg.norm(curr_kp - prev_kp, axis=1))
                    displacements.append(movement)
                
                avg_speed = np.mean(displacements)
                
                if avg_speed > self.thresholds['running_speed']:
                    alerts.append({
                        'type': 'unsafe_running',
                        'severity': 'medium',
                        'message': f'High-speed running detected (speed: {avg_speed:.2f})'
                    })
        
        return alerts
    
    def detect_boundary_violation(self, bbox, frame_shape):
        """
        Detect if person enters restricted zones
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            frame_shape: Shape of frame (height, width)
            
        Returns:
            List of boundary alerts
        """
        alerts = []
        
        if not self.rules['restricted_zones']:
            return alerts
        
        h, w = frame_shape[:2]
        
        # Normalize bbox coordinates
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / (2 * w)
        center_y = (y1 + y2) / (2 * h)
        
        # Check if center is in any restricted zone
        for zone in self.rules['restricted_zones']:
            x_range = zone.get('x', [0, 1])
            y_range = zone.get('y', [0, 1])
            
            if (x_range[0] <= center_x <= x_range[1] and
                y_range[0] <= center_y <= y_range[1]):
                alerts.append({
                    'type': 'boundary_violation',
                    'severity': 'high',
                    'message': f'Child entered restricted zone at ({center_x:.2f}, {center_y:.2f})'
                })
                break
        
        return alerts
    
    def detect_climbing(self, keypoints, activity):
        """
        Detect climbing behavior
        
        Args:
            keypoints: Array of keypoints
            activity: Current activity class
            
        Returns:
            List of climbing alerts
        """
        alerts = []
        
        try:
            # Get relevant keypoints
            left_wrist = keypoints[9]   # Index for left wrist
            right_wrist = keypoints[10]  # Index for right wrist
            left_shoulder = keypoints[5] # Index for left shoulder
            right_shoulder = keypoints[6] # Index for right shoulder
            
            # Check if wrists are above shoulders (climbing motion)
            if np.all(left_wrist != 0) and np.all(left_shoulder != 0):
                if left_wrist[1] < left_shoulder[1] - 0.2:  # Wrist is above shoulder
                    alerts.append({
                        'type': 'climbing',
                        'severity': 'high',
                        'message': 'Climbing detected - left arm raised above shoulder'
                    })
            
            if np.all(right_wrist != 0) and np.all(right_shoulder != 0):
                if right_wrist[1] < right_shoulder[1] - 0.2:  # Wrist is above shoulder
                    alerts.append({
                        'type': 'climbing',
                        'severity': 'high',
                        'message': 'Climbing detected - right arm raised above shoulder'
                    })
        
        except Exception as e:
            logger.debug(f"Error in climbing detection: {e}")
        
        return alerts
    
    def filter_alerts(self, alerts):
        """
        Filter duplicate alerts based on recent history
        
        Args:
            alerts: List of alerts
            
        Returns:
            Filtered list of alerts
        """
        unique_alerts = []
        current_time = len(self.alert_history)  # Use buffer length as timestamp
        
        for alert in alerts:
            is_duplicate = False
            
            # Check if similar alert occurred recently
            for existing_alert in self.alert_history:
                if (existing_alert.get('type') == alert.get('type') and
                    existing_alert.get('message') == alert.get('message')):
                    # Check if within cooldown period
                    if current_time - existing_alert.get('timestamp', 0) < 10:  # 10 second cooldown
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                alert['timestamp'] = current_time
                unique_alerts.append(alert)
                self.alert_history.append(alert)
        
        return unique_alerts
    
    def update_buffer(self, keypoints):
        """
        Update the keypoint buffer for sequence analysis
        
        Args:
            keypoints: Array of keypoints
        """
        self.event_buffer.append(keypoints)
    
    def get_safety_status(self):
        """
        Get current safety status
        
        Returns:
            Dictionary with safety status information
        """
        return {
            'active_rules': [rule for rule, enabled in self.rules.items() if enabled],
            'recent_alerts': list(self.alert_history)[-5:],
            'event_buffer_size': len(self.event_buffer)
        }