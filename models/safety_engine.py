# models/safety_engine.py
"""
Safety Rule Engine
Detects unsafe behaviors and generates alerts
"""

import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SafetyEngine:
    """
    Safety rule engine for detecting unsafe behaviors
    """
    
    def __init__(self):
        """
        Initialize the safety rule engine
        """
        self.unsafe_activities = ['falling', 'climbing', 'dangerous_running']
        self.unsafe_zones = ['kitchen', 'pool_area', 'stairs', 'balcony']
        
        self.safety_rules = {
            'fall_detection': {
                'threshold': 0.6,
                'severity': 'high',
                'message': 'Fall detected! Child may be injured.',
                'type': 'fall'
            },
            'climbing_detection': {
                'threshold': 0.6,
                'severity': 'high',
                'message': 'Child climbing on unsafe surface!',
                'type': 'climbing'
            },
            'running_detection': {
                'threshold': 0.7,
                'severity': 'medium',
                'message': 'Running detected! Risk of injury.',
                'type': 'running'
            },
            'boundary_crossing': {
                'threshold': 0.5,
                'severity': 'medium',
                'message': 'Child has entered restricted area!',
                'type': 'boundary'
            },
            'suspicious_pose': {
                'threshold': 0.5,
                'severity': 'medium',
                'message': 'Unusual posture detected!',
                'type': 'pose'
            }
        }
        
        self.alert_history = []
        self.alert_cooldown = 5  # seconds between alerts
        self.alert_count = 0
        self.max_alerts_per_minute = 10
        
        # For tracking state
        self.previous_activity = None
        self.activity_history = []
        self.fall_counter = 0
        self.min_fall_frames = 5  # Minimum frames to confirm fall
        
    def check_safety(self, activity, confidence, pose_keypoints=None, bbox=None, 
                     frame_time=None, zone=None):
        """
        Check if current activity/pose is unsafe
        
        Args:
            activity: Current activity label
            confidence: Confidence of activity detection
            pose_keypoints: Pose keypoints for detailed analysis
            bbox: Bounding box of the person
            frame_time: Timestamp of frame
            zone: Current zone (e.g., 'kitchen', 'stairs')
            
        Returns:
            result: Dictionary with safety status and alert info
        """
        if frame_time is None:
            frame_time = datetime.now()
        
        result = {
            'safe': True,
            'alert': None,
            'severity': 'low',
            'message': 'All safe',
            'detected_activity': activity,
            'confidence': confidence,
            'timestamp': frame_time.isoformat()
        }
        
        # Track activity history
        if activity and activity != 'unknown':
            self.activity_history.append({
                'activity': activity,
                'confidence': confidence,
                'timestamp': frame_time
            })
            if len(self.activity_history) > 50:
                self.activity_history = self.activity_history[-30:]
        
        # Check for unsafe activities
        if activity and confidence > 0.5:
            # Check if activity is unsafe
            if activity in self.unsafe_activities:
                # Special handling for falling
                if activity == 'falling':
                    self.fall_counter += 1
                    if self.fall_counter >= self.min_fall_frames:
                        result['safe'] = False
                        result['severity'] = 'high'
                        result['alert'] = 'falling'
                        result['message'] = 'Fall detected! Immediate attention required.'
                else:
                    result['safe'] = False
                    result['severity'] = 'high'
                    result['alert'] = activity
                    rule = self.safety_rules.get(f'{activity}_detection', {})
                    result['message'] = rule.get('message', f'{activity} detected!')
                    self.fall_counter = 0
            
            elif activity == 'running' and confidence > 0.7:
                result['safe'] = False
                result['severity'] = 'medium'
                result['alert'] = 'running'
                result['message'] = 'Running detected! Risk of injury.'
                self.fall_counter = 0
            
            else:
                self.fall_counter = 0
        
        # Check pose for fall if activity is not falling
        if pose_keypoints is not None and activity != 'falling' and activity != 'climbing':
            if self._detect_fall_pose(pose_keypoints):
                result['safe'] = False
                result['severity'] = 'high'
                result['alert'] = 'falling'
                result['message'] = 'Fall detected from pose analysis!'
                self.fall_counter += 1
        
        # Check zone safety
        if zone and zone in self.unsafe_zones:
            result['safe'] = False
            result['severity'] = 'high'
            result['alert'] = 'zone_violation'
            result['message'] = f'Child in restricted area: {zone}!'
        
        # Check for multiple unsafe events
        if not result['safe']:
            if self._should_alert(result['alert'], frame_time):
                result['alert_generated'] = True
                self.alert_count += 1
            else:
                result['alert_generated'] = False
        
        return result
    
    def _detect_fall_pose(self, keypoints):
        """
        Detect fall based on keypoint analysis
        
        Args:
            keypoints: numpy array of keypoints (33 x 4)
            
        Returns:
            bool: True if fall detected
        """
        if keypoints is None or len(keypoints) < 33:
            return False
        
        try:
            # Get key positions
            nose = keypoints[0]
            left_shoulder = keypoints[11]
            right_shoulder = keypoints[12]
            left_hip = keypoints[23]
            right_hip = keypoints[24]
            left_knee = keypoints[25]
            right_knee = keypoints[26]
            left_ankle = keypoints[27]
            right_ankle = keypoints[28]
            
            # Calculate heights
            shoulder_avg_y = (left_shoulder[1] + right_shoulder[1]) / 2
            hip_avg_y = (left_hip[1] + right_hip[1]) / 2
            knee_avg_y = (left_knee[1] + right_knee[1]) / 2
            ankle_avg_y = (left_ankle[1] + right_ankle[1]) / 2
            
            # Check if person is horizontal (falling)
            # In a fall, shoulders and hips are close to the ground
            shoulder_ankle_diff = abs(shoulder_avg_y - ankle_avg_y)
            hip_ankle_diff = abs(hip_avg_y - ankle_avg_y)
            shoulder_hip_diff = abs(shoulder_avg_y - hip_avg_y)
            
            # Normalize by image height (assume keypoints are normalized 0-1)
            # Fall indicators:
            # 1. Shoulders and hips are close to ankles (horizontal)
            is_horizontal = (shoulder_ankle_diff < 0.2 and hip_ankle_diff < 0.2)
            
            # 2. Shoulders and hips are at similar height (collapsed)
            is_collapsed = (shoulder_hip_diff < 0.1 and shoulder_ankle_diff < 0.3)
            
            # 3. Nose is close to ground
            nose_ground_diff = abs(nose[1] - ankle_avg_y)
            is_nose_down = nose_ground_diff < 0.15
            
            # Check visibility of keypoints
            shoulder_visible = left_shoulder[3] > 0.5 and right_shoulder[3] > 0.5
            hip_visible = left_hip[3] > 0.5 and right_hip[3] > 0.5
            
            if not (shoulder_visible and hip_visible):
                return False
            
            # Combine indicators
            fall_score = 0
            if is_horizontal:
                fall_score += 2
            if is_collapsed:
                fall_score += 2
            if is_nose_down:
                fall_score += 1
            
            return fall_score >= 3
            
        except Exception as e:
            logger.error(f"Error in fall detection: {e}")
            return False
    
    def _should_alert(self, alert_type, current_time):
        """
        Check if alert should be generated (rate limiting)
        
        Args:
            alert_type: Type of alert
            current_time: Current timestamp
            
        Returns:
            bool: True if alert should be generated
        """
        # Check cooldown for same alert type
        for history in self.alert_history:
            if history.get('type') == alert_type:
                time_diff = (current_time - history['time']).total_seconds()
                if time_diff < self.alert_cooldown:
                    return False
        
        # Check rate limit
        one_minute_ago = current_time - timedelta(minutes=1)
        recent_alerts = [a for a in self.alert_history 
                        if a['time'] > one_minute_ago]
        
        if len(recent_alerts) >= self.max_alerts_per_minute:
            logger.warning(f"Alert rate limit reached: {self.max_alerts_per_minute}/min")
            return False
        
        # Add to history
        self.alert_history.append({
            'type': alert_type,
            'time': current_time
        })
        
        # Keep only last 100 alerts
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-50:]
        
        return True
    
    def get_alert_statistics(self):
        """Get statistics about alerts"""
        if not self.alert_history:
            return {
                'total': 0,
                'by_type': {},
                'by_severity': {},
                'recent': []
            }
        
        # Count by type
        type_count = {}
        severity_count = {'high': 0, 'medium': 0, 'low': 0}
        
        for alert in self.alert_history:
            alert_type = alert.get('type', 'unknown')
            type_count[alert_type] = type_count.get(alert_type, 0) + 1
            
            # Get severity from rules
            severity = 'low'
            for rule_name, rule in self.safety_rules.items():
                if alert_type in rule_name:
                    severity = rule.get('severity', 'medium')
                    break
            severity_count[severity] = severity_count.get(severity, 0) + 1
        
        # Get recent alerts
        recent = sorted(self.alert_history, 
                       key=lambda x: x.get('time', datetime.min), 
                       reverse=True)[:10]
        
        return {
            'total': len(self.alert_history),
            'by_type': type_count,
            'by_severity': severity_count,
            'recent': recent
        }
    
    def reset(self):
        """Reset the safety engine state"""
        self.fall_counter = 0
        self.alert_history = []
        self.activity_history = []
        self.alert_count = 0
        self.previous_activity = None
        logger.info("Safety engine reset")