"""
Visualization utilities for the monitoring system
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
from datetime import datetime

class Visualizer:
    """Visualization utilities for displaying monitoring results"""
    
    def __init__(self):
        """Initialize visualizer"""
        self.colors = {
            'safe': (0, 255, 0),        # Green
            'warning': (0, 255, 255),   # Yellow
            'danger': (0, 0, 255),      # Red
            'info': (255, 255, 255),    # White
            'bbox': (0, 255, 0),        # Green
            'keypoints': (0, 255, 255), # Yellow
            'text': (255, 255, 255)     # White
        }
        
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def draw_detection(self, frame, detections):
        """
        Draw detection results on frame
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), self.colors['bbox'], 2)
            
            # Draw label
            label = f"{det['class_name']}: {confidence:.2f}"
            (w, h), _ = cv2.getTextSize(label, self.font, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w, y1), self.colors['bbox'], -1)
            cv2.putText(annotated, label, (x1, y1 - 5), self.font, 0.5, self.colors['text'], 1)
        
        return annotated
    
    def draw_pose(self, frame, keypoints, draw_connections=True):
        """
        Draw pose keypoints on frame
        
        Args:
            frame: Input frame
            keypoints: Array of keypoints (33x3)
            draw_connections: Whether to draw connections
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Define connections between keypoints
        connections = [
            (5, 6),  # Left shoulder to right shoulder
            (5, 11), # Left shoulder to left hip
            (6, 12), # Right shoulder to right hip
            (11, 12),# Left hip to right hip
            (5, 7),  # Left shoulder to left elbow
            (7, 9),  # Left elbow to left wrist
            (6, 8),  # Right shoulder to right elbow
            (8, 10), # Right elbow to right wrist
            (11, 13),# Left hip to left knee
            (13, 15),# Left knee to left ankle
            (12, 14),# Right hip to right knee
            (14, 16),# Right knee to right ankle
            (0, 5),  # Nose to left shoulder
            (0, 6)   # Nose to right shoulder
        ]
        
        # Draw connections
        if draw_connections:
            for connection in connections:
                if np.all(keypoints[connection[0]] != 0) and np.all(keypoints[connection[1]] != 0):
                    pt1 = tuple(keypoints[connection[0]][:2].astype(int))
                    pt2 = tuple(keypoints[connection[1]][:2].astype(int))
                    cv2.line(annotated, pt1, pt2, self.colors['keypoints'], 2)
        
        # Draw keypoints
        for kp in keypoints:
            if np.all(kp != 0):
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(annotated, (x, y), 5, self.colors['keypoints'], -1)
        
        return annotated
    
    def draw_safety_info(self, frame, alerts, activity=None, status='safe'):
        """
        Draw safety information on frame
        
        Args:
            frame: Input frame
            alerts: List of alerts
            activity: Current activity
            status: Overall safety status ('safe', 'warning', 'danger')
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        
        # Status color
        status_colors = {
            'safe': self.colors['safe'],
            'warning': self.colors['warning'],
            'danger': self.colors['danger']
        }
        status_color = status_colors.get(status, self.colors['info'])
        
        # Draw status indicator
        status_text = f"Status: {status.upper()}"
        cv2.putText(annotated, status_text, (10, 30), self.font, 0.7, status_color, 2)
        
        # Draw activity
        if activity:
            cv2.putText(annotated, f"Activity: {activity}", (10, 60), self.font, 0.6, self.colors['info'], 2)
        
        # Draw alerts
        y_offset = 90
        for i, alert in enumerate(alerts[-5:]):  # Show last 5 alerts
            alert_color = self.colors['danger'] if alert.get('severity') == 'high' else self.colors['warning']
            message = alert.get('message', 'Unknown alert')
            cv2.putText(annotated, f"⚠ {message}", (10, y_offset + i * 25), 
                       self.font, 0.5, alert_color, 1)
        
        # Draw timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated, timestamp, (w - 250, 30), self.font, 0.5, self.colors['info'], 1)
        
        return annotated
    
    def draw_combined(self, frame, detections, keypoints, alerts, activity=None, status='safe'):
        """
        Draw combined visualization
        
        Args:
            frame: Input frame
            detections: Detection results
            keypoints: Pose keypoints
            alerts: Safety alerts
            activity: Current activity
            status: Overall safety status
            
        Returns:
            Annotated frame
        """
        # Start with original frame
        annotated = frame.copy()
        
        # Draw detections
        annotated = self.draw_detection(annotated, detections)
        
        # Draw pose
        if keypoints is not None:
            annotated = self.draw_pose(annotated, keypoints)
        
        # Draw safety info
        annotated = self.draw_safety_info(annotated, alerts, activity, status)
        
        return annotated
    
    def create_activity_plot(self, activity_history):
        """
        Create a plot of activity history
        
        Args:
            activity_history: List of (timestamp, activity) pairs
            
        Returns:
            Matplotlib figure
        """
        if not activity_history:
            return None
        
        timestamps = [h[0] for h in activity_history]
        activities = [h[1] for h in activity_history]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Convert activities to numeric values for plotting
        unique_activities = list(set(activities))
        activity_to_num = {act: i for i, act in enumerate(unique_activities)}
        numeric_activities = [activity_to_num[act] for act in activities]
        
        ax.plot(timestamps, numeric_activities, 'b-', linewidth=2)
        ax.set_yticks(range(len(unique_activities)))
        ax.set_yticklabels(unique_activities)
        ax.set_xlabel('Time')
        ax.set_ylabel('Activity')
        ax.set_title('Activity History')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_safety_dashboard(self, frame, alerts, activity, status):
        """
        Create a dashboard view with multiple information panels
        
        Args:
            frame: Input frame
            alerts: List of alerts
            activity: Current activity
            status: Overall safety status
            
        Returns:
            Dashboard image
        """
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Create dashboard canvas
        dashboard = np.zeros((h + 100, w, 3), dtype=np.uint8)
        dashboard[:h, :w] = frame
        
        # Add status panel at bottom
        status_colors = {
            'safe': (0, 100, 0),
            'warning': (100, 100, 0),
            'danger': (100, 0, 0)
        }
        status_color = status_colors.get(status, (50, 50, 50))
        dashboard[h:h+100] = status_color
        
        # Draw status text
        cv2.putText(dashboard, f"Status: {status.upper()}", (10, h + 35), 
                   self.font, 0.8, (255, 255, 255), 2)
        
        cv2.putText(dashboard, f"Activity: {activity}", (10, h + 65), 
                   self.font, 0.6, (255, 255, 255), 1)
        
        # Draw alert count
        alert_text = f"Alerts: {len(alerts)}"
        cv2.putText(dashboard, alert_text, (w - 150, h + 35), 
                   self.font, 0.6, (255, 255, 255), 1)
        
        # Draw latest alert if any
        if alerts:
            latest_alert = alerts[-1]
            alert_color = (0, 0, 255) if latest_alert.get('severity') == 'high' else (0, 255, 255)
            cv2.putText(dashboard, f"Latest: {latest_alert['message'][:30]}", 
                       (w - 300, h + 65), self.font, 0.5, alert_color, 1)
        
        return dashboard