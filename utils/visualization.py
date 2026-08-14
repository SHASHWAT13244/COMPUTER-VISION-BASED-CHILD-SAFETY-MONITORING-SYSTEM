# utils/visualization.py
"""
Visualization utilities for Child Safety Monitoring System
"""

import cv2
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Visualizer:
    """
    Visualizer for drawing annotations on frames
    """
    
    def __init__(self, show_fps=True, show_info=True, show_activity=True):
        """
        Initialize visualizer
        
        Args:
            show_fps: Show FPS counter
            show_info: Show information overlay
            show_activity: Show activity status
        """
        self.show_fps = show_fps
        self.show_info = show_info
        self.show_activity = show_activity
        
        self.fps = 0
        self.frame_count = 0
        self.start_time = None
        
        # Colors
        self.colors = {
            'safe': (0, 255, 0),
            'warning': (0, 255, 255),
            'danger': (0, 0, 255),
            'info': (255, 255, 0),
            'text': (255, 255, 255),
            'background': (0, 0, 0)
        }
        
        # Status icons
        self.status_icons = {
            'safe': '✅',
            'warning': '⚠️',
            'danger': '🚨',
            'info': 'ℹ️'
        }
    
    def draw_status(self, frame, activity, confidence, is_safe=True, alerts=None,
                   extra_info=None):
        """
        Draw status information on frame
        
        Args:
            frame: Input frame
            activity: Current activity
            confidence: Confidence score
            is_safe: Safety status
            alerts: Recent alerts
            extra_info: Additional information
            
        Returns:
            annotated_frame: Frame with annotations
        """
        annotated_frame = frame.copy()
        h, w = annotated_frame.shape[:2]
        
        # Determine status color
        if is_safe:
            status_color = self.colors['safe']
            status_text = "SAFE"
            status_icon = self.status_icons['safe']
        else:
            status_color = self.colors['danger']
            status_text = "UNSAFE"
            status_icon = self.status_icons['danger']
        
        # Draw top status bar
        bar_height = 80 if self.show_info else 50
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_height), self.colors['background'], -1)
        cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
        
        # Status line
        y_offset = 25
        
        # Status and activity
        if self.show_activity:
            status_label = f"{status_icon} {status_text}"
            cv2.putText(annotated_frame, status_label, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            if activity and activity != 'None':
                activity_text = f"Activity: {activity}"
                if confidence > 0:
                    activity_text += f" ({confidence:.2%})"
                cv2.putText(annotated_frame, activity_text, (200, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)
        else:
            cv2.putText(annotated_frame, f"{status_icon} {status_text}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # FPS
        if self.show_fps:
            fps_text = f"FPS: {self.fps:.1f}"
            cv2.putText(annotated_frame, fps_text, (w - 120, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)
        
        # Extra info (second line)
        if self.show_info and extra_info:
            y_offset += 30
            info_text = " | ".join([f"{k}: {v}" for k, v in extra_info.items()])
            cv2.putText(annotated_frame, info_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['info'], 1)
        
        # Alerts (bottom of top bar)
        if alerts and self.show_info:
            y_offset = bar_height - 5
            for i, alert in enumerate(alerts[-3:]):  # Show last 3 alerts
                alert_text = f"⚠ {alert['message']}"
                alert_color = self.colors['danger'] if alert.get('severity') == 'high' else self.colors['warning']
                cv2.putText(annotated_frame, alert_text, (10, y_offset - i * 22),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, alert_color, 1)
        
        # Draw bottom info bar
        if self.show_info:
            cv2.rectangle(annotated_frame, (0, h-25), (w, h), self.colors['background'], -1)
            cv2.addWeighted(annotated_frame, 0.8, annotated_frame, 0.2, 0, annotated_frame)
            
            controls_text = "q: Quit | s: Save | r: Reset | v: Toggle Info"
            cv2.putText(annotated_frame, controls_text, (10, h-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
            
            # Time
            time_text = datetime.now().strftime('%H:%M:%S')
            cv2.putText(annotated_frame, time_text, (w - 80, h-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
        
        return annotated_frame
    
    def draw_bbox_with_label(self, frame, bbox, label, color=None, confidence=None):
        """
        Draw bounding box with label
        
        Args:
            frame: Input frame
            bbox: [x1, y1, x2, y2]
            label: Label text
            color: Color (BGR)
            confidence: Confidence score
            
        Returns:
            annotated_frame: Frame with bounding box
        """
        annotated_frame = frame.copy()
        x1, y1, x2, y2 = bbox
        
        if color is None:
            color = self.colors['info']
        
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # Create label
        if confidence is not None:
            label = f"{label} {confidence:.2f}"
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(annotated_frame, (x1, y1-25), (x1 + label_size[0] + 10, y1), color, -1)
        
        # Draw label text
        cv2.putText(annotated_frame, label, (x1+5, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        return annotated_frame
    
    def draw_pose(self, frame, keypoints, connections=None, color=None, thickness=2):
        """
        Draw pose keypoints and connections
        
        Args:
            frame: Input frame
            keypoints: Keypoints array (33 x 3) with x, y, z
            connections: List of connections to draw
            color: Color for drawing
            thickness: Line thickness
            
        Returns:
            annotated_frame: Frame with pose
        """
        annotated_frame = frame.copy()
        h, w = annotated_frame.shape[:2]
        
        if color is None:
            color = self.colors['info']
        
        if connections is None:
            # Default connections
            connections = [
                (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                (11, 23), (12, 24), (23, 24),
                (23, 25), (25, 27), (27, 29), (29, 31),
                (24, 26), (26, 28), (28, 30), (30, 32),
                (11, 21), (12, 22),
                (15, 17), (15, 19), (15, 21),
                (16, 18), (16, 20), (16, 22)
            ]
        
        # Draw connections
        for idx1, idx2 in connections:
            if idx1 < len(keypoints) and idx2 < len(keypoints):
                p1 = keypoints[idx1]
                p2 = keypoints[idx2]
                
                # Convert from normalized to pixel coordinates
                x1, y1 = int(p1[0] * w), int(p1[1] * h)
                x2, y2 = int(p2[0] * w), int(p2[1] * h)
                
                # Check if points are visible (visibility > 0.5)
                if len(p1) > 3 and p1[3] > 0.5 and len(p2) > 3 and p2[3] > 0.5:
                    cv2.line(annotated_frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw keypoints
        for i, kp in enumerate(keypoints):
            if len(kp) > 3 and kp[3] > 0.5:
                x, y = int(kp[0] * w), int(kp[1] * h)
                cv2.circle(annotated_frame, (x, y), 3, color, -1)
                
                # Draw index for debugging
                # cv2.putText(annotated_frame, str(i), (x+5, y-5),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return annotated_frame
    
    def draw_heatmap(self, frame, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Draw heatmap overlay on frame
        
        Args:
            frame: Input frame
            heatmap: Heatmap array
            alpha: Transparency
            colormap: Colormap type
            
        Returns:
            annotated_frame: Frame with heatmap
        """
        if heatmap is None or heatmap.size == 0:
            return frame
        
        annotated_frame = frame.copy()
        
        # Normalize heatmap
        heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_norm = heatmap_norm.astype(np.uint8)
        
        # Apply colormap
        heatmap_color = cv2.applyColorMap(heatmap_norm, colormap)
        
        # Overlay
        annotated_frame = cv2.addWeighted(annotated_frame, 1 - alpha, heatmap_color, alpha, 0)
        
        return annotated_frame
    
    def create_comparison(self, frames, labels, layout='horizontal'):
        """
        Create comparison view of multiple frames
        
        Args:
            frames: List of frames
            labels: List of labels
            layout: 'horizontal' or 'vertical'
            
        Returns:
            comparison_frame: Combined frame
        """
        if not frames:
            return None
        
        # Resize frames to same dimensions
        max_h = max([f.shape[0] for f in frames])
        max_w = max([f.shape[1] for f in frames])
        
        resized_frames = []
        for f in frames:
            h, w = f.shape[:2]
            if h != max_h or w != max_w:
                f = cv2.resize(f, (max_w, max_h))
            resized_frames.append(f)
        
        if layout == 'horizontal':
            # Stack horizontally
            comparison = np.hstack(resized_frames)
        else:
            # Stack vertically
            comparison = np.vstack(resized_frames)
        
        # Add labels
        if labels:
            y_offset = 20
            for i, (frame, label) in enumerate(zip(resized_frames, labels)):
                if layout == 'horizontal':
                    x_offset = i * max_w + 10
                else:
                    x_offset = 10
                    y_offset = i * max_h + 20
                
                cv2.putText(comparison, label, (x_offset, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return comparison
    
    def update_fps(self, frame_time):
        """Update FPS counter"""
        if self.start_time is None:
            self.start_time = frame_time
            return
        
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            elapsed = (frame_time - self.start_time).total_seconds()
            if elapsed > 0:
                self.fps = 30 / elapsed
                self.start_time = frame_time