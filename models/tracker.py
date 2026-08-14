# models/tracker.py
"""
Multi-Person Tracker
Tracks multiple people across frames using IOU matching
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonTracker:
    """
    Multi-person tracker using IOU-based matching
    """
    
    def __init__(self, max_lost_frames=10, min_confidence=0.5, 
                 max_distance=0.5, use_centroid=False):
        """
        Initialize tracker
        
        Args:
            max_lost_frames: Maximum frames to keep track without update
            min_confidence: Minimum confidence for detection
            max_distance: Maximum distance for matching (centroid method)
            use_centroid: Use centroid instead of IOU
        """
        self.tracks = {}
        self.next_id = 0
        self.max_lost_frames = max_lost_frames
        self.min_confidence = min_confidence
        self.max_distance = max_distance
        self.use_centroid = use_centroid
        
        self.frame_count = 0
        self.track_colors = {}
        
    def update(self, detections, frame=None):
        """
        Update tracks with new detections
        
        Args:
            detections: List of detections with bbox and confidence
            frame: Optional frame for visualization
            
        Returns:
            dict: Updated tracks
        """
        self.frame_count += 1
        
        # Filter detections by confidence
        valid_detections = [d for d in detections 
                           if d['confidence'] >= self.min_confidence]
        
        if not valid_detections:
            # Increment lost count for all tracks
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['lost_frames'] += 1
                if self.tracks[track_id]['lost_frames'] > self.max_lost_frames:
                    del self.tracks[track_id]
            return self.tracks
        
        # Extract bboxes
        detection_bboxes = [d['bbox'] for d in valid_detections]
        
        # If no tracks, create new ones
        if not self.tracks:
            for i, bbox in enumerate(detection_bboxes):
                track_id = self._create_track(
                    bbox, 
                    valid_detections[i]['confidence'],
                    frame
                )
            return self.tracks
        
        # Get existing track bboxes
        track_ids = list(self.tracks.keys())
        track_bboxes = [self.tracks[t]['bbox'] for t in track_ids]
        
        # Match detections to tracks
        if self.use_centroid:
            # Use centroid distance
            matches, unmatched_tracks, unmatched_detections = self._match_by_centroid(
                track_bboxes, detection_bboxes, track_ids
            )
        else:
            # Use IOU matching
            matches, unmatched_tracks, unmatched_detections = self._match_by_iou(
                track_bboxes, detection_bboxes, track_ids
            )
        
        # Update matched tracks
        for track_idx, det_idx in matches:
            track_id = track_ids[track_idx]
            self.tracks[track_id]['bbox'] = detection_bboxes[det_idx]
            self.tracks[track_id]['confidence'] = valid_detections[det_idx]['confidence']
            self.tracks[track_id]['lost_frames'] = 0
            self.tracks[track_id]['last_update'] = self.frame_count
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            self._create_track(
                detection_bboxes[det_idx],
                valid_detections[det_idx]['confidence'],
                frame
            )
        
        # Handle unmatched tracks
        for track_idx in unmatched_tracks:
            track_id = track_ids[track_idx]
            self.tracks[track_id]['lost_frames'] += 1
            if self.tracks[track_id]['lost_frames'] > self.max_lost_frames:
                del self.tracks[track_id]
        
        return self.tracks
    
    def _match_by_iou(self, track_bboxes, detection_bboxes, track_ids):
        """Match using IOU"""
        if not track_bboxes or not detection_bboxes:
            return [], list(range(len(track_bboxes))), list(range(len(detection_bboxes)))
        
        # Compute IOU matrix
        iou_matrix = self._compute_iou_matrix(track_bboxes, detection_bboxes)
        
        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        
        # Determine matches
        matches = []
        unmatched_tracks = []
        unmatched_detections = list(range(len(detection_bboxes)))
        
        for i, j in zip(row_ind, col_ind):
            if iou_matrix[i, j] > 0.3:  # IOU threshold
                matches.append((i, j))
                if j in unmatched_detections:
                    unmatched_detections.remove(j)
            else:
                unmatched_tracks.append(i)
        
        # Check for tracks without matches
        matched_tracks = set(i for i, _ in matches)
        for i in range(len(track_bboxes)):
            if i not in matched_tracks:
                unmatched_tracks.append(i)
        
        return matches, unmatched_tracks, unmatched_detections
    
    def _match_by_centroid(self, track_bboxes, detection_bboxes, track_ids):
        """Match using centroid distance"""
        if not track_bboxes or not detection_bboxes:
            return [], list(range(len(track_bboxes))), list(range(len(detection_bboxes)))
        
        # Compute centroids
        track_centroids = []
        for bbox in track_bboxes:
            x1, y1, x2, y2 = bbox
            track_centroids.append(((x1 + x2) / 2, (y1 + y2) / 2))
        
        det_centroids = []
        for bbox in detection_bboxes:
            x1, y1, x2, y2 = bbox
            det_centroids.append(((x1 + x2) / 2, (y1 + y2) / 2))
        
        # Compute distance matrix
        dist_matrix = np.zeros((len(track_centroids), len(det_centroids)))
        for i, tc in enumerate(track_centroids):
            for j, dc in enumerate(det_centroids):
                dist_matrix[i, j] = np.sqrt((tc[0] - dc[0])**2 + (tc[1] - dc[1])**2)
        
        # Normalize distances by image size
        # Assume max distance is 100 pixels
        dist_matrix = dist_matrix / 100.0
        
        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        
        # Determine matches
        matches = []
        unmatched_tracks = []
        unmatched_detections = list(range(len(detection_bboxes)))
        
        for i, j in zip(row_ind, col_ind):
            if dist_matrix[i, j] < self.max_distance:
                matches.append((i, j))
                if j in unmatched_detections:
                    unmatched_detections.remove(j)
            else:
                unmatched_tracks.append(i)
        
        # Check for tracks without matches
        matched_tracks = set(i for i, _ in matches)
        for i in range(len(track_bboxes)):
            if i not in matched_tracks:
                unmatched_tracks.append(i)
        
        return matches, unmatched_tracks, unmatched_detections
    
    def _compute_iou_matrix(self, bboxes1, bboxes2):
        """Compute IOU matrix between two sets of bboxes"""
        iou_matrix = np.zeros((len(bboxes1), len(bboxes2)))
        
        for i, bbox1 in enumerate(bboxes1):
            for j, bbox2 in enumerate(bboxes2):
                iou_matrix[i, j] = self._compute_iou(bbox1, bbox2)
        
        return iou_matrix
    
    def _compute_iou(self, bbox1, bbox2):
        """Compute IOU between two bboxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0
    
    def _create_track(self, bbox, confidence, frame=None):
        """Create a new track"""
        track_id = self.next_id
        self.next_id += 1
        
        # Generate a random color
        color = tuple(np.random.randint(0, 255, 3).tolist())
        self.track_colors[track_id] = color
        
        self.tracks[track_id] = {
            'bbox': bbox,
            'confidence': confidence,
            'lost_frames': 0,
            'created': self.frame_count,
            'last_update': self.frame_count,
            'color': color
        }
        
        return track_id
    
    def draw_tracks(self, frame, draw_id=True, draw_confidence=True,
                    draw_trail=True, trail_length=5):
        """
        Draw tracks on frame
        
        Args:
            frame: Input frame
            draw_id: Draw track ID
            draw_confidence: Draw confidence score
            draw_trail: Draw movement trail
            trail_length: Length of trail
            
        Returns:
            annotated_frame: Frame with tracks drawn
        """
        annotated_frame = frame.copy()
        
        for track_id, track in self.tracks.items():
            bbox = track['bbox']
            x1, y1, x2, y2 = bbox
            color = track.get('color', (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"ID: {track_id}"
            if draw_confidence:
                label += f" {track['confidence']:.2f}"
            
            # Background for label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1-20), 
                         (x1 + label_size[0] + 10, y1), color, -1)
            
            # Label text
            cv2.putText(annotated_frame, label, (x1+5, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw center point
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(annotated_frame, (cx, cy), 3, color, -1)
            
            # Draw trail
            if draw_trail and 'trail' in track:
                trail = track['trail']
                for i in range(1, len(trail)):
                    if i < trail_length:
                        alpha = i / trail_length
                        cv2.line(annotated_frame, trail[i-1], trail[i], 
                                color, max(1, int(alpha * 3)))
        
        return annotated_frame
    
    def get_track_count(self):
        """Get number of active tracks"""
        return len(self.tracks)
    
    def get_track_by_id(self, track_id):
        """Get track by ID"""
        return self.tracks.get(track_id)
    
    def get_all_tracks(self):
        """Get all tracks"""
        return self.tracks
    
    def reset(self):
        """Reset tracker"""
        self.tracks = {}
        self.next_id = 0
        self.track_colors = {}
        logger.info("Tracker reset")