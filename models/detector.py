# models/detector.py
"""
Child Detector using YOLOv8
Detects and tracks children/persons in video frames
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChildDetector:
    """
    Child detection using YOLOv8
    Detects persons in video frames and returns bounding boxes
    """
    
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.5, iou_threshold=0.4):
        """
        Initialize the child detector using YOLOv8
        
        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        try:
            self.model = YOLO(model_path)
            logger.info(f"Loaded YOLO model from {model_path}")
            logger.info(f"Using device: {self.device}")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            # Download model if not found
            if 'yolov8n.pt' in model_path:
                logger.info("Downloading YOLOv8n model...")
                self.model = YOLO('yolov8n.pt')
            else:
                raise
        
        self.class_names = self.model.names
        
    def detect(self, frame):
        """
        Detect persons/children in the frame
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            List of detections with bounding boxes and confidence scores
        """
        if frame is None or frame.size == 0:
            return []
        
        try:
            results = self.model(
                frame, 
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            detections = []
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class and confidence
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # Only detect persons (class 0 in COCO)
                        if cls_id == 0:  # person class
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': conf,
                                'class_id': cls_id,
                                'class_name': 'person'
                            })
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in detection: {e}")
            return []
    
    def detect_and_draw(self, frame):
        """
        Detect persons and draw bounding boxes on frame
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            annotated_frame: Frame with bounding boxes drawn
            detections: List of detections
        """
        detections = self.detect(frame)
        annotated_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label background
            label = f"Child {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1-20), (x1 + label_size[0] + 10, y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label, (x1+5, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return annotated_frame, detections
    
    def get_person_center(self, detection):
        """Get center point of a detection"""
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def get_person_area(self, detection):
        """Get area of a detection"""
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)
    
    def filter_detections_by_size(self, detections, min_area=1000, max_area=100000):
        """Filter detections by area size"""
        filtered = []
        for det in detections:
            area = self.get_person_area(det)
            if min_area <= area <= max_area:
                filtered.append(det)
        return filtered