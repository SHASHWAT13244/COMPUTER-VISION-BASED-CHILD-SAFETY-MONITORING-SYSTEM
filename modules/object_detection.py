"""
Object Detection module using YOLOv8
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ObjectDetector:
    """YOLOv8-based object detector for child detection"""
    
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5, device='cpu'):
        """
        Initialize the object detector
        
        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detections
            device: 'cpu' or 'cuda'
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # Load model
        try:
            self.model = YOLO(model_path)
            if device == 'cuda' and torch.cuda.is_available():
                self.model.to('cuda')
                logger.info("Using GPU for YOLO detection")
            else:
                logger.info("Using CPU for YOLO detection")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
        
        self.classes_of_interest = [0]  # COCO class 0 is person
    
    def detect(self, frame):
        """
        Detect persons in a frame
        
        Args:
            frame: Input image (numpy array)
            
        Returns:
            List of detection dictionaries with bounding boxes and confidence
        """
        results = self.model(frame, conf=self.confidence_threshold, classes=self.classes_of_interest)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': self.model.names[class_id]
                    })
        
        return detections
    
    def detect_and_draw(self, frame, draw=True):
        """
        Detect persons and optionally draw results on frame
        
        Args:
            frame: Input image
            draw: Whether to draw bounding boxes
            
        Returns:
            Detections list and annotated frame
        """
        detections = self.detect(frame)
        
        if draw:
            annotated_frame = frame.copy()
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                confidence = det['confidence']
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{det['class_name']}: {confidence:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return detections, annotated_frame
        
        return detections, frame