"""
Video Processing utilities
"""

import cv2
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Video processing utilities for the monitoring system"""
    
    def __init__(self, source=0, width=640, height=480, fps=30):
        """
        Initialize video processor
        
        Args:
            source: Camera index or video file path
            width: Frame width
            height: Frame height
            fps: Frames per second for recording
        """
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.writer = None
        self.is_recording = False
    
    def open(self):
        """Open video capture"""
        try:
            if isinstance(self.source, int):
                self.cap = cv2.VideoCapture(self.source)
            else:
                self.cap = cv2.VideoCapture(str(self.source))
            
            if not self.cap.isOpened():
                raise ValueError(f"Failed to open video source: {self.source}")
            
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            logger.info(f"Video source opened: {self.source}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to open video source: {e}")
            return False
    
    def read_frame(self):
        """Read a frame from the video source"""
        if self.cap is None:
            raise ValueError("Video capture not opened")
        
        ret, frame = self.cap.read()
        if not ret:
            logger.debug("End of video stream")
            return None
        
        return frame
    
    def start_recording(self, output_path='outputs/monitoring_output.avi'):
        """
        Start recording video output
        
        Args:
            output_path: Path to save the recording
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                self.fps,
                (self.width, self.height)
            )
            
            if not self.writer.isOpened():
                raise ValueError("Failed to open video writer")
            
            self.is_recording = True
            logger.info(f"Recording started: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
    
    def write_frame(self, frame):
        """Write a frame to the recording"""
        if self.is_recording and self.writer is not None:
            self.writer.write(frame)
    
    def stop_recording(self):
        """Stop recording"""
        if self.writer is not None:
            self.writer.release()
            self.is_recording = False
            logger.info("Recording stopped")
    
    def close(self):
        """Close video capture and writer"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            self.is_recording = False
        
        logger.info("Video resources released")
    
    def get_frame_info(self):
        """Get information about the current video stream"""
        if self.cap is None:
            return None
        
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(self.cap.get(cv2.CAP_PROP_FPS)),
            'frame_count': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'is_opened': self.cap.isOpened()
        }
    
    def resize_frame(self, frame, width=None, height=None):
        """Resize a frame"""
        if width is None and height is None:
            return frame
        
        h, w = frame.shape[:2]
        
        if width is None:
            scale = height / h
            new_width = int(w * scale)
            new_height = height
        elif height is None:
            scale = width / w
            new_width = width
            new_height = int(h * scale)
        else:
            new_width = width
            new_height = height
        
        return cv2.resize(frame, (new_width, new_height))