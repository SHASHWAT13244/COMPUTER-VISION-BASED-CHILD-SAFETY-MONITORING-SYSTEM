"""
Video Processing utilities for the monitoring system
"""

import cv2
import numpy as np
from pathlib import Path
import logging
import threading
from queue import Queue
import time

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Video processing utilities with threading support"""
    
    def __init__(self, source=0, width=640, height=480, fps=30, buffer_size=30):
        """
        Initialize video processor
        
        Args:
            source: Camera index or video file path
            width: Frame width
            height: Frame height
            fps: Frames per second for recording
            buffer_size: Size of frame buffer for threaded processing
        """
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size
        
        self.cap = None
        self.writer = None
        self.is_recording = False
        self.is_running = False
        
        # Threading components
        self.frame_buffer = Queue(maxsize=buffer_size)
        self.thread = None
        self.stop_event = threading.Event()
        
        # Statistics
        self.frame_count = 0
        self.fps_tracker = []
        self.processing_time = []
        
        logger.info(f"VideoProcessor initialized with source: {source}")
    
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
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Get actual properties
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Video source opened: {self.source} ({self.width}x{self.height} @ {self.actual_fps}fps)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to open video source: {e}")
            return False
    
    def start_threaded_capture(self):
        """Start threaded frame capture"""
        if self.thread is not None:
            logger.warning("Threaded capture already running")
            return
        
        self.is_running = True
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info("Threaded capture started")
    
    def _capture_loop(self):
        """Background thread for frame capture"""
        while not self.stop_event.is_set():
            if self.cap is None:
                break
            
            ret, frame = self.cap.read()
            if not ret:
                # Wait and retry for video files
                time.sleep(0.01)
                continue
            
            # Clear buffer if it's getting too full
            if self.frame_buffer.qsize() >= self.buffer_size:
                try:
                    self.frame_buffer.get_nowait()
                except:
                    pass
            
            # Add frame to buffer
            try:
                self.frame_buffer.put_nowait(frame)
            except:
                pass
            
            self.frame_count += 1
            
            # Check for video end
            if isinstance(self.source, str):
                if self.cap.get(cv2.CAP_PROP_POS_FRAMES) >= self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    break
        
        logger.info("Capture thread stopped")
    
    def read_frame(self):
        """
        Read a frame from the video source
        
        Returns:
            Frame or None if no frame available
        """
        if self.thread is not None:
            # Threaded mode
            try:
                return self.frame_buffer.get_nowait()
            except:
                return None
        else:
            # Non-threaded mode
            if self.cap is None:
                raise ValueError("Video capture not opened")
            
            ret, frame = self.cap.read()
            if not ret:
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
            self.recording_path = output_path
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
            self.writer = None
            self.is_recording = False
            logger.info(f"Recording stopped: {self.recording_path}")
    
    def close(self):
        """Close video capture and writer"""
        # Stop threaded capture
        if self.thread is not None:
            self.stop_event.set()
            self.thread.join(timeout=2)
            self.thread = None
        
        # Release capture
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Release writer
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        
        self.is_running = False
        logger.info("Video resources released")
    
    def get_frame_info(self):
        """Get information about the current video stream"""
        if self.cap is None:
            return None
        
        info = {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(self.cap.get(cv2.CAP_PROP_FPS)),
            'frame_count': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'is_opened': self.cap.isOpened(),
            'buffer_size': self.frame_buffer.qsize(),
            'is_recording': self.is_recording
        }
        
        if self.is_recording:
            info['recording_path'] = str(self.recording_path)
        
        return info
    
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
    
    def rotate_frame(self, frame, angle):
        """Rotate a frame"""
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(frame, matrix, (w, h))
    
    def add_overlay(self, frame, text, position=(10, 30), color=(255, 255, 255), 
                    font_scale=0.7, thickness=2, background=True):
        """Add text overlay to frame"""
        if background:
            # Add background rectangle
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                                  font_scale, thickness)
            cv2.rectangle(frame, 
                         (position[0] - 5, position[1] - text_h - 5),
                         (position[0] + text_w + 5, position[1] + 5),
                         (0, 0, 0), -1)
        
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, color, thickness)
        
        return frame