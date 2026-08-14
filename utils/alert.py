# utils/alert.py
"""
Alert System for Child Safety Monitoring
Generates and manages alerts for unsafe events
"""

import datetime
import json
import os
import time
import logging
from collections import defaultdict
from threading import Lock

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertSystem:
    """
    Alert system for generating and managing alerts
    """
    
    def __init__(self, sound_enabled=True, display_enabled=True, log_enabled=True,
                 alert_file='alert_log.json', max_alerts=1000):
        """
        Initialize the alert system
        
        Args:
            sound_enabled: Whether to play sound
            display_enabled: Whether to display alerts
            log_enabled: Whether to log alerts
            alert_file: Path to log file
            max_alerts: Maximum alerts to keep in memory
        """
        self.sound_enabled = sound_enabled
        self.display_enabled = display_enabled
        self.log_enabled = log_enabled
        self.alert_file = alert_file
        self.max_alerts = max_alerts
        
        self.alert_log = []
        self.alert_count = 0
        self.lock = Lock()
        
        # Statistics
        self.stats = {
            'total': 0,
            'by_severity': defaultdict(int),
            'by_activity': defaultdict(int),
            'by_hour': defaultdict(int),
            'last_alert': None,
            'first_alert': None
        }
        
        # Load existing alerts
        self._load_alerts()
        
        # Recent alerts for throttling
        self.recent_alerts = []
        self.throttle_window = 60  # seconds
        self.max_per_window = 10
        
        logger.info("Alert system initialized")
    
    def generate_alert(self, message, severity='high', activity=None, bbox=None,
                       location=None, image=None):
        """
        Generate an alert
        
        Args:
            message: Alert message
            severity: 'high', 'medium', or 'low'
            activity: Activity that triggered the alert
            bbox: Bounding box of the person
            location: Location of the event
            image: Optional image frame
            
        Returns:
            dict: Alert information
        """
        # Check throttling
        if not self._check_throttle():
            logger.debug("Alert throttled")
            return None
        
        timestamp = datetime.datetime.now()
        
        alert_info = {
            'id': self.alert_count + 1,
            'timestamp': timestamp.isoformat(),
            'datetime': timestamp,
            'message': message,
            'severity': severity,
            'activity': activity,
            'location': location,
            'bbox': bbox,
            'image_saved': False
        }
        
        self.alert_count += 1
        
        # Save image if provided
        if image is not None and bbox is not None:
            try:
                self._save_alert_image(image, bbox, alert_info['id'])
                alert_info['image_saved'] = True
            except Exception as e:
                logger.error(f"Error saving alert image: {e}")
        
        # Display alert
        if self.display_enabled:
            self._display_alert(alert_info)
        
        # Play sound
        if self.sound_enabled:
            self._play_sound()
        
        # Log alert
        if self.log_enabled:
            self._log_alert(alert_info)
        
        # Update statistics
        with self.lock:
            self.stats['total'] += 1
            self.stats['by_severity'][severity] += 1
            if activity:
                self.stats['by_activity'][activity] += 1
            
            hour = timestamp.hour
            self.stats['by_hour'][hour] += 1
            
            if self.stats['first_alert'] is None:
                self.stats['first_alert'] = timestamp
            self.stats['last_alert'] = timestamp
        
        # Add to recent alerts
        self.recent_alerts.append({
            'time': timestamp,
            'type': activity or 'unknown'
        })
        
        # Trim recent alerts
        cutoff = timestamp - datetime.timedelta(seconds=self.throttle_window)
        self.recent_alerts = [a for a in self.recent_alerts if a['time'] > cutoff]
        
        # Store alert
        with self.lock:
            self.alert_log.append(alert_info)
            if len(self.alert_log) > self.max_alerts:
                self.alert_log = self.alert_log[-self.max_alerts:]
        
        logger.info(f"Alert generated: {message} (ID: {alert_info['id']})")
        
        return alert_info
    
    def _display_alert(self, alert_info):
        """Display alert in console"""
        severity_emoji = {
            'high': '🔴',
            'medium': '🟡',
            'low': '🟢'
        }.get(alert_info['severity'], '⚪')
        
        print('\n' + '='*70)
        print(f"{severity_emoji} ALERT #{alert_info['id']} - {alert_info['timestamp']}")
        print('='*70)
        print(f"Message: {alert_info['message']}")
        print(f"Severity: {alert_info['severity'].upper()}")
        if alert_info['activity']:
            print(f"Activity: {alert_info['activity']}")
        if alert_info['location']:
            print(f"Location: {alert_info['location']}")
        if alert_info['bbox']:
            print(f"BBox: {alert_info['bbox']}")
        if alert_info['image_saved']:
            print(f"Image: Saved")
        print('='*70 + '\n')
    
    def _play_sound(self):
        """Play alert sound"""
        try:
            # Try different methods for sound
            try:
                import winsound
                winsound.Beep(1000, 300)
                time.sleep(0.2)
                winsound.Beep(1200, 300)
            except ImportError:
                # Try using system command
                if os.name == 'posix':
                    os.system('printf "\\a"')
                else:
                    import ctypes
                    ctypes.windll.kernel32.Beep(1000, 300)
                    time.sleep(0.2)
                    ctypes.windll.kernel32.Beep(1200, 300)
        except Exception as e:
            logger.debug(f"Could not play sound: {e}")
    
    def _log_alert(self, alert_info):
        """Log alert to file"""
        try:
            # Load existing log
            log_data = []
            if os.path.exists(self.alert_file):
                try:
                    with open(self.alert_file, 'r') as f:
                        log_data = json.load(f)
                except:
                    pass
            
            # Add new alert
            alert_copy = alert_info.copy()
            alert_copy['datetime'] = alert_copy['datetime'].isoformat()
            log_data.append(alert_copy)
            
            # Save
            with open(self.alert_file, 'w') as f:
                json.dump(log_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error logging alert: {e}")
    
    def _save_alert_image(self, image, bbox, alert_id):
        """Save alert image with bounding box"""
        try:
            import cv2
            
            # Create directory
            alert_dir = 'alerts'
            os.makedirs(alert_dir, exist_ok=True)
            
            # Crop and save
            x1, y1, x2, y2 = bbox
            cropped = image[y1:y2, x1:x2]
            
            if cropped.size > 0:
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{alert_dir}/alert_{alert_id}_{timestamp}.jpg"
                cv2.imwrite(filename, cropped)
                logger.debug(f"Alert image saved: {filename}")
            
        except Exception as e:
            logger.error(f"Error saving alert image: {e}")
    
    def _check_throttle(self):
        """Check if alert should be throttled"""
        now = datetime.datetime.now()
        cutoff = now - datetime.timedelta(seconds=self.throttle_window)
        
        # Count recent alerts
        recent = [a for a in self.recent_alerts if a['time'] > cutoff]
        
        if len(recent) >= self.max_per_window:
            return False
        
        # Check if too many of same type
        if recent:
            # Count alerts of same type in last 10 seconds
            recent_cutoff = now - datetime.timedelta(seconds=10)
            same_type = [a for a in recent if a['time'] > recent_cutoff]
            
            if len(same_type) >= 3:
                return False
        
        return True
    
    def _load_alerts(self):
        """Load alerts from file"""
        try:
            if os.path.exists(self.alert_file):
                with open(self.alert_file, 'r') as f:
                    log_data = json.load(f)
                    for alert in log_data[-self.max_alerts:]:
                        self.alert_log.append(alert)
                        self.alert_count = max(self.alert_count, alert.get('id', 0))
        except Exception as e:
            logger.error(f"Error loading alerts: {e}")
    
    def get_recent_alerts(self, count=10):
        """Get recent alerts"""
        return self.alert_log[-count:] if self.alert_log else []
    
    def get_alert_by_id(self, alert_id):
        """Get alert by ID"""
        for alert in self.alert_log:
            if alert.get('id') == alert_id:
                return alert
        return None
    
    def get_alert_statistics(self):
        """Get alert statistics"""
        return {
            'total': self.stats['total'],
            'by_severity': dict(self.stats['by_severity']),
            'by_activity': dict(self.stats['by_activity']),
            'by_hour': dict(self.stats['by_hour']),
            'first_alert': self.stats['first_alert'].isoformat() if self.stats['first_alert'] else None,
            'last_alert': self.stats['last_alert'].isoformat() if self.stats['last_alert'] else None
        }
    
    def clear_alerts(self):
        """Clear all alerts"""
        with self.lock:
            self.alert_log = []
            self.alert_count = 0
            self.stats = {
                'total': 0,
                'by_severity': defaultdict(int),
                'by_activity': defaultdict(int),
                'by_hour': defaultdict(int),
                'last_alert': None,
                'first_alert': None
            }
            self.recent_alerts = []
        
        logger.info("All alerts cleared")
    
    def export_alerts(self, format='json'):
        """Export alerts"""
        if format == 'json':
            return json.dumps(self.alert_log, indent=2, default=str)
        elif format == 'csv':
            import csv
            import io
            
            output = io.StringIO()
            if self.alert_log:
                fieldnames = ['id', 'timestamp', 'severity', 'activity', 'message', 'location']
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                for alert in self.alert_log:
                    row = {k: alert.get(k, '') for k in fieldnames}
                    writer.writerow(row)
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")