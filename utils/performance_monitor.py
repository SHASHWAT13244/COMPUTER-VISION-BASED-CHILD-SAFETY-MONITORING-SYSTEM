# utils/performance_monitor.py
"""
Performance Monitoring for Child Safety Monitoring System
Tracks system performance metrics in real-time
"""

import time
import psutil
import threading
import logging
import json
from collections import deque, defaultdict
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Performance monitoring for the system
    """
    
    def __init__(self, history_length=100, monitor_interval=2):
        """
        Initialize performance monitor
        
        Args:
            history_length: Number of samples to keep
            monitor_interval: Interval between samples (seconds)
        """
        self.history_length = history_length
        self.monitor_interval = monitor_interval
        
        # Metrics storage - all values are deques for consistent iteration
        self.metrics = {
            'fps': deque(maxlen=history_length),
            'inference_time': deque(maxlen=history_length),
            'cpu_usage': deque(maxlen=history_length),
            'memory_usage': deque(maxlen=history_length),
            'memory_available': deque(maxlen=history_length),
            'temperature': deque(maxlen=history_length),
            'gpu_usage': deque(maxlen=history_length),
            'gpu_memory': deque(maxlen=history_length),
            'disk_usage': deque(maxlen=history_length),
            'network_sent': deque(maxlen=history_length),
            'network_recv': deque(maxlen=history_length)
        }
        
        # Store CPU cores separately (not in metrics dict to avoid iteration issues)
        self.cpu_cores = psutil.cpu_count()
        
        # Custom metrics
        self.custom_metrics = defaultdict(lambda: deque(maxlen=history_length))
        
        # Monitoring state
        self.monitoring = False
        self.monitor_thread = None
        
        # Start time
        self.start_time = datetime.now()
        
        # Network counters
        self.net_io_prev = psutil.net_io_counters()
        
        # Log file
        self.log_file = f"performance_log_{datetime.now().strftime('%Y%m%d')}.json"
        
        logger.info("Performance monitor initialized")
    
    def start_monitoring(self):
        """Start performance monitoring"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
        
        # Save final metrics
        self._save_metrics()
    
    def _monitor_loop(self):
        """Monitor loop running in background thread"""
        while self.monitoring:
            try:
                self._collect_metrics()
                
                # Log metrics periodically
                if len(self.metrics['cpu_usage']) % 10 == 0 and len(self.metrics['cpu_usage']) > 0:
                    self._save_metrics()
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                time.sleep(5)
    
    def _collect_metrics(self):
        """Collect system metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.5)
            self.metrics['cpu_usage'].append(cpu_percent)
            
            # Memory usage
            mem = psutil.virtual_memory()
            self.metrics['memory_usage'].append(mem.percent)
            self.metrics['memory_available'].append(mem.available / (1024**3))  # GB
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.metrics['disk_usage'].append(disk.percent)
            
            # Network
            net_io = psutil.net_io_counters()
            sent_diff = net_io.bytes_sent - self.net_io_prev.bytes_sent
            recv_diff = net_io.bytes_recv - self.net_io_prev.bytes_recv
            self.metrics['network_sent'].append(sent_diff / (1024**2))  # MB
            self.metrics['network_recv'].append(recv_diff / (1024**2))  # MB
            self.net_io_prev = net_io
            
            # Temperature (if available)
            temp = self._get_temperature()
            if temp is not None:
                self.metrics['temperature'].append(temp)
            
            # GPU usage (if available)
            gpu_info = self._get_gpu_info()
            if gpu_info:
                self.metrics['gpu_usage'].append(gpu_info.get('usage', 0))
                self.metrics['gpu_memory'].append(gpu_info.get('memory', 0))
                
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
    
    def _get_temperature(self):
        """Get system temperature"""
        try:
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                if temps:
                    for sensor in temps.values():
                        if sensor:
                            return sensor[0].current
        except:
            pass
        return None
    
    def _get_gpu_info(self):
        """Get GPU information if available"""
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    'usage': torch.cuda.utilization(),
                    'memory': torch.cuda.memory_allocated() / (1024**3)  # GB
                }
        except:
            pass
        return None
    
    def add_metric(self, name, value):
        """Add a custom metric"""
        if name in self.custom_metrics:
            self.custom_metrics[name].append(value)
        else:
            self.custom_metrics[name] = deque([value], maxlen=self.history_length)
    
    def get_metric(self, name):
        """Get metric values"""
        if name in self.metrics:
            return list(self.metrics[name])
        elif name in self.custom_metrics:
            return list(self.custom_metrics[name])
        return None
    
    def get_summary(self):
        """Get performance summary"""
        summary = {}
        
        for name, values in self.metrics.items():
            if values:  # Check if deque is not empty
                values_list = list(values)
                summary[name] = {
                    'current': values_list[-1] if values_list else 0,
                    'average': sum(values_list) / len(values_list) if values_list else 0,
                    'max': max(values_list) if values_list else 0,
                    'min': min(values_list) if values_list else 0,
                    'count': len(values_list)
                }
            else:
                summary[name] = {
                    'current': 0,
                    'average': 0,
                    'max': 0,
                    'min': 0,
                    'count': 0
                }
        
        # Add custom metrics
        for name, values in self.custom_metrics.items():
            if values:
                values_list = list(values)
                summary[name] = {
                    'current': values_list[-1] if values_list else 0,
                    'average': sum(values_list) / len(values_list) if values_list else 0,
                    'max': max(values_list) if values_list else 0,
                    'min': min(values_list) if values_list else 0,
                    'count': len(values_list)
                }
        
        # Add CPU cores info
        summary['cpu_cores'] = {
            'value': self.cpu_cores,
            'current': self.cpu_cores,
            'average': self.cpu_cores,
            'max': self.cpu_cores,
            'min': self.cpu_cores,
            'count': 1
        }
        
        return summary
    
    def get_health_status(self):
        """Get system health status"""
        summary = self.get_summary()
        health = {
            'status': 'healthy',
            'warnings': [],
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
        }
        
        # Check CPU usage
        if 'cpu_usage' in summary and summary['cpu_usage']['count'] > 0:
            cpu_avg = summary['cpu_usage']['average']
            cpu_current = summary['cpu_usage']['current']
            
            if cpu_avg > 80 or cpu_current > 85:
                health['warnings'].append(f"High CPU usage: avg {cpu_avg:.1f}%, current {cpu_current:.1f}%")
                if cpu_avg > 90 or cpu_current > 95:
                    health['status'] = 'critical'
        
        # Check memory usage
        if 'memory_usage' in summary and summary['memory_usage']['count'] > 0:
            mem_avg = summary['memory_usage']['average']
            mem_current = summary['memory_usage']['current']
            
            if mem_avg > 80 or mem_current > 85:
                health['warnings'].append(f"High memory usage: avg {mem_avg:.1f}%, current {mem_current:.1f}%")
                if mem_avg > 90 or mem_current > 95:
                    health['status'] = 'critical'
        
        # Check FPS
        if 'fps' in summary and summary['fps']['count'] > 0:
            fps_avg = summary['fps']['average']
            fps_current = summary['fps']['current']
            
            if fps_avg < 15 and fps_avg > 0:
                health['warnings'].append(f"Low FPS: avg {fps_avg:.1f}, current {fps_current:.1f}")
                if fps_avg < 5:
                    health['status'] = 'critical'
        
        # Check temperature
        if 'temperature' in summary and summary['temperature']['count'] > 0:
            temp_avg = summary['temperature']['average']
            temp_current = summary['temperature']['current']
            
            if temp_avg > 70 or temp_current > 75:
                health['warnings'].append(f"High temperature: avg {temp_avg:.1f}°C, current {temp_current:.1f}°C")
                if temp_avg > 85 or temp_current > 90:
                    health['status'] = 'critical'
        
        # Check if metrics are being collected
        if not any(self.metrics['cpu_usage']):
            health['warnings'].append("No metrics collected")
            if len(health['warnings']) > 0:
                health['status'] = 'warning'
        
        return health
    
    def _save_metrics(self):
        """Save metrics to file"""
        try:
            summary = self.get_summary()
            health = self.get_health_status()
            
            data = {
                'timestamp': datetime.now().isoformat(),
                'summary': summary,
                'health': health,
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
            }
            
            # Append to log file
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    try:
                        existing = json.load(f)
                        if isinstance(existing, list):
                            existing.append(data)
                            data = existing
                    except:
                        data = [data]
            
            with open(self.log_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def get_history(self, metric_name=None, limit=100):
        """Get metric history"""
        if metric_name:
            if metric_name in self.metrics:
                return list(self.metrics[metric_name])[-limit:]
            elif metric_name in self.custom_metrics:
                return list(self.custom_metrics[metric_name])[-limit:]
            return []
        
        # Return all metrics
        result = {}
        for name, values in self.metrics.items():
            result[name] = list(values)[-limit:]
        for name, values in self.custom_metrics.items():
            result[name] = list(values)[-limit:]
        return result
    
    def reset(self):
        """Reset all metrics"""
        for key in self.metrics:
            self.metrics[key].clear()
        for key in self.custom_metrics:
            self.custom_metrics[key].clear()
        self.start_time = datetime.now()
        logger.info("Performance metrics reset")
