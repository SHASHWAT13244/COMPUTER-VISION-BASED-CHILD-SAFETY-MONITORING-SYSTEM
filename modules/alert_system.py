"""
Alert System module for notifications
"""

import time
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class AlertSystem:
    """Alert system for sending notifications"""
    
    def __init__(self, config=None):
        """
        Initialize alert system
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.enable_console = self.config.get('enable_console', True)
        self.enable_email = self.config.get('enable_email', False)
        self.enable_sms = self.config.get('enable_sms', False)
        
        # Email config
        self.email_recipient = self.config.get('email_recipient', '')
        self.email_sender = self.config.get('email_sender', '')
        self.email_password = self.config.get('email_password', '')
        self.smtp_server = self.config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = self.config.get('smtp_port', 587)
        
        # SMS config (Twilio)
        self.twilio_account_sid = self.config.get('twilio_account_sid', '')
        self.twilio_auth_token = self.config.get('twilio_auth_token', '')
        self.twilio_from_number = self.config.get('twilio_from_number', '')
        self.sms_recipient = self.config.get('sms_recipient', '')
        
        # Alert history
        self.alert_history = []
        self.last_alert_time = {}
        
        # Log file for alerts
        self.log_file = Path('outputs/alerts.jsonl')
        self.log_file.parent.mkdir(exist_ok=True)
    
    def send_alert(self, alert_type, message, severity='medium', metadata=None):
        """
        Send alert through configured channels
        
        Args:
            alert_type: Type of alert (e.g., 'fall', 'running')
            message: Alert message
            severity: 'low', 'medium', 'high'
            metadata: Additional metadata
            
        Returns:
            Boolean indicating success
        """
        alert_data = {
            'type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        # Check cooldown for this alert type
        if alert_type in self.last_alert_time:
            cooldown = self.config.get('cooldown_seconds', 10)
            if time.time() - self.last_alert_time[alert_type] < cooldown:
                logger.debug(f"Alert cooldown active for {alert_type}")
                return False
        
        # Update last alert time
        self.last_alert_time[alert_type] = time.time()
        
        # Add to history
        self.alert_history.append(alert_data)
        
        # Log alert
        self._log_alert(alert_data)
        
        # Send through channels
        success = True
        
        if self.enable_console:
            self._console_alert(alert_data)
        
        if self.enable_email:
            email_success = self._email_alert(alert_data)
            success = success and email_success
        
        if self.enable_sms:
            sms_success = self._sms_alert(alert_data)
            success = success and sms_success
        
        # Also save to file
        self._save_alert_to_file(alert_data)
        
        return success
    
    def _console_alert(self, alert_data):
        """Display alert in console"""
        severity_colors = {
            'low': '\033[92m',    # Green
            'medium': '\033[93m', # Yellow
            'high': '\033[91m'    # Red
        }
        reset_color = '\033[0m'
        
        color = severity_colors.get(alert_data['severity'], '\033[94m')
        
        print(f"{color}🚨 ALERT [{alert_data['severity'].upper()}]: {alert_data['type']}")
        print(f"   {alert_data['message']}")
        print(f"   Time: {alert_data['timestamp']}{reset_color}")
        print("-" * 50)
    
    def _email_alert(self, alert_data):
        """Send email alert"""
        if not all([self.email_sender, self.email_password, self.email_recipient]):
            logger.warning("Email credentials not configured")
            return False
        
        try:
            # Create email
            msg = MIMEMultipart()
            msg['From'] = self.email_sender
            msg['To'] = self.email_recipient
            msg['Subject'] = f"🚨 Safety Alert: {alert_data['type'].upper()}"
            
            # Email body
            body = f"""
            Safety Alert Notification
            
            Alert Type: {alert_data['type']}
            Severity: {alert_data['severity'].upper()}
            Message: {alert_data['message']}
            Timestamp: {alert_data['timestamp']}
            
            Please check the monitoring system immediately.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_sender, self.email_password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent to {self.email_recipient}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def _sms_alert(self, alert_data):
        """Send SMS alert using Twilio"""
        if not all([self.twilio_account_sid, self.twilio_auth_token, 
                   self.twilio_from_number, self.sms_recipient]):
            logger.warning("SMS credentials not configured")
            return False
        
        try:
            from twilio.rest import Client
            
            client = Client(self.twilio_account_sid, self.twilio_auth_token)
            
            message = f"ALERT: {alert_data['type'].upper()} - {alert_data['message']}"
            
            client.messages.create(
                body=message,
                from_=self.twilio_from_number,
                to=self.sms_recipient
            )
            
            logger.info(f"SMS alert sent to {self.sms_recipient}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send SMS alert: {e}")
            return False
    
    def _log_alert(self, alert_data):
        """Log alert to console"""
        log_message = f"[{alert_data['timestamp']}] {alert_data['severity'].upper()}: {alert_data['type']} - {alert_data['message']}"
        logger.warning(log_message)
    
    def _save_alert_to_file(self, alert_data):
        """Save alert to JSON file"""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(alert_data) + '\n')
        except Exception as e:
            logger.error(f"Failed to save alert to file: {e}")
    
    def get_alert_history(self, limit=None):
        """
        Get alert history
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of alerts
        """
        if limit:
            return self.alert_history[-limit:]
        return self.alert_history
    
    def clear_history(self):
        """Clear alert history"""
        self.alert_history = []
        self.last_alert_time = {}
    
    def send_test_alert(self):
        """Send a test alert to verify configuration"""
        return self.send_alert(
            alert_type='test',
            message='This is a test alert from the Child Safety Monitoring System',
            severity='medium'
        )