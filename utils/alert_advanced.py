# utils/alert_advanced.py
"""
Advanced Alert System with multiple notification methods
Supports Email, SMS, Telegram, and Webhooks
"""

import json
import os
import logging
import requests
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import smtplib
import threading
from typing import Dict, List, Optional, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedAlertSystem:
    """
    Advanced alert system with multiple notification methods
    """
    
    def __init__(self, config_file='alert_config.json'):
        """
        Initialize the advanced alert system
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.config = self._load_config()
        
        self.notification_methods = {
            'email': self._send_email,
            'sms': self._send_sms,
            'telegram': self._send_telegram,
            'webhook': self._send_webhook,
            'desktop': self._send_desktop
        }
        
        self.enabled_methods = self.config.get('enabled_methods', ['desktop'])
        self.enabled_methods = [m for m in self.enabled_methods if m in self.notification_methods]
        
        # Queue for async sending
        self.queue = []
        self.processing = False
        
        logger.info(f"Advanced alert system initialized with methods: {self.enabled_methods}")
    
    def _load_config(self):
        """Load configuration from file"""
        default_config = {
            'enabled_methods': ['desktop'],
            'email': {
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'sender_email': '',
                'sender_password': '',
                'recipient_emails': []
            },
            'sms': {
                'api_key': '',
                'account_sid': '',
                'auth_token': '',
                'from_number': '',
                'phone_numbers': []
            },
            'telegram': {
                'bot_token': '',
                'chat_ids': []
            },
            'webhook': {
                'urls': []
            },
            'desktop': {
                'enable_sound': True,
                'enable_popup': True
            },
            'throttling': {
                'max_alerts_per_minute': 10,
                'cooldown_seconds': 5
            }
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    # Merge with default
                    for key in default_config:
                        if key not in config:
                            config[key] = default_config[key]
                    return config
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        
        return default_config
    
    def send_alert(self, alert_data: Dict[str, Any]) -> bool:
        """
        Send alert through all enabled methods
        
        Args:
            alert_data: Alert information
            
        Returns:
            bool: True if at least one method succeeded
        """
        # Add timestamp
        if 'timestamp' not in alert_data:
            alert_data['timestamp'] = datetime.now().isoformat()
        
        # Check throttling
        if not self._check_throttling(alert_data):
            logger.debug(f"Alert throttled: {alert_data.get('message', '')}")
            return False
        
        # Send through each enabled method
        success_count = 0
        for method in self.enabled_methods:
            try:
                if method in self.notification_methods:
                    result = self.notification_methods[method](alert_data)
                    if result:
                        success_count += 1
            except Exception as e:
                logger.error(f"Error sending alert via {method}: {e}")
        
        return success_count > 0
    
    def _send_email(self, alert_data: Dict[str, Any]) -> bool:
        """Send email alert"""
        email_config = self.config.get('email', {})
        
        if not email_config.get('sender_email') or not email_config.get('sender_password'):
            logger.warning("Email not configured properly")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = email_config['sender_email']
            msg['To'] = ', '.join(email_config.get('recipient_emails', []))
            msg['Subject'] = f"🚨 Child Safety Alert: {alert_data.get('severity', 'Unknown')}"
            
            # Build body
            body = f"""
            <html>
            <body>
            <h2>🚨 Child Safety Alert</h2>
            <table border="1" cellpadding="5">
                <tr><td><strong>Time</strong></td><td>{alert_data.get('timestamp', 'N/A')}</td></tr>
                <tr><td><strong>Message</strong></td><td>{alert_data.get('message', 'N/A')}</td></tr>
                <tr><td><strong>Severity</strong></td><td>{alert_data.get('severity', 'Unknown')}</td></tr>
                <tr><td><strong>Activity</strong></td><td>{alert_data.get('activity', 'Unknown')}</td></tr>
                <tr><td><strong>Confidence</strong></td><td>{alert_data.get('confidence', 0):.2%}</td></tr>
                <tr><td><strong>Location</strong></td><td>{alert_data.get('location', 'Camera feed')}</td></tr>
            </table>
            <hr>
            <p><small>This is an automated alert from Child Safety Monitoring System.</small></p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # Send
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['sender_email'], email_config['sender_password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent: {alert_data.get('message', '')}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    def _send_sms(self, alert_data: Dict[str, Any]) -> bool:
        """Send SMS alert using Twilio"""
        sms_config = self.config.get('sms', {})
        
        if not sms_config.get('account_sid') or not sms_config.get('auth_token'):
            logger.warning("SMS not configured properly")
            return False
        
        try:
            from twilio.rest import Client
            
            client = Client(sms_config['account_sid'], sms_config['auth_token'])
            
            message = f"🚨 Child Safety Alert: {alert_data.get('message', 'Unknown event')}"
            
            for phone in sms_config.get('phone_numbers', []):
                client.messages.create(
                    body=message,
                    from_=sms_config.get('from_number'),
                    to=phone
                )
            
            logger.info(f"SMS alert sent: {alert_data.get('message', '')}")
            return True
            
        except ImportError:
            logger.warning("Twilio not installed. Install with: pip install twilio")
            return False
        except Exception as e:
            logger.error(f"Error sending SMS: {e}")
            return False
    
    def _send_telegram(self, alert_data: Dict[str, Any]) -> bool:
        """Send Telegram message"""
        telegram_config = self.config.get('telegram', {})
        
        if not telegram_config.get('bot_token'):
            logger.warning("Telegram not configured properly")
            return False
        
        try:
            message = f"""
🚨 <b>Child Safety Alert</b> 🚨

<b>Time:</b> {alert_data.get('timestamp', 'N/A')}
<b>Message:</b> {alert_data.get('message', 'N/A')}
<b>Severity:</b> {alert_data.get('severity', 'Unknown')}
<b>Activity:</b> {alert_data.get('activity', 'Unknown')}
<b>Confidence:</b> {alert_data.get('confidence', 0):.2%}
<b>Location:</b> {alert_data.get('location', 'Camera feed')}
            """
            
            url = f"https://api.telegram.org/bot{telegram_config['bot_token']}/sendMessage"
            
            for chat_id in telegram_config.get('chat_ids', []):
                payload = {
                    'chat_id': chat_id,
                    'text': message,
                    'parse_mode': 'HTML'
                }
                response = requests.post(url, data=payload, timeout=5)
                if response.status_code != 200:
                    logger.error(f"Telegram error: {response.text}")
            
            logger.info(f"Telegram alert sent: {alert_data.get('message', '')}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending Telegram: {e}")
            return False
    
    def _send_webhook(self, alert_data: Dict[str, Any]) -> bool:
        """Send webhook alert"""
        webhook_config = self.config.get('webhook', {})
        
        if not webhook_config.get('urls'):
            logger.warning("Webhook not configured")
            return False
        
        try:
            for url in webhook_config['urls']:
                response = requests.post(url, json=alert_data, timeout=5)
                if response.status_code not in [200, 201, 202]:
                    logger.error(f"Webhook error: {response.status_code} - {response.text}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending webhook: {e}")
            return False
    
    def _send_desktop(self, alert_data: Dict[str, Any]) -> bool:
        """Send desktop notification"""
        desktop_config = self.config.get('desktop', {})
        
        try:
            # Console notification
            severity_emoji = {
                'high': '🔴',
                'medium': '🟡',
                'low': '🟢'
            }.get(alert_data.get('severity', 'low'), '⚪')
            
            print('\n' + '='*70)
            print(f"{severity_emoji} ALERT - {alert_data.get('timestamp', 'N/A')}")
            print('='*70)
            print(f"Message: {alert_data.get('message', 'Unknown alert')}")
            print(f"Severity: {alert_data.get('severity', 'unknown').upper()}")
            if alert_data.get('activity'):
                print(f"Activity: {alert_data['activity']}")
            if alert_data.get('location'):
                print(f"Location: {alert_data['location']}")
            print('='*70 + '\n')
            
            # Play sound
            if desktop_config.get('enable_sound', True):
                try:
                    import winsound
                    winsound.Beep(1000, 500)
                    winsound.Beep(1200, 500)
                except ImportError:
                    if os.name == 'nt':
                        import ctypes
                        ctypes.windll.kernel32.Beep(1000, 500)
                        ctypes.windll.kernel32.Beep(1200, 500)
                    else:
                        os.system('printf "\\a\\a"')
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending desktop notification: {e}")
            return False
    
    def _check_throttling(self, alert_data: Dict[str, Any]) -> bool:
        """Check if alert should be throttled"""
        # Implementation depends on storage
        # This is a simplified version
        return True
    
    def reload_config(self):
        """Reload configuration from file"""
        self.config = self._load_config()
        self.enabled_methods = self.config.get('enabled_methods', ['desktop'])
        self.enabled_methods = [m for m in self.enabled_methods if m in self.notification_methods]
        logger.info("Configuration reloaded")
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update configuration"""
        # Merge with existing
        for key, value in new_config.items():
            if isinstance(value, dict) and key in self.config:
                self.config[key].update(value)
            else:
                self.config[key] = value
        
        # Save to file
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            self.reload_config()
        except Exception as e:
            logger.error(f"Error saving config: {e}")