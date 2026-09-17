# utils/alert_advanced.py
"""
Advanced multi-channel alert system.

Supported channels
------------------
    desktop   – local popup (delegated to AlertSystem)
    email     – SMTP (TLS / SSL / plain)
    telegram  – Telegram Bot API
    sms       – HTTP-based SMS provider (generic, pluggable)
    webhook   – POST JSON to an external URL
    log       – local JSONL file (always available)
"""

import os
import json
import time
import logging
import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

from .alert import AlertSystem

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import requests
    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False
    logger.warning("requests not installed — email/telegram/sms/webhook "
                   "channels will be disabled")


# =====================================================================
#  DEFAULT CONFIG
# =====================================================================
DEFAULT_CONFIG = {
    "enabled_methods": ["desktop", "log"],

    "email": {
        "smtp_server":      "smtp.gmail.com",
        "smtp_port":        587,
        "smtp_security":    "tls",           # tls | ssl | none
        "sender_email":     "",
        "sender_password":  "",              # app password, NOT account password
        "recipient_emails": [],
        "use_ssl_fallback": True,
    },

    "telegram": {
        "bot_token":  "",
        "chat_ids":   [],
        "parse_mode": "Markdown",
    },

    "sms": {
        # Generic HTTP provider. Point "endpoint" at your SMS gateway.
        # Common options:
        #   Twilio:    https://api.twilio.com/2010-04-01/Accounts/{SID}/Messages.json
        #   Fast2SMS:  https://www.fast2sms.com/dev/bulkV2
        #   MSG91:     https://api.msg91.com/api/v5/flow/
        "provider":   "generic",             # generic | twilio | fast2sms | msg91
        "endpoint":   "",
        "method":     "POST",                # POST | GET
        "auth_header": {
            "name":  "Authorization",
            "value": "",                     # e.g. "Bearer xxxx" or "key xxxx"
        },
        "extra_headers": {},
        "extra_params":  {},
        "recipients":    [],                 # ["+919999999999", ...]
        "sender_id":     "",
        "message_field": "message",          # key expected by the gateway
        "to_field":      "to",
    },

    "webhook": {
        "urls":     [],                      # one or more URLs
        "method":   "POST",
        "headers":  {"Content-Type": "application/json"},
        "timeout":  5,
    },

    "throttling": {
        "cooldown_seconds":         5,
        "max_alerts_per_minute":    10,
        "duplicate_window_seconds": 30,
    },
}


# =====================================================================
#  HELPERS
# =====================================================================
def _merge_defaults(base, override):
    """Recursively merge `override` on top of `base` without mutating base."""
    result = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _merge_defaults(result[k], v)
        else:
            result[k] = v
    return result


# =====================================================================
#  MAIN CLASS
# =====================================================================
class AdvancedAlertSystem:
    """
    Multi-channel alert dispatcher with per-channel enable flags,
    throttling, deduplication and per-channel test methods.
    """

    def __init__(self, config_path='alert_config.json'):
        self.config_path = config_path
        self.config      = self._load_config()

        # Underlying desktop + log handler
        self.basic = AlertSystem(
            sound_enabled=True,
            display_enabled=True,
            log_enabled=True,
            cooldown_seconds=0,       # throttling handled here
            max_alerts_per_minute=10_000,
        )

        # Throttling state
        self._lock              = threading.Lock()
        self._last_alert_time   = 0.0
        self._recent_signatures = {}    # {signature: timestamp}

    # ------------------------------------------------------------------
    def _load_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    user_cfg = json.load(f)
                cfg = _merge_defaults(DEFAULT_CONFIG, user_cfg)
                logger.info(f"Loaded alert config from {self.config_path}")
                return cfg
            except Exception as e:
                logger.error(f"Failed to read {self.config_path}: {e}. "
                             f"Using defaults.")
        else:
            logger.info(f"{self.config_path} not found. Using defaults. "
                        f"Copy alert_config.example.json to configure.")
        return dict(DEFAULT_CONFIG)

    # ------------------------------------------------------------------
    def reload_config(self):
        """Re-read alert_config.json from disk."""
        self.config = self._load_config()

    # ------------------------------------------------------------------
    def save_config(self):
        """Persist current config to disk."""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Saved alert config to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save {self.config_path}: {e}")

    # ------------------------------------------------------------------
    def _is_throttled(self, alert):
        now = time.time()
        cfg = self.config['throttling']

        # Global cooldown
        if now - self._last_alert_time < cfg['cooldown_seconds']:
            return True, "cooldown"

        # Rate limit
        minute_cut = now - 60
        with self._lock:
            self._recent_signatures = {
                s: t for s, t in self._recent_signatures.items()
                if t >= minute_cut
            }
            if len(self._recent_signatures) >= cfg['max_alerts_per_minute']:
                return True, "rate_limit"

            # Duplicate suppression
            sig = f"{alert.get('activity')}|{alert.get('message')}"
            last = self._recent_signatures.get(sig, 0)
            if now - last < cfg['duplicate_window_seconds']:
                return True, "duplicate"

            self._recent_signatures[sig] = now
        return False, None

    # ------------------------------------------------------------------
    def send_alert(self, alert_info):
        """
        Dispatch `alert_info` to every enabled channel.

        Returns a dict {channel: True|False|'skipped'}.
        """
        if not isinstance(alert_info, dict):
            return {'error': 'alert_info must be a dict'}

        throttled, reason = self._is_throttled(alert_info)
        if throttled:
            logger.debug(f"Alert throttled ({reason})")
            return {'throttled': reason}

        with self._lock:
            self._last_alert_time = time.time()

        methods = self.config.get('enabled_methods', ['desktop', 'log'])
        results = {}

        for method in methods:
            try:
                if method == 'desktop':
                    results['desktop'] = self._send_desktop(alert_info)
                elif method == 'email':
                    results['email'] = self._send_email(alert_info)
                elif method == 'telegram':
                    results['telegram'] = self._send_telegram(alert_info)
                elif method == 'sms':
                    results['sms'] = self._send_sms(alert_info)
                elif method == 'webhook':
                    results['webhook'] = self._send_webhook(alert_info)
                elif method == 'log':
                    results['log'] = self._send_log(alert_info)
                else:
                    results[method] = 'unknown_method'
            except Exception as e:
                logger.error(f"Channel '{method}' failed: {e}")
                results[method] = False

        logger.info(f"Alert dispatched → {results}")
        return results

    # ================================================================
    #  CHANNEL: DESKTOP
    # ================================================================
    def _send_desktop(self, alert):
        try:
            self.basic._show_desktop(alert)
            self.basic._play_sound()
            return True
        except Exception as e:
            logger.error(f"Desktop alert failed: {e}")
            return False

    # ================================================================
    #  CHANNEL: LOG
    # ================================================================
    def _send_log(self, alert):
        try:
            self.basic._log_alert(alert)
            return True
        except Exception as e:
            logger.error(f"Log alert failed: {e}")
            return False

    # ================================================================
    #  CHANNEL: EMAIL
    # ================================================================
    def _send_email(self, alert):
        if not HAS_REQUESTS and False:
            pass  # email uses stdlib, no requests needed
        cfg = self.config['email']

        smtp_server = cfg.get('smtp_server')
        port        = int(cfg.get('smtp_port', 587))
        security    = (cfg.get('smtp_security') or 'tls').lower()
        sender      = cfg.get('sender_email')
        password    = cfg.get('sender_password')
        recipients  = cfg.get('recipient_emails') or []

        if not (smtp_server and sender and password and recipients):
            logger.warning("Email channel skipped — configuration incomplete")
            return 'skipped'

        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = self._email_subject(alert)
            msg['From']    = sender
            msg['To']      = ', '.join(recipients)

            text = self._plain_body(alert)
            html = self._html_body(alert)

            msg.attach(MIMEText(text, 'plain', 'utf-8'))
            msg.attach(MIMEText(html,  'html',  'utf-8'))

            if security == 'ssl':
                server = smtplib.SMTP_SSL(smtp_server, port, timeout=10)
            else:
                server = smtplib.SMTP(smtp_server, port, timeout=10)
                server.ehlo()
                if security == 'tls':
                    server.starttls()
                    server.ehlo()

            server.login(sender, password)
            server.sendmail(sender, recipients, msg.as_string())
            server.quit()
            logger.info(f"📧 Email sent to {recipients}")
            return True

        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"Email auth failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return False

    def _email_subject(self, alert):
        sev = (alert.get('severity') or 'medium').upper()
        return f"[{sev}] Child Safety Alert — {alert.get('activity','event')}"

    def _plain_body(self, alert):
        return (
            f"Child Safety Alert\n"
            f"{'=' * 40}\n\n"
            f"Time:      {alert.get('timestamp','')}\n"
            f"Activity:  {alert.get('activity','unknown')}\n"
            f"Severity:  {alert.get('severity','medium')}\n"
            f"Message:   {alert.get('message','')}\n"
            f"Source:    {alert.get('source','realtime')}\n\n"
            f"This alert was generated automatically by the\n"
            f"Child Safety Monitoring System.\n"
        )

    def _html_body(self, alert):
        sev = (alert.get('severity') or 'medium').lower()
        color = {'high': '#dc2626', 'medium': '#f59e0b'}.get(sev, '#64748b')
        return f"""\
<html><body style="font-family:Arial,sans-serif;color:#0f172a">
  <div style="max-width:560px;margin:auto;border:1px solid #e2e8f0;
              border-radius:10px;overflow:hidden">
    <div style="background:{color};color:#fff;padding:16px 20px">
      <h2 style="margin:0;font-size:1.15rem">
        Child Safety Alert — {alert.get('activity','event').capitalize()}
      </h2>
    </div>
    <table style="width:100%;border-collapse:collapse">
      <tr><td style="padding:10px 20px;color:#64748b;width:110px">Time</td>
          <td style="padding:10px 20px">{alert.get('timestamp','')}</td></tr>
      <tr><td style="padding:10px 20px;color:#64748b">Activity</td>
          <td style="padding:10px 20px">{alert.get('activity','unknown')}</td></tr>
      <tr><td style="padding:10px 20px;color:#64748b">Severity</td>
          <td style="padding:10px 20px">{sev.upper()}</td></tr>
      <tr><td style="padding:10px 20px;color:#64748b">Message</td>
          <td style="padding:10px 20px">{alert.get('message','')}</td></tr>
      <tr><td style="padding:10px 20px;color:#64748b">Source</td>
          <td style="padding:10px 20px">{alert.get('source','realtime')}</td></tr>
    </table>
    <div style="padding:12px 20px;background:#f1f5f9;font-size:.8rem;
                color:#64748b">
      Child Safety Monitoring System — automated notification
    </div>
  </div>
</body></html>"""

    # ================================================================
    #  CHANNEL: TELEGRAM
    # ================================================================
    def _send_telegram(self, alert):
        if not HAS_REQUESTS:
            return 'skipped'

        cfg   = self.config['telegram']
        token = cfg.get('bot_token')
        chats = cfg.get('chat_ids') or []

        if not (token and chats):
            logger.warning("Telegram channel skipped — configuration incomplete")
            return 'skipped'

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        text = self._telegram_body(alert)

        sent_ok = False
        for chat_id in chats:
            try:
                r = requests.post(
                    url,
                    data={
                        'chat_id':    chat_id,
                        'text':       text,
                        'parse_mode': cfg.get('parse_mode', 'Markdown'),
                        'disable_web_page_preview': True,
                    },
                    timeout=8,
                )
                if r.status_code == 200 and r.json().get('ok'):
                    logger.info(f"📨 Telegram sent to chat {chat_id}")
                    sent_ok = True
                else:
                    logger.error(f"Telegram error for {chat_id}: "
                                 f"{r.status_code} {r.text[:200]}")
            except Exception as e:
                logger.error(f"Telegram send failed for {chat_id}: {e}")

        return sent_ok

    def _telegram_body(self, alert):
        emoji = {'high': '🚨', 'medium': '⚠️'}.get(
            (alert.get('severity') or '').lower(), 'ℹ️')
        return (
            f"{emoji} *Child Safety Alert*\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"*Time*:     `{alert.get('timestamp','')}`\n"
            f"*Activity*: {alert.get('activity','unknown')}\n"
            f"*Severity*: {alert.get('severity','medium')}\n"
            f"*Message*:  {alert.get('message','')}\n"
            f"_Automated notification_"
        )

    # ================================================================
    #  CHANNEL: SMS
    # ================================================================
    def _send_sms(self, alert):
        if not HAS_REQUESTS:
            return 'skipped'

        cfg        = self.config['sms']
        endpoint   = cfg.get('endpoint')
        recipients = cfg.get('recipients') or []
        message    = self._sms_body(alert)

        if not (endpoint and recipients):
            logger.warning("SMS channel skipped — configuration incomplete")
            return 'skipped'

        provider = (cfg.get('provider') or 'generic').lower()
        method   = (cfg.get('method') or 'POST').upper()
        headers  = dict(cfg.get('extra_headers') or {})

        # Auth header (if configured)
        auth = cfg.get('auth_header') or {}
        if auth.get('name') and auth.get('value'):
            headers[auth['name']] = auth['value']

        ok_any = False
        for to_number in recipients:
            try:
                if provider == 'twilio':
                    payload = {
                        'To':   to_number,
                        'From': cfg.get('sender_id', ''),
                        'Body': message,
                    }
                    # Twilio expects form-encoded
                    headers.setdefault('Content-Type',
                                       'application/x-www-form-urlencoded')
                    if method == 'POST':
                        r = requests.post(endpoint, data=payload,
                                          headers=headers, timeout=8)
                    else:
                        r = requests.get(endpoint, params=payload,
                                         headers=headers, timeout=8)
                else:
                    # Generic / fast2sms / msg91
                    payload = dict(cfg.get('extra_params') or {})
                    payload[cfg.get('to_field', 'to')]        = to_number
                    payload[cfg.get('message_field','message')] = message
                    if cfg.get('sender_id'):
                        payload['sender'] = cfg['sender_id']

                    if method == 'POST':
                        if headers.get('Content-Type') == 'application/json':
                            r = requests.post(endpoint, json=payload,
                                              headers=headers, timeout=8)
                        else:
                            r = requests.post(endpoint, data=payload,
                                              headers=headers, timeout=8)
                    else:
                        r = requests.get(endpoint, params=payload,
                                         headers=headers, timeout=8)

                if 200 <= r.status_code < 300:
                    logger.info(f"📱 SMS sent to {to_number}")
                    ok_any = True
                else:
                    logger.error(f"SMS gateway error for {to_number}: "
                                 f"{r.status_code} {r.text[:200]}")
            except Exception as e:
                logger.error(f"SMS send failed for {to_number}: {e}")

        return ok_any

    def _sms_body(self, alert):
        return (f"[CHILD SAFETY] {alert.get('activity','event').upper()} "
                f"({alert.get('severity','')}) — {alert.get('message','')}")

    # ================================================================
    #  CHANNEL: WEBHOOK
    # ================================================================
    def _send_webhook(self, alert):
        if not HAS_REQUESTS:
            return 'skipped'

        cfg  = self.config['webhook']
        urls = cfg.get('urls') or []
        if not urls:
            logger.warning("Webhook channel skipped — no URLs configured")
            return 'skipped'

        method  = (cfg.get('method') or 'POST').upper()
        headers = dict(cfg.get('headers') or
                       {'Content-Type': 'application/json'})
        timeout = cfg.get('timeout', 5)

        ok_any = False
        for url in urls:
            try:
                if method == 'POST':
                    r = requests.post(url, json=alert,
                                      headers=headers, timeout=timeout)
                else:
                    r = requests.get(url, params=alert,
                                     headers=headers, timeout=timeout)
                if 200 <= r.status_code < 300:
                    logger.info(f"🔗 Webhook OK → {url}")
                    ok_any = True
                else:
                    logger.error(f"Webhook error {url}: "
                                 f"{r.status_code} {r.text[:200]}")
            except Exception as e:
                logger.error(f"Webhook send failed → {url}: {e}")

        return ok_any

    # ================================================================
    #  PER-CHANNEL TEST METHODS
    # ================================================================
    def test_desktop(self):
        return self._send_desktop(self._test_payload())

    def test_email(self):
        return self._send_email(self._test_payload())

    def test_telegram(self):
        return self._send_telegram(self._test_payload())

    def test_sms(self):
        return self._send_sms(self._test_payload())

    def test_webhook(self):
        return self._send_webhook(self._test_payload())

    def test_all(self):
        return self.send_alert(self._test_payload())

    def _test_payload(self):
        return {
            'id':         0,
            'timestamp':  datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'message':    'TEST ALERT — this is a drill. No child is at risk.',
            'severity':   'low',
            'activity':   'test',
            'confidence': 1.0,
            'source':     'test_button',
        }
