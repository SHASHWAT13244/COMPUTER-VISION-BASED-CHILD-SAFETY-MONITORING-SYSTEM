# utils/alert.py
"""
Unified alert system.

One class handles:
  * throttling (cooldown / rate limit / duplicate suppression)
  * statistics (total, by severity, by activity)
  * dispatch through enabled channels (desktop log/sound, email,
    telegram, SMS, webhook)

`AdvancedAlertSystem` is kept as an alias for backward compatibility.
"""

import os
import json
import time
import threading
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from collections import deque, defaultdict

import requests

try:
    from playsound import playsound
    _HAS_PLAYSOUND = True
except Exception:
    _HAS_PLAYSOUND = False

logger = logging.getLogger(__name__)


class AlertSystem:
    """Consolidated alert manager.

    Usage
    -----
    >>> alerts = AlertSystem(config_path="alert_config.json")
    >>> info = alerts.generate_alert(
    ...     message="Child fell down",
    ...     severity="high",
    ...     activity="falling",
    ...     bbox=(10, 20, 100, 200),
    ...     source="live_feed",
    ... )
    >>> if info:                 # None when throttled
    ...     do_something(info)
    """

    DEFAULT_COOLDOWN_SECONDS   = 5
    DEFAULT_MAX_PER_MINUTE     = 10
    DEFAULT_DUPLICATE_WINDOW_S = 30

    # ------------------------------------------------------------------ #
    # init
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        config_path: str = None,
        sound_enabled: bool = True,
        display_enabled: bool = True,
        log_enabled: bool = True,
    ):
        self.sound_enabled   = bool(sound_enabled)
        self.display_enabled = bool(display_enabled)
        self.log_enabled     = bool(log_enabled)

        self.config_path = config_path
        self.config: dict = {}
        self._config_lock = threading.Lock()

        if self.config_path:
            self._load_config()

        # -------- throttling config --------
        thr = self.config.get("throttling", {}) if self.config else {}
        self.cooldown_seconds = int(
            thr.get("cooldown_seconds", self.DEFAULT_COOLDOWN_SECONDS)
        )
        self.max_alerts_per_minute = int(
            thr.get("max_alerts_per_minute", self.DEFAULT_MAX_PER_MINUTE)
        )
        self.duplicate_window_seconds = int(
            thr.get("duplicate_window_seconds", self.DEFAULT_DUPLICATE_WINDOW_S)
        )

        # -------- runtime state --------
        self._state_lock       = threading.Lock()
        self._last_activity_ts = {}   # activity -> last fire timestamp
        self._last_message_ts  = {}   # message  -> last fire timestamp
        self._recent_alerts    = deque(
            maxlen=max(1, self.max_alerts_per_minute * 3)
        )
        self._stats = {
            "total":       0,
            "by_severity": defaultdict(int),
            "by_activity": defaultdict(int),
        }
        self._started = time.time()

        self._sound_file = (
            self.config.get("sound_file") if self.config else None
        )

    # ------------------------------------------------------------------ #
    # config
    # ------------------------------------------------------------------ #
    def _load_config(self):
        """Load alert_config.json (silently tolerated if missing)."""
        try:
            if self.config_path and os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    self.config = json.load(f)
                logger.info(f"Loaded alert config from {self.config_path}")
            else:
                logger.warning(f"Alert config not found: {self.config_path}")
                self.config = {}
        except Exception as e:
            logger.error(f"Failed to load alert config: {e}")
            self.config = {}

    def reload_config(self):
        """Re-read config file from disk (used by /api/settings)."""
        with self._config_lock:
            self._load_config()

    # ------------------------------------------------------------------ #
    # throttling
    # ------------------------------------------------------------------ #
    def _is_throttled(self, activity: str, message: str, now: float) -> bool:
        """Return True if this (activity, message) should be suppressed now."""
        # Per-activity cooldown.
        last_a = self._last_activity_ts.get(activity)
        if last_a is not None and (now - last_a) < self.cooldown_seconds:
            return True

        # Duplicate suppression.
        last_m = self._last_message_ts.get(message)
        if last_m is not None and (now - last_m) < self.duplicate_window_seconds:
            return True

        # Global rate limit (sliding 60 s window).
        cutoff = now - 60.0
        while self._recent_alerts and self._recent_alerts[0] < cutoff:
            self._recent_alerts.popleft()
        if len(self._recent_alerts) >= self.max_alerts_per_minute:
            return True

        return False

    def _record_fired(self, activity: str, message: str, now: float) -> None:
        self._last_activity_ts[activity] = now
        self._last_message_ts[message]   = now
        self._recent_alerts.append(now)

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def generate_alert(
        self,
        message: str,
        severity: str = "medium",
        activity: str = "unknown",
        bbox=None,
        source: str = "unknown",
        confidence: float = 0.0,
        image_path: str = None,
        send: bool = True,
    ):
        """Apply throttling, record statistics, optionally dispatch.

        Returns
        -------
        dict | None
            The alert dict if it was created.
            `None` if the alert was suppressed by throttling.
        """
        now = time.time()

        with self._state_lock:
            if self._is_throttled(activity, message, now):
                return None

            self._record_fired(activity, message, now)

            alert_info = {
                "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "message":    message,
                "severity":   severity,
                "activity":   activity,
                "confidence": float(confidence),
                "bbox":       list(bbox) if bbox is not None else None,
                "source":     source,
                "image_path": image_path,
            }

            self._stats["total"] += 1
            self._stats["by_severity"][severity] += 1
            self._stats["by_activity"][activity] += 1

        if send:
            self.send_alert(alert_info)
        return alert_info

    def send_alert(self, alert_info: dict):
        """Dispatch an already-created alert through all enabled channels.

        Safe to call from a thread pool. Never raises.
        """
        if not alert_info:
            return

        if self.display_enabled:
            logger.warning(
                f"🚨 ALERT [{alert_info.get('severity')}] "
                f"{alert_info.get('activity')}: {alert_info.get('message')}"
            )
        if self.sound_enabled:
            self._play_sound()
        if self.log_enabled:
            self._log_alert(alert_info)

        methods = self.config.get("enabled_methods", []) if self.config else []
        for m in methods:
            try:
                if m == "desktop":
                    continue  # already handled above
                if m == "email":
                    self._send_email(alert_info)
                elif m == "telegram":
                    self._send_telegram(alert_info)
                elif m == "sms":
                    self._send_sms(alert_info)
                elif m == "webhook":
                    self._send_webhook(alert_info)
            except Exception as e:
                logger.error(f"Alert channel '{m}' failed: {e}")

    def get_alert_statistics(self) -> dict:
        with self._state_lock:
            return {
                "total":          self._stats["total"],
                "by_severity":    dict(self._stats["by_severity"]),
                "by_activity":    dict(self._stats["by_activity"]),
                "uptime_seconds": time.time() - self._started,
            }

    def reset(self):
        """Clear throttling state and statistics."""
        with self._state_lock:
            self._last_activity_ts.clear()
            self._last_message_ts.clear()
            self._recent_alerts.clear()
            self._stats = {
                "total":       0,
                "by_severity": defaultdict(int),
                "by_activity": defaultdict(int),
            }

    # ------------------------------------------------------------------ #
    # channel implementations
    # ------------------------------------------------------------------ #
    def _play_sound(self):
        if not _HAS_PLAYSOUND or not self._sound_file:
            return
        if not os.path.exists(self._sound_file):
            return
        try:
            playsound(self._sound_file, block=False)
        except Exception:
            pass

    def _log_alert(self, alert_info: dict):
        try:
            log_dir = (self.config.get("log_dir") if self.config else None)
            if not log_dir:
                log_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "..", "alerts",
                )
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, "alerts.log"), "a") as f:
                f.write(json.dumps(alert_info) + "\n")
        except Exception as e:
            logger.error(f"Log alert failed: {e}")

    def _send_email(self, alert_info: dict):
        cfg = self.config.get("email", {})
        sender     = cfg.get("sender_email")
        password   = cfg.get("sender_password")
        recipients = cfg.get("recipient_emails", [])
        if not (sender and password and recipients):
            return

        msg = MIMEMultipart()
        msg["From"]    = sender
        msg["To"]      = ", ".join(recipients)
        msg["Subject"] = (
            f"[Child Safety] "
            f"{str(alert_info.get('severity', '')).upper()} - "
            f"{alert_info.get('activity')}"
        )
        body = (
            f"Alert:      {alert_info.get('message')}\n"
            f"Time:       {alert_info.get('timestamp')}\n"
            f"Activity:   {alert_info.get('activity')}\n"
            f"Severity:   {alert_info.get('severity')}\n"
            f"Confidence: {alert_info.get('confidence')}\n"
            f"Source:     {alert_info.get('source')}\n"
        )
        msg.attach(MIMEText(body, "plain"))

        server   = cfg.get("smtp_server", "smtp.gmail.com")
        port     = int(cfg.get("smtp_port", 587))
        security = str(cfg.get("smtp_security", "tls")).lower()

        with smtplib.SMTP(server, port, timeout=10) as smtp:
            if security == "tls":
                smtp.starttls()
            smtp.login(sender, password)
            smtp.sendmail(sender, recipients, msg.as_string())

    def _send_telegram(self, alert_info: dict):
        cfg = self.config.get("telegram", {})
        token    = cfg.get("bot_token")
        chat_ids = cfg.get("chat_ids", [])
        if not (token and chat_ids):
            return

        text = (
            f"🚨 *Child Safety Alert*\n"
            f"*Activity:* {alert_info.get('activity')}\n"
            f"*Severity:* {alert_info.get('severity')}\n"
            f"*Message:* {alert_info.get('message')}\n"
            f"*Time:* {alert_info.get('timestamp')}"
        )
        parse_mode = cfg.get("parse_mode", "Markdown")
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        for chat_id in chat_ids:
            requests.post(
                url,
                json={
                    "chat_id":    chat_id,
                    "text":       text,
                    "parse_mode": parse_mode,
                },
                timeout=8,
            )

    def _send_sms(self, alert_info: dict):
        cfg = self.config.get("sms", {})
        endpoint   = cfg.get("endpoint")
        recipients = cfg.get("recipients", [])
        if not (endpoint and recipients):
            return

        method        = str(cfg.get("method", "POST")).upper()
        headers       = dict(cfg.get("extra_headers", {}))
        auth          = cfg.get("auth_header") or {}
        message_field = cfg.get("message_field", "message")
        to_field      = cfg.get("to_field", "to")
        sender_id     = cfg.get("sender_id")

        if auth.get("name") and auth.get("value"):
            headers[auth["name"]] = auth["value"]

        for to in recipients:
            payload = dict(cfg.get("extra_params", {}))
            payload[message_field] = alert_info.get("message", "")
            payload[to_field]      = to
            if sender_id:
                payload["sender_id"] = sender_id
            if method == "POST":
                requests.post(endpoint, json=payload,
                              headers=headers, timeout=8)
            else:
                requests.get(endpoint, params=payload,
                             headers=headers, timeout=8)

    def _send_webhook(self, alert_info: dict):
        cfg = self.config.get("webhook", {})
        urls = cfg.get("urls", [])
        if not urls:
            return
        method  = str(cfg.get("method", "POST")).upper()
        headers = dict(cfg.get("headers", {}))
        timeout = float(cfg.get("timeout", 5))
        for url in urls:
            if method == "POST":
                requests.post(url, json=alert_info,
                              headers=headers, timeout=timeout)
            else:
                requests.get(url, params=alert_info,
                             headers=headers, timeout=timeout)


# ---------------------------------------------------------------------- #
# Backward-compatible alias: the two classes are now the same one.
# ---------------------------------------------------------------------- #
AdvancedAlertSystem = AlertSystem


__all__ = ["AlertSystem", "AdvancedAlertSystem"]
