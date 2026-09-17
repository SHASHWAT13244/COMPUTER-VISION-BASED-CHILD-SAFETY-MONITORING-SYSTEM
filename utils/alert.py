# utils/alert.py
"""
Basic alert system: desktop popup + sound + local log.
Used as a fallback when the advanced alert system isn't configured.
"""

import os
import logging
import threading
import time
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# Optional dependencies — degrade gracefully if missing
try:
    import playsound
    HAS_PLAYSOUND = True
except Exception:
    HAS_PLAYSOUND = False

try:
    from plyer import notification as plyer_notification
    HAS_PLYER = True
except Exception:
    HAS_PLYER = False


class AlertSystem:
    """
    Basic alert system with desktop notification, optional sound,
    and local logging. Thread-safe and rate-limited.
    """

    def __init__(self, sound_enabled=True, display_enabled=True,
                 log_enabled=True,
                 log_file='alerts/alert_log.jsonl',
                 sound_file='assets/alert.wav',
                 cooldown_seconds=5,
                 max_alerts_per_minute=10):
        self.sound_enabled   = sound_enabled
        self.display_enabled = display_enabled
        self.log_enabled     = log_enabled
        self.log_file        = log_file
        self.sound_file      = sound_file
        self.cooldown_seconds = cooldown_seconds
        self.max_alerts_per_minute = max_alerts_per_minute

        # In-memory alert history
        self.alerts      = []
        self._lock       = threading.Lock()
        self._last_alert_time   = 0.0
        self._minute_window     = []   # timestamps for rate-limiting

        os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)

        if not HAS_PLYER:
            logger.warning("plyer not installed — desktop popups disabled")
        if not HAS_PLAYSOUND:
            logger.info("playsound not installed — desktop sound disabled")

    # ------------------------------------------------------------------
    def generate_alert(self, message, severity='medium',
                       activity='unknown', bbox=None, **extra):
        """
        Create and dispatch a new alert. Returns the alert dict.
        Applies cooldown + rate-limit before dispatching.
        """
        now = time.time()

        # Cooldown check
        if now - self._last_alert_time < self.cooldown_seconds:
            logger.debug("Alert suppressed (cooldown)")
            return None

        # Rate-limit check
        with self._lock:
            self._minute_window = [t for t in self._minute_window
                                   if now - t < 60]
            if len(self._minute_window) >= self.max_alerts_per_minute:
                logger.warning("Alert suppressed (rate limit)")
                return None
            self._minute_window.append(now)

        alert = {
            'id':         len(self.alerts) + 1,
            'timestamp':  datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'message':    str(message),
            'severity':   str(severity).lower(),
            'activity':   str(activity).lower(),
            'bbox':       list(bbox) if bbox else None,
            'source':     extra.get('source', 'realtime'),
        }
        alert.update(extra)

        with self._lock:
            self.alerts.append(alert)
            self._last_alert_time = now

        # Dispatch
        if self.display_enabled:
            self._show_desktop(alert)
        if self.sound_enabled:
            self._play_sound()
        if self.log_enabled:
            self._log_alert(alert)

        logger.info(f"🔔 Alert [{alert['severity']}]: {alert['message']}")
        return alert

    # ------------------------------------------------------------------
    def _show_desktop(self, alert):
        """Show a native desktop notification (best effort)."""
        if not HAS_PLYER:
            return
        try:
            title = {
                'high':   '🚨 UNSAFE — Child Safety Alert',
                'medium': '⚠️ Caution — Child Safety',
                'low':    'ℹ️ Child Safety Notice',
            }.get(alert['severity'], '🔔 Child Safety Alert')

            plyer_notification.notify(
                title=title,
                message=alert['message'][:240],
                app_name='Child Safety Monitor',
                timeout=8,
            )
        except Exception as e:
            logger.debug(f"Desktop notification failed: {e}")

    # ------------------------------------------------------------------
    def _play_sound(self):
        """Play the alert sound on a background thread."""
        if not HAS_PLAYSOUND or not os.path.exists(self.sound_file):
            return
        try:
            threading.Thread(
                target=lambda: playsound.playsound(self.sound_file,
                                                   block=False),
                daemon=True,
            ).start()
        except Exception as e:
            logger.debug(f"Sound playback failed: {e}")

    # ------------------------------------------------------------------
    def _log_alert(self, alert):
        """Append one JSON line per alert to the log file."""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(alert, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Failed to log alert: {e}")

    # ------------------------------------------------------------------
    def get_alert_statistics(self):
        """Return aggregate statistics over all alerts so far."""
        with self._lock:
            alerts = list(self.alerts)
        stats = {
            'total':       len(alerts),
            'by_severity': {},
            'by_activity': {},
            'last_alert':  alerts[-1]['timestamp'] if alerts else None,
        }
        for a in alerts:
            sev = a.get('severity', 'unknown')
            act = a.get('activity', 'unknown')
            stats['by_severity'][sev] = stats['by_severity'].get(sev, 0) + 1
            stats['by_activity'][act] = stats['by_activity'].get(act, 0) + 1
        return stats

    # ------------------------------------------------------------------
    def clear(self):
        """Reset the in-memory alert list."""
        with self._lock:
            self.alerts = []
            self._minute_window = []
