# models/pose_estimator.py
"""
MediaPipe Pose wrapper with per-track pose tracking for multi-person scenes.

Design
------
MediaPipe's Pose solution is *single-person* and *stateful*: one instance
tracks at most one person across frames, and calling process() twice on the
same instance per frame clobbers its tracker state.

To support multiple people we keep a small pool of Pose instances, one per
track id. Each Pose runs on the padded crop of that person's bbox, so the
returned landmarks are normalized to the crop (translation/scale invariant
relative to the person). Frame-space normalized landmarks are also returned
for drawing and for the safety engine.
"""

import threading
from collections import OrderedDict

import cv2
import numpy as np
import mediapipe as mp


class PoseEstimator:
    # How many simultaneous tracks we keep Pose instances for (LRU).
    MAX_POSE_INSTANCES = 10

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity

        # Default single-person instance (used by extract_keypoints()).
        self._default_pose = self._make_pose()

        # Per-track Pose instances, keyed by track_id. OrderedDict = LRU.
        self._pool = OrderedDict()
        self._pool_lock = threading.Lock()

        # Last landmarks for backward-compatible draw_pose().
        self.last_landmarks = None

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    def _make_pose(self):
        return self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

    def _get_track_pose(self, track_id) -> "mp.solutions.pose.Pose":
        with self._pool_lock:
            if track_id in self._pool:
                self._pool.move_to_end(track_id)
                return self._pool[track_id]

            if len(self._pool) >= self.MAX_POSE_INSTANCES:
                _, old = self._pool.popitem(last=False)
                try:
                    old.close()
                except Exception:
                    pass

            p = self._make_pose()
            self._pool[track_id] = p
            return p

    @staticmethod
    def _landmarks_to_array(pose_landmarks) -> np.ndarray:
        return np.array(
            [[lm.x, lm.y, lm.z, lm.visibility]
             for lm in pose_landmarks.landmark],
            dtype=np.float32,
        )

    @staticmethod
    def _crop_to_frame(keypoints_crop, cx1, cy1, cx2, cy2, fw, fh) -> np.ndarray:
        kp = keypoints_crop.copy()
        cw = max(1, cx2 - cx1)
        ch = max(1, cy2 - cy1)
        kp[:, 0] = (kp[:, 0] * cw + cx1) / fw
        kp[:, 1] = (kp[:, 1] * ch + cy1) / fh
        return kp

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def extract_keypoints(self, frame):
        """Single-person pose on full frame (backward-compatible)."""
        if frame is None or frame.size == 0:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._default_pose.process(rgb)
        if not results.pose_landmarks:
            return None
        self.last_landmarks = results.pose_landmarks
        return self._landmarks_to_array(results.pose_landmarks)

    def extract_keypoints_from_bbox(
        self, frame, bbox, track_id=None, pad: float = 0.15
    ):
        """Run pose on a padded crop of `bbox`.

        Returns a dict or None:
            {
              'keypoints_crop':  (33, 4) float32, normalized to [0, 1] within the crop
              'keypoints_frame': (33, 4) float32, normalized to [0, 1] within the frame
              'landmarks':       MediaPipe landmark list
              'crop_box':        (cx1, cy1, cx2, cy2)
            }
        """
        if frame is None or frame.size == 0 or bbox is None:
            return None

        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = (int(b) for b in bbox)

        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            return None

        px = int(bw * pad)
        py = int(bh * pad)

        cx1 = max(0, x1 - px)
        cy1 = max(0, y1 - py)
        cx2 = min(fw, x2 + px)
        cy2 = min(fh, y2 + py)
        if cx2 - cx1 < 8 or cy2 - cy1 < 8:
            return None

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None

        pose = (
            self._get_track_pose(track_id)
            if track_id is not None
            else self._default_pose
        )

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if not results.pose_landmarks:
            return None

        kp_crop = self._landmarks_to_array(results.pose_landmarks)
        kp_frame = self._crop_to_frame(kp_crop, cx1, cy1, cx2, cy2, fw, fh)

        return {
            "keypoints_crop":  kp_crop,
            "keypoints_frame": kp_frame,
            "landmarks":       results.pose_landmarks,
            "crop_box":        (cx1, cy1, cx2, cy2),
        }

    def forget_track(self, track_id):
        with self._pool_lock:
            p = self._pool.pop(track_id, None)
        if p is not None:
            try:
                p.close()
            except Exception:
                pass

    def forget_stale_tracks(self, active_ids):
        active = set(active_ids)
        with self._pool_lock:
            stale = [tid for tid in self._pool.keys() if tid not in active]
        for tid in stale:
            self.forget_track(tid)

    def forget_all_tracks(self):
        with self._pool_lock:
            items = list(self._pool.items())
            self._pool.clear()
        for _, p in items:
            try:
                p.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # drawing
    # ------------------------------------------------------------------ #
    def draw_pose(self, frame, landmarks=None):
        """Backward-compatible: draws the given (or last) landmark set."""
        lm = landmarks if landmarks is not None else self.last_landmarks
        if lm is None:
            return frame
        try:
            self.mp_drawing.draw_landmarks(
                frame, lm, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles
                    .get_default_pose_landmarks_style(),
            )
        except Exception:
            pass
        return frame

    def draw_pose_from_keypoints(
        self, frame, keypoints_frame, color=(0, 255, 0), threshold: float = 0.3
    ):
        """Draw a pose from frame-normalized (33, 4) keypoints using cv2."""
        if keypoints_frame is None:
            return frame
        h, w = frame.shape[:2]
        pts = []
        for x, y, _z, _v in keypoints_frame:
            pts.append((int(x * w), int(y * h)))

        for a, b in self.mp_pose.POSE_CONNECTIONS:
            if keypoints_frame[a, 3] > threshold and keypoints_frame[b, 3] > threshold:
                cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
        for i, pt in enumerate(pts):
            if keypoints_frame[i, 3] > threshold:
                cv2.circle(frame, pt, 3, (0, 0, 255), -1, cv2.LINE_AA)
        return frame

    def close(self):
        try:
            self._default_pose.close()
        except Exception:
            pass
        self.forget_all_tracks()
