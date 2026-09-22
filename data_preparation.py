# data_preparation.py
"""
Data Preparation Script for Child Safety Monitoring System
Extracts pose keypoints from videos and prepares training data.
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import json
from tqdm import tqdm
import argparse
import logging
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreparator:
    """Prepare training data from videos."""

    def __init__(self, sequence_length=30, num_keypoints=33,
                 min_detection_confidence=0.5):
        self.sequence_length = sequence_length
        self.num_keypoints = num_keypoints
        self.min_detection_confidence = min_detection_confidence

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )

        self.activities = list(Config.ACTIVITY_CLASSES)

        # Lazy-loaded detector (used for cropping each person's bbox so
        # training keypoints are person-relative and match inference).
        self._detector = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        try:
            self.pose.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # detector helper
    # ------------------------------------------------------------------ #
    def _get_detector(self):
        if self._detector is None:
            # Import lazily to avoid pulling ultralytics at module import time
            from models.detector import ChildDetector
            self._detector = ChildDetector(
                model_path=Config.YOLO_MODEL,
                conf_threshold=Config.CONFIDENCE_THRESHOLD,
                iou_threshold=Config.IOU_THRESHOLD,
            )
        return self._detector

    @staticmethod
    def _pick_largest_bbox(detections):
        if not detections:
            return None
        detections = sorted(
            detections,
            key=lambda d: ((d['bbox'][2] - d['bbox'][0]) *
                           (d['bbox'][3] - d['bbox'][1])),
            reverse=True,
        )
        return detections[0]['bbox']

    @staticmethod
    def _pad_crop_bbox(bbox, frame_shape, pad=0.15):
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = (int(v) for v in bbox)
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            return None
        px = int(bw * pad)
        py = int(bh * pad)
        cx1 = max(0, x1 - px)
        cy1 = max(0, y1 - py)
        cx2 = min(w, x2 + px)
        cy2 = min(h, y2 + py)
        if cx2 - cx1 < 8 or cy2 - cy1 < 8:
            return None
        return cx1, cy1, cx2, cy2

    # ------------------------------------------------------------------ #
    # extraction
    # ------------------------------------------------------------------ #
    def extract_sequences_from_video(self, video_path):
        """Extract pose sequences from a video, cropping each person's bbox
        so keypoints are person-relative (matches inference-time normalization).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return []

        detector = self._get_detector()

        sequences = []
        current_sequence = []
        frame_count = 0
        success_count = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # 1) Detect the person and crop.
                try:
                    detections = detector.detect(frame)
                except Exception as e:
                    logger.debug(f"Detection failed on frame {frame_count}: {e}")
                    detections = []

                bbox = self._pick_largest_bbox(detections)
                crop_box = None
                if bbox is not None:
                    crop_box = self._pad_crop_bbox(bbox, frame.shape)

                keypoints = None
                if crop_box is not None:
                    cx1, cy1, cx2, cy2 = crop_box
                    crop = frame[cy1:cy2, cx1:cx2]
                    if crop.size > 0:
                        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        results = self.pose.process(rgb)
                        if results.pose_landmarks:
                            flat = []
                            for lm in results.pose_landmarks.landmark:
                                flat.extend([lm.x, lm.y, lm.z])
                            keypoints = flat

                # 2) Buffer the pose (or hold last valid one on gap).
                if keypoints is not None:
                    current_sequence.append(keypoints)
                    success_count += 1
                elif current_sequence:
                    # Hold the last valid frame instead of inserting zeros.
                    current_sequence.append(current_sequence[-1])

                if len(current_sequence) == self.sequence_length:
                    sequences.append(np.array(current_sequence))
                    current_sequence = []

        finally:
            cap.release()

        logger.info(
            f"  Processed {frame_count} frames, "
            f"{success_count} with pose, "
            f"extracted {len(sequences)} sequences"
        )
        return sequences

    def prepare_data_from_videos(self, data_dir='data/activities',
                                 output_path='data/training_data.npz'):
        X_data = []
        y_data = []

        logger.info("=" * 60)
        logger.info("DATA PREPARATION")
        logger.info("=" * 60)

        data_path = Path(data_dir)
        if not data_path.exists():
            logger.error(f"Data directory not found: {data_dir}")
            logger.info("Creating directory structure...")
            self.create_directory_structure(data_dir)
            return None, None

        total_sequences = 0
        for activity_idx, activity in enumerate(self.activities):
            activity_dir = data_path / activity
            if not activity_dir.exists():
                logger.warning(f"Directory not found: {activity_dir}")
                continue

            video_files = []
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm']:
                video_files.extend(activity_dir.glob(ext))

            if not video_files:
                logger.warning(f"No video files found in {activity_dir}")
                continue

            logger.info(f"\n📂 Processing: {activity}")
            logger.info(f"   Found {len(video_files)} video(s)")

            activity_sequences = 0
            for video_file in tqdm(video_files, desc=f"   Processing {activity}"):
                sequences = self.extract_sequences_from_video(str(video_file))
                for seq in sequences:
                    X_data.append(seq)
                    y_data.append(activity_idx)
                activity_sequences += len(sequences)

            logger.info(
                f"   Extracted {activity_sequences} sequences for {activity}")
            total_sequences += activity_sequences

        if not X_data:
            logger.error("No data extracted. Please check your video files.")
            return None, None

        X_data = np.array(X_data)
        y_data = np.array(y_data)

        logger.info("\n" + "=" * 60)
        logger.info("DATA SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total sequences: {len(X_data)}")
        logger.info(f"Feature shape: {X_data.shape}")

        for idx, activity in enumerate(self.activities):
            count = int(np.sum(y_data == idx))
            logger.info(f"   {activity}: {count} sequences")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, X=X_data, y=y_data)
        logger.info(f"\n✅ Data saved to {output_path}")

        metadata = {
            'created':         datetime.now().isoformat(),
            'num_samples':     len(X_data),
            'num_classes':     len(self.activities),
            'classes':         self.activities,
            'sequence_length': self.sequence_length,
            'num_keypoints':   self.num_keypoints,
            'feature_shape':   list(X_data.shape[1:]),
        }
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✅ Metadata saved to {metadata_path}")

        return X_data, y_data

    def create_directory_structure(self, data_dir='data/activities'):
        data_path = Path(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)

        for activity in self.activities:
            activity_path = data_path / activity
            activity_path.mkdir(exist_ok=True)
            logger.info(f"✅ Created: {activity_path}")

        readme_path = data_path / 'README.txt'
        with open(readme_path, 'w') as f:
            f.write(f"""
CHILD SAFETY MONITORING - TRAINING DATA

Directory Structure:
- walking/    : Videos of people walking
- running/    : Videos of people running
- sitting/    : Videos of people sitting
- falling/    : Videos of people falling
- climbing/   : Videos of people climbing

Requirements:
- Each video should focus on a single person
- Person should be clearly visible
- Videos should be at least 30 frames (1 second at 30fps)
- Supported formats: .mp4, .avi, .mov, .mkv, .webm

Instructions:
1. Place videos in the appropriate activity folder
2. Run data_preparation.py to extract features
3. Train the model using main.py

For best results:
- Use different lighting conditions
- Use different camera angles
- Use different clothing styles
- Use different backgrounds
- Videos should be 10-30 seconds long

Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")

        logger.info(f"✅ Created: {readme_path}")
        logger.info("\n📝 Instructions:")
        logger.info("1. Add video files to the respective activity folders")
        logger.info("2. Run: python data_preparation.py --process")

    def generate_synthetic_data(self, num_samples=100,
                                output_path='data/training_data_synthetic.npz'):
        logger.info("\n" + "=" * 60)
        logger.info("GENERATING SYNTHETIC DATA")
        logger.info("=" * 60)

        X_data = []
        y_data = []

        logger.info(f"Generating {num_samples} synthetic sequences per class...")

        NOSE_X       = 0 * 3
        L_ANKLE_X    = 27 * 3
        R_ANKLE_X    = 28 * 3
        L_SHOULDER_Y = 11 * 3 + 1
        R_SHOULDER_Y = 12 * 3 + 1
        L_HIP_Y      = 23 * 3 + 1
        R_HIP_Y      = 24 * 3 + 1
        L_KNEE_Y     = 25 * 3 + 1
        R_KNEE_Y     = 26 * 3 + 1

        for activity_idx, activity in enumerate(self.activities):
            for _ in tqdm(range(num_samples), desc=f"   {activity}"):
                sequence = np.random.rand(
                    self.sequence_length, self.num_keypoints * 3)

                if activity == 'walking':
                    for i in range(self.sequence_length):
                        sequence[i, NOSE_X]    += 0.10 * np.sin(i * 0.2)
                        sequence[i, L_ANKLE_X] += 0.10 * np.sin(i * 0.2 + 1)
                        sequence[i, R_ANKLE_X] += 0.10 * np.sin(i * 0.2 + 2)
                elif activity == 'running':
                    for i in range(self.sequence_length):
                        sequence[i, NOSE_X]    += 0.20 * np.sin(i * 0.4)
                        sequence[i, L_ANKLE_X] += 0.20 * np.sin(i * 0.4 + 1)
                        sequence[i, R_ANKLE_X] += 0.20 * np.sin(i * 0.4 + 2)
                elif activity == 'sitting':
                    for i in range(self.sequence_length):
                        sequence[i, L_HIP_Y]  = 0.5
                        sequence[i, R_HIP_Y]  = 0.5
                        sequence[i, L_KNEE_Y] = 0.5
                        sequence[i, R_KNEE_Y] = 0.5
                elif activity == 'falling':
                    for i in range(self.sequence_length):
                        decay = 0.02 * i
                        sequence[i, L_SHOULDER_Y] -= decay
                        sequence[i, R_SHOULDER_Y] -= decay
                        sequence[i, L_HIP_Y]      -= decay
                        sequence[i, R_HIP_Y]      -= decay
                elif activity == 'climbing':
                    for i in range(self.sequence_length):
                        growth = 0.015 * i
                        sequence[i, L_SHOULDER_Y] -= growth
                        sequence[i, R_SHOULDER_Y] -= growth
                        sequence[i, L_HIP_Y]      -= growth
                        sequence[i, R_HIP_Y]      -= growth

                X_data.append(sequence)
                y_data.append(activity_idx)

        X_data = np.array(X_data)
        y_data = np.array(y_data)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, X=X_data, y=y_data)

        logger.info(f"\n✅ Synthetic data saved to {output_path}")
        logger.info(f"   Total samples: {len(X_data)}")
        return X_data, y_data

    def preview_data(self, data_path='data/training_data.npz', num_samples=5):
        """Preview the prepared data (safe on empty datasets)."""
        if not os.path.exists(data_path):
            logger.error(f"Data not found: {data_path}")
            return

        data = np.load(data_path)
        X = data['X']
        y = data['y']

        logger.info("\n" + "=" * 60)
        logger.info("DATA PREVIEW")
        logger.info("=" * 60)
        logger.info(f"Total samples: {len(X)}")
        logger.info(f"Feature shape: {X.shape}")
        logger.info(f"Labels shape: {y.shape}")

        if len(X) == 0:
            logger.warning("Empty dataset — nothing to preview.")
            return

        logger.info("\nRandom samples:")
        n = min(num_samples, len(X))
        indices = np.random.choice(len(X), n, replace=False)

        for idx in indices:
            activity = (self.activities[y[idx]]
                        if y[idx] < len(self.activities) else 'unknown')
            logger.info(f"\n  Sample {idx}:")
            logger.info(f"    Activity: {activity}")
            logger.info(f"    Sequence shape: {X[idx].shape}")
            logger.info(f"    Min: {X[idx].min():.3f}, Max: {X[idx].max():.3f}")
            logger.info(f"    Mean: {X[idx].mean():.3f}, "
                        f"Std: {X[idx].std():.3f}")

        logger.info("\nClass distribution:")
        for idx, activity in enumerate(self.activities):
            count = int(np.sum(y == idx))
            percentage = 100 * count / len(y) if len(y) else 0.0
            logger.info(f"  {activity}: {count} ({percentage:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare training data for child safety monitoring'
    )
    parser.add_argument('--create-dirs', action='store_true',
                        help='Create directory structure')
    parser.add_argument('--process', action='store_true',
                        help='Process videos and extract features')
    parser.add_argument('--synthetic', type=int, nargs='?', const=100,
                        help='Generate synthetic data '
                             '(default: 100 samples per class)')
    parser.add_argument('--preview', type=str, nargs='?',
                        const='data/training_data.npz',
                        help='Preview data (default: data/training_data.npz)')
    parser.add_argument('--data-dir', type=str, default='data/activities',
                        help='Data directory (default: data/activities)')
    parser.add_argument('--output', type=str, default='data/training_data.npz',
                        help='Output path (default: data/training_data.npz)')
    parser.add_argument('--sequence-length', type=int, default=30,
                        help='Sequence length (default: 30)')

    args = parser.parse_args()

    with DataPreparator(sequence_length=args.sequence_length) as preparator:
        if args.create_dirs:
            preparator.create_directory_structure(args.data_dir)
            return
        if args.synthetic:
            preparator.generate_synthetic_data(
                num_samples=args.synthetic,
                output_path='data/training_data_synthetic.npz'
            )
            return
        if args.preview:
            preparator.preview_data(args.preview)
            return
        if args.process:
            preparator.prepare_data_from_videos(
                data_dir=args.data_dir,
                output_path=args.output
            )
            return

        print("""
Usage:
    python data_preparation.py --create-dirs    Create directory structure
    python data_preparation.py --process        Process existing videos
    python data_preparation.py --synthetic [N]  Generate synthetic data
    python data_preparation.py --preview [PATH] Preview data

Examples:
    python data_preparation.py --create-dirs
    python data_preparation.py --synthetic 200
    python data_preparation.py --preview
    python data_preparation.py --process
    """)


if __name__ == "__main__":
    main()
