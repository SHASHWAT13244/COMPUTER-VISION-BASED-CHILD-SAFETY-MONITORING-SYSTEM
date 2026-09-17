# data_augmentation.py
import numpy as np
import cv2
import random
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter


class DataAugmenter:
    """Data augmentation for pose keypoint sequences"""

    def __init__(self):
        self.augmentation_methods = [
            'add_noise',
            'scale',
            'rotate',
            'flip_horizontal',
            'time_shift',
            'smooth',
            'random_crop',
        ]

    def augment_sequence(self, sequence, method='random'):
        """Apply augmentation to a sequence of keypoints"""
        if method == 'random':
            method = random.choice(self.augmentation_methods)

        if method == 'add_noise':
            return self.add_noise(sequence)
        elif method == 'scale':
            return self.scale(sequence)
        elif method == 'rotate':
            return self.rotate(sequence)
        elif method == 'flip_horizontal':
            return self.flip_horizontal(sequence)
        elif method == 'time_shift':
            return self.time_shift(sequence)
        elif method == 'smooth':
            return self.smooth(sequence)
        elif method == 'random_crop':
            return self.random_crop(sequence)
        else:
            return sequence

    def add_noise(self, sequence, noise_level=0.02):
        """Add Gaussian noise to keypoints"""
        noise = np.random.normal(0, noise_level, sequence.shape)
        return sequence + noise

    def scale(self, sequence, scale_range=(0.9, 1.1)):
        """Scale keypoints"""
        scale_factor = random.uniform(*scale_range)
        return sequence * scale_factor

    def rotate(self, sequence, angle_range=(-15, 15)):
        """Rotate keypoints around center.

        NOTE: This treats the input as (T, 33*3) and rotates only the
        first landmark (x,y) pair by mistake if called on flat vectors.
        For 2-D sequences of (x, y, z) triplets use rotate_landmarks().
        Kept as-is for backwards compatibility with the existing
        training pipeline that flattens keypoints.
        """
        angle = random.uniform(*angle_range)
        angle_rad = np.radians(angle)

        flat = sequence.ndim == 2 and sequence.shape[1] % 3 == 0

        if flat:
            seq3 = sequence.reshape(sequence.shape[0], -1, 3)
        else:
            seq3 = sequence

        center = np.mean(seq3[..., :2], axis=1, keepdims=True)
        centered = seq3[..., :2] - center

        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        x = centered[..., 0]
        y = centered[..., 1]
        rx = cos_a * x - sin_a * y + center[..., 0]
        ry = sin_a * x + cos_a * y + center[..., 1]

        seq3[..., 0] = rx
        seq3[..., 1] = ry

        if flat:
            return seq3.reshape(sequence.shape)
        return seq3

    def flip_horizontal(self, sequence):
        """Flip horizontally (mirror).

        Works for both normalized [0,1] coordinates and pixel coordinates.
        Expects a flat (T, 33*3) sequence.
        """
        flipped = sequence.copy()
        if flipped.ndim != 2 or flipped.shape[1] % 3 != 0:
            return flipped

        x_cols = np.arange(0, flipped.shape[1], 3)
        xs = flipped[:, x_cols]

        if xs.min() < -0.01 or xs.max() > 1.01:
            # Assume pixel coordinates: mirror around horizontal center
            mid = (xs.min() + xs.max()) / 2.0
            flipped[:, x_cols] = 2.0 * mid - xs
        else:
            # Normalized coordinates: mirror around 0.5
            flipped[:, x_cols] = 1.0 - xs

        return flipped

    def time_shift(self, sequence, shift_range=(-5, 5)):
        """Shift sequence in time, repeating boundary frames (no zeros)."""
        shift = random.randint(*shift_range)
        if shift == 0:
            return sequence

        if shift > 0:
            pad = np.repeat(sequence[:1], shift, axis=0)
            return np.vstack([pad, sequence[:-shift]])
        pad = np.repeat(sequence[-1:], -shift, axis=0)
        return np.vstack([sequence[-shift:], pad])

    def smooth(self, sequence, window_length=5, polyorder=2):
        """Smooth keypoints using Savitzky-Golay filter"""
        try:
            # window_length must be odd and > polyorder
            wl = window_length if window_length % 2 == 1 else window_length + 1
            if wl <= polyorder:
                return sequence
            return savgol_filter(sequence, wl, polyorder, axis=0)
        except Exception:
            return sequence

    def random_crop(self, sequence, crop_range=(0.8, 1.0)):
        """Randomly crop sequence length, then pad back to original length
        by repeating the boundary frame. Preserves fixed-length input for LSTM.
        """
        original_len = len(sequence)
        if original_len == 0:
            return sequence

        keep_ratio = random.uniform(*crop_range)
        new_len = max(1, int(original_len * keep_ratio))
        start_idx = random.randint(0, original_len - new_len)
        cropped = sequence[start_idx:start_idx + new_len]

        if len(cropped) < original_len:
            pad = np.repeat(cropped[-1:], original_len - len(cropped), axis=0)
            cropped = np.vstack([cropped, pad])

        return cropped[:original_len]


def generate_augmented_dataset(X, y, augmentations_per_sample=5):
    """Generate augmented dataset"""
    augmenter = DataAugmenter()
    X_augmented = []
    y_augmented = []

    for i in range(len(X)):
        X_augmented.append(X[i])
        y_augmented.append(y[i])

        for _ in range(augmentations_per_sample):
            aug_method = random.choice(augmenter.augmentation_methods)
            augmented = augmenter.augment_sequence(X[i], aug_method)
            X_augmented.append(augmented)
            y_augmented.append(y[i])

    return np.array(X_augmented), np.array(y_augmented)
