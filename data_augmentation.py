# data_augmentation.py
import numpy as np
import random
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
        if method == 'random':
            method = random.choice(self.augmentation_methods)

        if method == 'add_noise':
            return self.add_noise(sequence)
        if method == 'scale':
            return self.scale(sequence)
        if method == 'rotate':
            return self.rotate(sequence)
        if method == 'flip_horizontal':
            return self.flip_horizontal(sequence)
        if method == 'time_shift':
            return self.time_shift(sequence)
        if method == 'smooth':
            return self.smooth(sequence)
        if method == 'random_crop':
            return self.random_crop(sequence)
        return sequence

    def add_noise(self, sequence, noise_level=0.02):
        noise = np.random.normal(0, noise_level, sequence.shape)
        return sequence + noise

    def scale(self, sequence, scale_range=(0.9, 1.1)):
        scale_factor = random.uniform(*scale_range)
        return sequence * scale_factor

    def rotate(self, sequence, angle_range=(-15, 15)):
        """Rotate keypoints around the per-frame center (safe copy)."""
        angle = random.uniform(*angle_range)
        angle_rad = np.radians(angle)

        seq = np.array(sequence, dtype=np.float32, copy=True)

        if seq.ndim == 2 and seq.shape[1] % 3 == 0:
            seq3 = seq.reshape(seq.shape[0], -1, 3)
            was_flat = True
        else:
            seq3 = seq
            was_flat = False

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

        return seq3.reshape(seq.shape) if was_flat else seq3

    def flip_horizontal(self, sequence):
        flipped = np.array(sequence, dtype=np.float32, copy=True)
        if flipped.ndim != 2 or flipped.shape[1] % 3 != 0:
            return flipped

        x_cols = np.arange(0, flipped.shape[1], 3)
        xs = flipped[:, x_cols]
        if xs.size == 0:
            return flipped

        if xs.min() < -0.01 or xs.max() > 1.01:
            mid = (xs.min() + xs.max()) / 2.0
            flipped[:, x_cols] = 2.0 * mid - xs
        else:
            flipped[:, x_cols] = 1.0 - xs
        return flipped

    def time_shift(self, sequence, shift_range=(-5, 5)):
        seq = np.array(sequence, dtype=np.float32, copy=True)
        shift = random.randint(*shift_range)
        if shift == 0 or len(seq) == 0:
            return seq

        if shift > 0:
            pad = np.repeat(seq[:1], shift, axis=0)
            return np.vstack([pad, seq[:-shift]])
        pad = np.repeat(seq[-1:], -shift, axis=0)
        return np.vstack([seq[-shift:], pad])

    def smooth(self, sequence, window_length=5, polyorder=2):
        seq = np.array(sequence, dtype=np.float32, copy=True)
        try:
            wl = window_length if window_length % 2 == 1 else window_length + 1
            if wl <= polyorder or wl > len(seq):
                return seq
            return savgol_filter(seq, wl, polyorder, axis=0).astype(np.float32)
        except Exception:
            return seq

    def random_crop(self, sequence, crop_range=(0.8, 1.0)):
        seq = np.array(sequence, dtype=np.float32, copy=True)
        original_len = len(seq)
        if original_len == 0:
            return seq

        keep_ratio = random.uniform(*crop_range)
        new_len = max(1, int(original_len * keep_ratio))
        start_idx = random.randint(0, original_len - new_len)
        cropped = seq[start_idx:start_idx + new_len]

        if len(cropped) < original_len:
            pad = np.repeat(cropped[-1:], original_len - len(cropped), axis=0)
            cropped = np.vstack([cropped, pad])
        return cropped[:original_len]


def generate_augmented_dataset(X, y, augmentations_per_sample=5):
    """Generate augmented dataset without mutating the originals."""
    augmenter = DataAugmenter()
    X_augmented = []
    y_augmented = []

    for i in range(len(X)):
        X_augmented.append(np.array(X[i], copy=True))
        y_augmented.append(y[i])

        for _ in range(augmentations_per_sample):
            aug_method = random.choice(augmenter.augmentation_methods)
            augmented = augmenter.augment_sequence(X[i], aug_method)
            X_augmented.append(augmented)
            y_augmented.append(y[i])

    return np.array(X_augmented), np.array(y_augmented)
