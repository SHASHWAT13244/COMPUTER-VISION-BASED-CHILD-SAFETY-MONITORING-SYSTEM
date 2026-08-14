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
            'random_crop'
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
        """Rotate keypoints around center"""
        angle = random.uniform(*angle_range)
        angle_rad = np.radians(angle)
        
        # Get center of keypoints
        center = np.mean(sequence, axis=0)
        centered = sequence - center
        
        # Rotation matrix
        rot_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        
        # Apply rotation to x,y coordinates
        rotated = np.dot(centered[:, :2], rot_matrix.T)
        rotated = np.hstack([rotated, centered[:, 2:]])
        
        return rotated + center
    
    def flip_horizontal(self, sequence):
        """Flip horizontally (mirror)"""
        # Flip x coordinates
        flipped = sequence.copy()
        flipped[:, 0] = 1.0 - flipped[:, 0]  # Assuming normalized coordinates
        return flipped
    
    def time_shift(self, sequence, shift_range=(-5, 5)):
        """Shift sequence in time"""
        shift = random.randint(*shift_range)
        if shift == 0:
            return sequence
        
        if shift > 0:
            shifted = np.vstack([np.zeros((shift, sequence.shape[1])), sequence[:-shift]])
        else:
            shifted = np.vstack([sequence[-shift:], np.zeros((-shift, sequence.shape[1]))])
        
        return shifted
    
    def smooth(self, sequence, window_length=5, polyorder=2):
        """Smooth keypoints using Savitzky-Golay filter"""
        try:
            smoothed = savgol_filter(sequence, window_length, polyorder, axis=0)
            return smoothed
        except:
            return sequence
    
    def random_crop(self, sequence, crop_range=(0.8, 1.0)):
        """Randomly crop sequence length"""
        keep_ratio = random.uniform(*crop_range)
        new_len = int(len(sequence) * keep_ratio)
        start_idx = random.randint(0, len(sequence) - new_len)
        return sequence[start_idx:start_idx + new_len]

def generate_augmented_dataset(X, y, augmentations_per_sample=5):
    """Generate augmented dataset"""
    augmenter = DataAugmenter()
    X_augmented = []
    y_augmented = []
    
    for i in range(len(X)):
        # Add original
        X_augmented.append(X[i])
        y_augmented.append(y[i])
        
        # Add augmented versions
        for _ in range(augmentations_per_sample):
            aug_method = random.choice(augmenter.augmentation_methods)
            augmented = augmenter.augment_sequence(X[i], aug_method)
            X_augmented.append(augmented)
            y_augmented.append(y[i])
    
    return np.array(X_augmented), np.array(y_augmented)