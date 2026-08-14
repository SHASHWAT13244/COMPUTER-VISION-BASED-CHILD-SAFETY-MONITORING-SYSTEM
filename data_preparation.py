# data_preparation.py
"""
Data Preparation Script for Child Safety Monitoring System
Extracts pose keypoints from videos and prepares training data
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import json
from tqdm import tqdm
import argparse
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreparator:
    """
    Prepare training data from videos
    """
    
    def __init__(self, sequence_length=30, num_keypoints=33, min_detection_confidence=0.5):
        """
        Initialize data preparator
        
        Args:
            sequence_length: Number of frames per sequence
            num_keypoints: Number of keypoints from MediaPipe
            min_detection_confidence: Minimum confidence for detection
        """
        self.sequence_length = sequence_length
        self.num_keypoints = num_keypoints
        self.min_detection_confidence = min_detection_confidence
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        
        self.activities = ['walking', 'running', 'sitting', 'falling', 'climbing']
        
    def extract_sequences_from_video(self, video_path):
        """
        Extract pose sequences from a video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            list: List of sequences (each sequence is list of keypoints)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return []
        
        sequences = []
        current_sequence = []
        frame_count = 0
        success_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Extract keypoints (x, y, z)
                keypoints = []
                for landmark in results.pose_landmarks.landmark:
                    keypoints.extend([landmark.x, landmark.y, landmark.z])
                
                current_sequence.append(keypoints)
                success_count += 1
                
                # If we have enough frames, save sequence
                if len(current_sequence) == self.sequence_length:
                    sequences.append(np.array(current_sequence))
                    current_sequence = []
            else:
                # No pose detected, add zeros to maintain sequence
                if current_sequence:
                    current_sequence.append(np.zeros(self.num_keypoints * 3))
                    
                    if len(current_sequence) == self.sequence_length:
                        sequences.append(np.array(current_sequence))
                        current_sequence = []
        
        cap.release()
        
        logger.info(f"  Processed {frame_count} frames, {success_count} with pose, "
                   f"extracted {len(sequences)} sequences")
        
        return sequences
    
    def prepare_data_from_videos(self, data_dir='data/activities', output_path='data/training_data.npz'):
        """
        Prepare training data from videos in directories
        
        Args:
            data_dir: Directory containing activity subdirectories
            output_path: Path to save the prepared data
            
        Returns:
            tuple: (X_data, y_data) arrays
        """
        X_data = []
        y_data = []
        
        logger.info("="*60)
        logger.info("DATA PREPARATION")
        logger.info("="*60)
        
        # Check if data directory exists
        data_path = Path(data_dir)
        if not data_path.exists():
            logger.error(f"Data directory not found: {data_dir}")
            logger.info("Creating directory structure...")
            self.create_directory_structure(data_dir)
            return None, None
        
        # Process each activity
        total_sequences = 0
        for activity_idx, activity in enumerate(self.activities):
            activity_dir = data_path / activity
            if not activity_dir.exists():
                logger.warning(f"Directory not found: {activity_dir}")
                continue
            
            # Get video files
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
            
            logger.info(f"   Extracted {activity_sequences} sequences for {activity}")
            total_sequences += activity_sequences
        
        if not X_data:
            logger.error("No data extracted. Please check your video files.")
            return None, None
        
        # Convert to numpy arrays
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        
        logger.info("\n" + "="*60)
        logger.info("DATA SUMMARY")
        logger.info("="*60)
        logger.info(f"Total sequences: {len(X_data)}")
        logger.info(f"Feature shape: {X_data.shape}")
        
        # Count per class
        for idx, activity in enumerate(self.activities):
            count = np.sum(y_data == idx)
            logger.info(f"   {activity}: {count} sequences")
        
        # Save data
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, X=X_data, y=y_data)
        logger.info(f"\n✅ Data saved to {output_path}")
        
        # Save metadata
        metadata = {
            'created': datetime.now().isoformat(),
            'num_samples': len(X_data),
            'num_classes': len(self.activities),
            'classes': self.activities,
            'sequence_length': self.sequence_length,
            'num_keypoints': self.num_keypoints,
            'feature_shape': X_data.shape[1:]
        }
        
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✅ Metadata saved to {metadata_path}")
        
        return X_data, y_data
    
    def create_directory_structure(self, data_dir='data/activities'):
        """
        Create the directory structure for training data
        
        Args:
            data_dir: Root data directory
        """
        data_path = Path(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        
        # Create activity directories
        for activity in self.activities:
            activity_path = data_path / activity
            activity_path.mkdir(exist_ok=True)
            logger.info(f"✅ Created: {activity_path}")
        
        # Create README
        readme_path = data_path / 'README.txt'
        with open(readme_path, 'w') as f:
            f.write("""
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

Created: {}
""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        
        logger.info(f"✅ Created: {readme_path}")
        logger.info("\n📝 Instructions:")
        logger.info("1. Add video files to the respective activity folders")
        logger.info("2. Run: python data_preparation.py --process")
    
    def generate_synthetic_data(self, num_samples=100, output_path='data/training_data_synthetic.npz'):
        """
        Generate synthetic training data for testing
        
        Args:
            num_samples: Number of samples per class
            output_path: Path to save the data
            
        Returns:
            tuple: (X_data, y_data) arrays
        """
        logger.info("\n" + "="*60)
        logger.info("GENERATING SYNTHETIC DATA")
        logger.info("="*60)
        
        X_data = []
        y_data = []
        
        logger.info(f"Generating {num_samples} synthetic sequences per class...")
        
        for activity_idx, activity in enumerate(self.activities):
            for _ in tqdm(range(num_samples), desc=f"   {activity}"):
                # Generate random keypoints
                sequence = np.random.rand(self.sequence_length, self.num_keypoints * 3)
                
                # Add patterns based on activity
                if activity == 'walking':
                    # Walking: regular movement
                    for i in range(self.sequence_length):
                        sequence[i, 0] += 0.1 * np.sin(i * 0.2)  # Nose
                        sequence[i, 15] += 0.1 * np.sin(i * 0.2 + 1)  # Left ankle
                        sequence[i, 28] += 0.1 * np.sin(i * 0.2 + 2)  # Right ankle
                        
                elif activity == 'running':
                    # Running: faster movement
                    for i in range(self.sequence_length):
                        sequence[i, 0] += 0.2 * np.sin(i * 0.4)
                        sequence[i, 15] += 0.2 * np.sin(i * 0.4 + 1)
                        sequence[i, 28] += 0.2 * np.sin(i * 0.4 + 2)
                        
                elif activity == 'sitting':
                    # Sitting: minimal movement, hips and knees at same height
                    for i in range(self.sequence_length):
                        sequence[i, 23:29] = 0.5  # Hips and knees
                        
                elif activity == 'falling':
                    # Falling: rapid descent
                    for i in range(self.sequence_length):
                        decay = 0.02 * i
                        sequence[i, 11:15] -= decay  # Shoulders dropping
                        sequence[i, 23:29] -= decay  # Hips dropping
                        
                elif activity == 'climbing':
                    # Climbing: upward movement
                    for i in range(self.sequence_length):
                        growth = 0.015 * i
                        sequence[i, 11:15] -= growth  # Shoulders going up
                        sequence[i, 23:29] -= growth  # Hips going up
                
                X_data.append(sequence)
                y_data.append(activity_idx)
        
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        
        # Save data
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, X=X_data, y=y_data)
        
        logger.info(f"\n✅ Synthetic data saved to {output_path}")
        logger.info(f"   Total samples: {len(X_data)}")
        
        return X_data, y_data
    
    def preview_data(self, data_path='data/training_data.npz', num_samples=5):
        """
        Preview the prepared data
        
        Args:
            data_path: Path to training data
            num_samples: Number of samples to preview
        """
        if not os.path.exists(data_path):
            logger.error(f"Data not found: {data_path}")
            return
        
        data = np.load(data_path)
        X = data['X']
        y = data['y']
        
        logger.info("\n" + "="*60)
        logger.info("DATA PREVIEW")
        logger.info("="*60)
        logger.info(f"Total samples: {len(X)}")
        logger.info(f"Feature shape: {X.shape}")
        logger.info(f"Labels shape: {y.shape}")
        
        # Sample preview
        logger.info(f"\nRandom samples:")
        indices = np.random.choice(len(X), min(num_samples, len(X)), replace=False)
        
        for idx in indices:
            activity = self.activities[y[idx]]
            logger.info(f"\n  Sample {idx}:")
            logger.info(f"    Activity: {activity}")
            logger.info(f"    Sequence shape: {X[idx].shape}")
            logger.info(f"    Min: {X[idx].min():.3f}, Max: {X[idx].max():.3f}")
            logger.info(f"    Mean: {X[idx].mean():.3f}, Std: {X[idx].std():.3f}")
        
        # Class distribution
        logger.info(f"\nClass distribution:")
        for idx, activity in enumerate(self.activities):
            count = np.sum(y == idx)
            percentage = 100 * count / len(y)
            logger.info(f"  {activity}: {count} ({percentage:.1f}%)")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Prepare training data for child safety monitoring'
    )
    parser.add_argument(
        '--create-dirs',
        action='store_true',
        help='Create directory structure'
    )
    parser.add_argument(
        '--process',
        action='store_true',
        help='Process videos and extract features'
    )
    parser.add_argument(
        '--synthetic',
        type=int,
        nargs='?',
        const=100,
        help='Generate synthetic data (default: 100 samples per class)'
    )
    parser.add_argument(
        '--preview',
        type=str,
        nargs='?',
        const='data/training_data.npz',
        help='Preview data (default: data/training_data.npz)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/activities',
        help='Data directory (default: data/activities)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/training_data.npz',
        help='Output path (default: data/training_data.npz)'
    )
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=30,
        help='Sequence length (default: 30)'
    )
    
    args = parser.parse_args()
    
    # Initialize preparator
    preparator = DataPreparator(sequence_length=args.sequence_length)
    
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
    
    # Default: show help
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