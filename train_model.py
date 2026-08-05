"""
Training script for activity recognition model
"""

import sys
import numpy as np
import torch
from pathlib import Path
import argparse
import logging
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from modules.activity_recognition import ActivityRecognizer
import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActivityModelTrainer:
    """Trainer for activity recognition model"""
    
    def __init__(self):
        self.recognizer = ActivityRecognizer(
            sequence_length=config.SEQUENCE_LENGTH,
            features_per_frame=config.FEATURES_PER_FRAME,
            num_classes=len(config.ACTIVITY_CLASSES)
        )
        self.recognizer.initialize_model()
    
    def load_data(self, data_path):
        """
        Load training data from file
        
        Args:
            data_path: Path to data file or directory
            
        Returns:
            X, y arrays
        """
        # If data directory is provided, load all files
        data_path = Path(data_path)
        
        if data_path.is_dir():
            X_list = []
            y_list = []
            
            for file in data_path.glob('*.npy'):
                data = np.load(file)
                X_list.append(data['X'])
                y_list.append(data['y'])
            
            X = np.concatenate(X_list)
            y = np.concatenate(y_list)
        else:
            data = np.load(data_path)
            X = data['X']
            y = data['y']
        
        logger.info(f"Loaded data: {X.shape[0]} samples")
        return X, y
    
    def generate_synthetic_data(self, num_samples=1000, save_path=None):
        """
        Generate synthetic training data
        
        Args:
            num_samples: Number of samples to generate
            save_path: Path to save the generated data
            
        Returns:
            X, y arrays
        """
        logger.info(f"Generating {num_samples} synthetic samples...")
        X, y = self.recognizer.generate_sample_data(num_samples)
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(exist_ok=True)
            np.savez(save_path, X=X, y=y)
            logger.info(f"Data saved to {save_path}")
        
        return X, y
    
    def train(self, X, y, epochs=50, batch_size=32, validation_split=0.2):
        """
        Train the model
        
        Args:
            X: Input data
            y: Labels
            epochs: Number of epochs
            batch_size: Batch size
            validation_split: Validation split ratio
            
        Returns:
            Training history
        """
        # Prepare sequences
        X_train, y_train = self.recognizer.prepare_sequences(X, y)
        
        # Train the model
        history = self.recognizer.train(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split
        )
        
        return history
    
    def save_model(self, save_dir='models'):
        """
        Save the trained model
        
        Args:
            save_dir: Directory to save the model
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        self.recognizer.save_model(save_dir)
        logger.info(f"Model saved to {save_dir}")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model
        
        Args:
            X_test: Test data
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        # Prepare data
        X_test_seq, y_test_seq = self.recognizer.prepare_sequences(X_test, y_test)
        
        # Convert to tensors
        X_test_tensor = torch.FloatTensor(X_test_seq)
        y_test_tensor = torch.LongTensor(y_test_seq)
        
        # Evaluate
        self.recognizer.model.eval()
        with torch.no_grad():
            outputs = self.recognizer.model(X_test_tensor)
            _, predicted = torch.max(outputs.data, 1)
            
            correct = (predicted == y_test_tensor).sum().item()
            total = y_test_tensor.size(0)
            accuracy = correct / total
            
            # Calculate per-class accuracy
            from sklearn.metrics import classification_report
            report = classification_report(
                y_test_tensor.numpy(),
                predicted.numpy(),
                target_names=self.recognizer.classes
            )
        
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Classification Report:\n{report}")
        
        return {
            'accuracy': accuracy,
            'classification_report': report
        }

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Activity Recognition Model')
    parser.add_argument('--data', type=str, help='Path to training data')
    parser.add_argument('--generate', action='store_true', help='Generate synthetic data')
    parser.add_argument('--samples', type=int, default=1000, help='Number of synthetic samples')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--save', type=str, default='models', help='Model save directory')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the trained model')
    
    args = parser.parse_args()
    
    trainer = ActivityModelTrainer()
    
    if args.generate:
        # Generate synthetic data
        data_path = Path('data/synthetic_activity_data.npz')
        X, y = trainer.generate_synthetic_data(num_samples=args.samples, save_path=data_path)
    elif args.data:
        # Load data from file
        X, y = trainer.load_data(args.data)
    else:
        logger.error("Please specify --data or --generate")
        return
    
    # Train the model
    history = trainer.train(X, y, epochs=args.epochs, batch_size=args.batch_size)
    
    # Save the model
    trainer.save_model(args.save)
    
    # Evaluate if requested
    if args.evaluate:
        trainer.evaluate(X, y)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()