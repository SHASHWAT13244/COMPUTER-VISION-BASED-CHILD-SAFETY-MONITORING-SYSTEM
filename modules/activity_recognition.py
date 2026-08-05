"""
Activity Recognition module using LSTM
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class LSTMActivityClassifier(nn.Module):
    """LSTM-based activity classifier"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=5, dropout=0.3):
        """
        Initialize LSTM classifier
        
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden layer size
            num_layers: Number of LSTM layers
            num_classes: Number of activity classes
            dropout: Dropout rate
        """
        super(LSTMActivityClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        # Take the output from the last time step
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output

class ActivityRecognizer:
    """Activity recognition system using LSTM"""
    
    def __init__(self, sequence_length=30, features_per_frame=99, num_classes=5):
        """
        Initialize activity recognizer
        
        Args:
            sequence_length: Number of frames in a sequence
            features_per_frame: Number of features per frame (33 keypoints * 3)
            num_classes: Number of activity classes
        """
        self.sequence_length = sequence_length
        self.features_per_frame = features_per_frame
        self.num_classes = num_classes
        self.classes = ['walking', 'running', 'sitting', 'standing', 'falling']
        
        self.model = None
        self.label_encoder = None
        self.input_buffer = []
    
    def initialize_model(self, hidden_size=128, num_layers=2, dropout=0.3):
        """Initialize the LSTM model"""
        self.model = LSTMActivityClassifier(
            input_size=self.features_per_frame,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=self.num_classes,
            dropout=dropout
        )
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.classes)
    
    def prepare_sequences(self, keypoints_sequences, labels):
        """
        Prepare sequences for training
        
        Args:
            keypoints_sequences: List of keypoints sequences
            labels: Corresponding activity labels
            
        Returns:
            X, y ready for training
        """
        X = np.array(keypoints_sequences)
        y = self.label_encoder.transform(labels)
        
        return X, y
    
    def train(self, X, y, epochs=50, batch_size=32, validation_split=0.2):
        """
        Train the LSTM model
        
        Args:
            X: Input sequences (samples, sequence_length, features)
            y: Labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            
        Returns:
            Training history
        """
        if self.model is None:
            self.initialize_model()
        
        # Convert to PyTorch tensors
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = correct / total
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, '
                          f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        return history
    
    def predict(self, sequence):
        """
        Predict activity from a sequence of keypoints
        
        Args:
            sequence: Array of keypoints (sequence_length, features)
            
        Returns:
            Predicted activity class and confidence
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")
        
        self.model.eval()
        with torch.no_grad():
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dimension
            outputs = self.model(sequence_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            class_name = self.label_encoder.inverse_transform(predicted.numpy())[0]
            confidence_score = confidence.numpy()[0]
            
            return class_name, confidence_score
    
    def update_buffer(self, keypoints):
        """
        Update the input buffer with new keypoints
        
        Args:
            keypoints: Array of keypoints (33x3)
            
        Returns:
            Complete sequence if buffer is full, else None
        """
        # Flatten the keypoints
        flattened = keypoints.flatten()
        self.input_buffer.append(flattened)
        
        if len(self.input_buffer) >= self.sequence_length:
            sequence = np.array(self.input_buffer[-self.sequence_length:])
            return sequence
        return None
    
    def save_model(self, path):
        """Save the trained model and label encoder"""
        if self.model is None:
            raise ValueError("No model to save")
        
        torch.save(self.model.state_dict(), path / 'lstm_model.pth')
        with open(path / 'label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a trained model"""
        if self.model is None:
            self.initialize_model()
        
        self.model.load_state_dict(torch.load(path / 'lstm_model.pth', map_location='cpu'))
        with open(path / 'label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
        self.model.eval()
        logger.info(f"Model loaded from {path}")
    
    def generate_sample_data(self, num_samples=1000):
        """
        Generate synthetic data for testing
        
        This creates simulated keypoint sequences for different activities
        """
        import random
        
        X = []
        y = []
        
        for _ in range(num_samples):
            # Randomly select activity
            activity = random.choice(self.classes)
            
            # Generate synthetic keypoints based on activity
            sequence = []
            for _ in range(self.sequence_length):
                # Base keypoints with slight variations
                keypoints = np.random.randn(33, 3) * 0.1
                
                # Adjust keypoints based on activity
                if activity == 'walking':
                    # Simulate walking motion
                    step = _ % 20
                    keypoints[self.KEYPOINT_INDICES['left_ankle']] = [0.4 + 0.1 * np.sin(step * 0.5), 0.5, 0]
                    keypoints[self.KEYPOINT_INDICES['right_ankle']] = [0.6 + 0.1 * np.cos(step * 0.5), 0.5, 0]
                elif activity == 'running':
                    # Simulate running motion
                    step = _ % 10
                    keypoints[self.KEYPOINT_INDICES['left_knee']] = [0.4, 0.3 + 0.2 * np.sin(step * 2), 0]
                    keypoints[self.KEYPOINT_INDICES['right_knee']] = [0.6, 0.3 + 0.2 * np.cos(step * 2), 0]
                elif activity == 'sitting':
                    # Simulate sitting position
                    keypoints[self.KEYPOINT_INDICES['left_hip']] = [0.4, 0.6, 0]
                    keypoints[self.KEYPOINT_INDICES['right_hip']] = [0.6, 0.6, 0]
                    keypoints[self.KEYPOINT_INDICES['left_knee']] = [0.35, 0.8, 0]
                    keypoints[self.KEYPOINT_INDICES['right_knee']] = [0.65, 0.8, 0]
                elif activity == 'falling':
                    # Simulate falling
                    if _ < self.sequence_length // 2:
                        # Falling down
                        keypoints[self.KEYPOINT_INDICES['left_shoulder']] = [0.4, 0.1 + 0.2 * (_ / self.sequence_length), 0]
                        keypoints[self.KEYPOINT_INDICES['right_shoulder']] = [0.6, 0.1 + 0.2 * (_ / self.sequence_length), 0]
                # standing is default (no specific adjustment)
                
                sequence.append(keypoints.flatten())
            
            X.append(sequence)
            y.append(activity)
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        return X, y

# Note: Define KEYPOINT_INDICES for the class
ActivityRecognizer.KEYPOINT_INDICES = {
    'nose': 0,
    'left_shoulder': 5, 'right_shoulder': 6,
    'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10,
    'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14,
    'left_ankle': 15, 'right_ankle': 16
}