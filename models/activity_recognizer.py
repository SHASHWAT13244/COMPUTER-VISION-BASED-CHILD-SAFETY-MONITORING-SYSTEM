# models/activity_recognizer.py
"""
Activity Recognition using LSTM
Classifies activities from pose keypoint sequences
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMActivityRecognizer(nn.Module):
    """
    LSTM model for activity recognition
    """
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=5, 
                 dropout=0.2, bidirectional=False):
        """
        Args:
            input_size: Number of features per timestep
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            num_classes: Number of activity classes
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMActivityRecognizer, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Initialize hidden state
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Use only the last output
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        
        return out

class ActivityRecognizer:
    """
    Activity Recognition system using LSTM
    """
    
    def __init__(self, sequence_length=30, num_keypoints=33, num_classes=5, 
                 hidden_size=128, num_layers=2, dropout=0.2, bidirectional=False):
        """
        Initialize the activity recognizer
        
        Args:
            sequence_length: Number of frames in a sequence
            num_keypoints: Number of keypoints from MediaPipe
            num_classes: Number of activity classes
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
        """
        self.sequence_length = sequence_length
        self.num_keypoints = num_keypoints
        self.num_classes = num_classes
        self.input_size = num_keypoints * 3  # x, y, z coordinates
        
        self.model = LSTMActivityRecognizer(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        self.model_path = None
        self.sequence_buffer = []
        self.activity_labels = ['walking', 'running', 'sitting', 'falling', 'climbing']
        self.is_trained = False
        
    def load_model(self, model_path):
        """
        Load a trained model
        
        Args:
            model_path: Path to model weights
        """
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.model_path = model_path
            self.is_trained = True
            logger.info(f"Loaded model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def save_model(self, model_path):
        """Save model weights"""
        try:
            torch.save(self.model.state_dict(), model_path)
            logger.info(f"Saved model to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, 
                    learning_rate=0.001, save_path=None):
        """
        Train the LSTM model
        
        Args:
            X_train: Training features (n_samples, sequence_length, features)
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            save_path: Path to save model
            
        Returns:
            training history
        """
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        history = {
            'loss': [], 
            'val_loss': [], 
            'accuracy': [], 
            'val_accuracy': [],
            'best_val_accuracy': 0
        }
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward pass
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += y_batch.size(0)
                    val_correct += (predicted == y_batch).sum().item()
            
            # Record history
            train_acc = 100 * correct / total
            val_acc = 100 * val_correct / val_total
            avg_loss = total_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            history['loss'].append(avg_loss)
            history['val_loss'].append(avg_val_loss)
            history['accuracy'].append(train_acc)
            history['val_accuracy'].append(val_acc)
            
            if val_acc > history['best_val_accuracy']:
                history['best_val_accuracy'] = val_acc
                if save_path:
                    self.save_model(save_path)
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f'Epoch {epoch+1}/{epochs}, '
                    f'Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
                    f'Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%'
                )
        
        self.is_trained = True
        
        if save_path:
            self.model_path = save_path
        
        return history
    
    def predict_activity(self, keypoint_sequence, return_probabilities=False):
        """
        Predict activity from a sequence of keypoints
        
        Args:
            keypoint_sequence: List or array of keypoints (sequence_length, features)
            return_probabilities: Whether to return probability distribution
            
        Returns:
            activity: Predicted activity label
            confidence: Confidence score
            (optional) probabilities: Probability distribution
        """
        if len(keypoint_sequence) < self.sequence_length:
            return None, 0.0, None if return_probabilities else None
        
        # Use the last sequence_length frames
        sequence = keypoint_sequence[-self.sequence_length:]
        
        # Ensure correct shape
        sequence = np.array(sequence)
        if len(sequence.shape) == 2:
            sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(sequence).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            pred_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][pred_idx].item()
            
            activity = self.activity_labels[pred_idx] if pred_idx < len(self.activity_labels) else 'unknown'
            
            if return_probabilities:
                probs = probabilities[0].cpu().numpy()
                return activity, confidence, probs
            
            return activity, confidence
    
    def predict_with_memory(self, keypoints, threshold=0.6):
        """
        Predict activity with memory (maintains state between predictions)
        
        Args:
            keypoints: Single frame keypoints (33, 4)
            threshold: Confidence threshold for prediction
            
        Returns:
            activity: Predicted activity or 'unknown'
            confidence: Confidence score
        """
        if keypoints is None:
            self.sequence_buffer.append(np.zeros(self.input_size))
        else:
            # Use x, y, z coordinates
            keypoints_flat = keypoints[:, :3].flatten()
            self.sequence_buffer.append(keypoints_flat)
        
        # Keep only the last sequence_length frames
        if len(self.sequence_buffer) > self.sequence_length * 2:
            self.sequence_buffer = self.sequence_buffer[-self.sequence_length:]
        
        # Predict if we have enough frames
        if len(self.sequence_buffer) >= self.sequence_length:
            activity, confidence = self.predict_activity(self.sequence_buffer)
            
            if confidence >= threshold:
                return activity, confidence
        
        return 'unknown', 0.0
    
    def reset_buffer(self):
        """Reset the sequence buffer"""
        self.sequence_buffer = []
        logger.debug("Sequence buffer reset")
    
    def get_buffer_size(self):
        """Get current buffer size"""
        return len(self.sequence_buffer)
    
    def get_activity_distribution(self, keypoint_sequence):
        """Get probability distribution over all activities"""
        _, _, probs = self.predict_activity(keypoint_sequence, return_probabilities=True)
        return probs
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test data"""
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X_test).to(self.device)
        y_tensor = torch.LongTensor(y_test).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y_tensor).sum().item()
            accuracy = correct / len(y_test)
            
            # Calculate per-class accuracy
            class_correct = torch.zeros(self.num_classes)
            class_total = torch.zeros(self.num_classes)
            
            for i in range(len(y_test)):
                label = y_tensor[i]
                pred = predicted[i]
                if pred == label:
                    class_correct[label] += 1
                class_total[label] += 1
            
            class_accuracies = class_correct / (class_total + 1e-6)
            
        return {
            'accuracy': accuracy,
            'class_accuracies': class_accuracies.cpu().numpy(),
            'total_samples': len(y_test)
        }