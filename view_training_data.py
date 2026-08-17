"""
view_training_data.py - View and inspect training data
"""

import numpy as np
import json
import os

def view_data(data_path='data/training_data.npz'):
    """View training data details"""
    
    if not os.path.exists(data_path):
        print(f"❌ File not found: {data_path}")
        return
    
    data = np.load(data_path)
    X = data['X']
    y = data['y']
    classes = ['walking', 'running', 'sitting', 'falling', 'climbing']
    
    print("\n" + "="*60)
    print("📊 TRAINING DATA OVERVIEW")
    print("="*60)
    
    print(f"\n📁 File: {data_path}")
    print(f"📏 Total samples: {len(X)}")
    print(f"📐 Feature shape: {X.shape}")
    
    # Class distribution
    print(f"\n📊 Class Distribution:")
    print("-"*40)
    for i, name in enumerate(classes):
        count = np.sum(y == i)
        percentage = (count / len(y)) * 100 if len(y) > 0 else 0
        bar = '█' * int(percentage / 2) + '░' * (50 - int(percentage / 2))
        print(f"   {name:10} : {count:3} samples ({percentage:5.1f}%) {bar}")
    
    # Load metadata
    meta_path = data_path.replace('.npz', '.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        print(f"\n📋 Metadata:")
        for key, value in metadata.items():
            print(f"   {key}: {value}")

if __name__ == "__main__":
    view_data()