"""
view_training_data.py - View and inspect training data
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config


def view_data(data_path='data/training_data.npz'):
    if not os.path.exists(data_path):
        print(f"❌ File not found: {data_path}")
        return

    data = np.load(data_path)
    X = data['X']
    y = data['y']
    classes = list(Config.ACTIVITY_CLASSES)

    print("\n" + "=" * 60)
    print("📊 TRAINING DATA OVERVIEW")
    print("=" * 60)

    print(f"\n📁 File: {data_path}")
    print(f"📏 Total samples: {len(X)}")
    print(f"📐 Feature shape: {X.shape}")

    print(f"\n📊 Class Distribution:")
    print("-" * 40)
    for i, name in enumerate(classes):
        count = int(np.sum(y == i))
        percentage = (count / len(y)) * 100 if len(y) > 0 else 0.0
        bar_len = int(percentage / 2)
        bar = '█' * bar_len + '░' * (50 - bar_len)
        print(f"   {name:10} : {count:3} samples ({percentage:5.1f}%) {bar}")

    meta_path = Path(data_path).with_suffix('.json')
    if meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            print(f"\n📋 Metadata:")
            for key, value in metadata.items():
                print(f"   {key}: {value}")
        except Exception as e:
            print(f"\n⚠️  Could not read metadata: {e}")


if __name__ == "__main__":
    view_data()
