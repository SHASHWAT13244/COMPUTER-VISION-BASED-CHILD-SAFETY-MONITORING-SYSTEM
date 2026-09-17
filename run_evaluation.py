"""
run_evaluation.py - Quick launcher for model evaluation
"""

import subprocess
import sys
import os


def run_evaluation():
    print("\n🚀 Running Model Evaluation...")

    if not os.path.exists('data/training_data.npz'):
        print("❌ Training data not found!")
        print("Run: python data_preparation.py --process")
        return

    if not os.path.exists('saved_models/activity_model.pth'):
        print("⚠️  Model not found! Using untrained model...")

    cmd = [sys.executable, 'evaluate_model.py']
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n❌ Evaluation failed (exit code {result.returncode})")
        return

    print("\n✅ Evaluation complete!")
    print("📄 Results saved to:")
    print("   - evaluation_results.json")
    print("   - evaluation_plots.png")


if __name__ == "__main__":
    run_evaluation()
