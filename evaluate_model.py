"""
evaluate_model.py - Model Performance Evaluation
Calculates Accuracy, Precision, Recall, F1-Score with visualizations
"""

import os
import sys
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
)
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config
from models.activity_recognizer import ActivityRecognizer


def evaluate_model(data_path='data/training_data.npz',
                   model_path='saved_models/activity_model.pth'):
    print("\n" + "=" * 70)
    print("📊 MODEL PERFORMANCE EVALUATION")
    print("=" * 70)

    if not os.path.exists(data_path):
        print(f"❌ Data not found: {data_path}")
        return None

    data = np.load(data_path)
    X = data['X']
    y = data['y']

    print(f"\n📁 Loaded {len(X)} samples")
    print(f"📐 Feature shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.TRAIN_TEST_SPLIT,
        random_state=Config.RANDOM_SEED, stratify=y
    )

    print(f"📚 Training: {len(X_train)} samples")
    print(f"📚 Testing:  {len(X_test)} samples")

    recognizer = ActivityRecognizer(
        sequence_length=Config.SEQUENCE_LENGTH,
        num_keypoints=33,
        num_classes=len(Config.ACTIVITY_CLASSES)
    )

    if not os.path.exists(model_path):
        print(f"⚠️  Model not found: {model_path}")
        print("Using untrained model...")
    else:
        try:
            recognizer.load_model(model_path)
            print(f"✅ Model loaded from: {model_path}")
        except Exception as e:
            print(f"⚠️  Failed to load model ({e}). Using untrained model...")

    print("\n🔮 Making predictions...")
    predictions = []
    confidences = []
    dropped = 0

    for seq in X_test:
        activity, confidence = recognizer.predict_activity(seq)
        if activity in recognizer.activity_labels:
            activity_idx = recognizer.activity_labels.index(activity)
        else:
            activity_idx = -2
            dropped += 1
        predictions.append(activity_idx)
        confidences.append(confidence)

    predictions = np.array(predictions)
    y_test = np.array(y_test)
    all_classes = recognizer.activity_labels

    if dropped:
        print(f"\n⚠️  {dropped} prediction(s) referenced unknown classes.")
        print("   Counting them as incorrect predictions.")
        unique_present = np.unique(y_test)
        if len(unique_present) > 0:
            wrong_label = int(unique_present[0])
            predictions = np.where(predictions == -2, wrong_label, predictions)

    present_classes = np.unique(y_test)
    present_class_names = [all_classes[i] for i in present_classes]

    print("\n" + "=" * 70)
    print("📊 PERFORMANCE METRICS")
    print("=" * 70)

    accuracy = accuracy_score(y_test, predictions)
    print(f"\n🎯 Overall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    precision = precision_score(y_test, predictions, average=None,
                                zero_division=0, labels=present_classes)
    recall = recall_score(y_test, predictions, average=None,
                          zero_division=0, labels=present_classes)
    f1 = f1_score(y_test, predictions, average=None,
                  zero_division=0, labels=present_classes)

    print(f"\n📈 Per-Class Metrics:")
    print("-" * 70)
    print(f"{'Class':<12} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>10}")
    print("-" * 70)

    supports = []
    for i, name in enumerate(present_class_names):
        original_idx = all_classes.index(name)
        support = int(np.sum(y_test == original_idx))
        supports.append(support)
        print(f"{name:<12} {precision[i]:>12.4f} {recall[i]:>12.4f} "
              f"{f1[i]:>12.4f} {support:>10}")

    macro_precision = float(np.mean(precision)) if len(precision) else 0.0
    macro_recall    = float(np.mean(recall))    if len(recall)    else 0.0
    macro_f1        = float(np.mean(f1))        if len(f1)        else 0.0

    print("-" * 70)
    print(f"{'Macro Avg':<12} {macro_precision:>12.4f} {macro_recall:>12.4f} "
          f"{macro_f1:>12.4f}")

    weighted_precision = precision_score(y_test, predictions,
                                         average='weighted', zero_division=0)
    weighted_recall = recall_score(y_test, predictions,
                                   average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_test, predictions,
                           average='weighted', zero_division=0)

    print(f"{'Weighted Avg':<12} {weighted_precision:>12.4f} "
          f"{weighted_recall:>12.4f} {weighted_f1:>12.4f}")

    cm = confusion_matrix(y_test, predictions, labels=present_classes)
    print(f"\n📊 Confusion Matrix:")
    print("    " + " ".join([f"{c[:4]:>6}" for c in present_class_names]))
    for i, row in enumerate(cm):
        print(f"{present_class_names[i][:4]:<4} "
              + " ".join([f"{val:>6}" for val in row]))

    print(f"\n📋 Classification Report:")
    report = classification_report(
        y_test, predictions,
        labels=present_classes,
        target_names=present_class_names,
        zero_division=0
    )
    print(report)

    results = {
        'accuracy': float(accuracy),
        'per_class': {
            present_class_names[i]: {
                'precision': float(precision[i]),
                'recall':    float(recall[i]),
                'f1':        float(f1[i]),
                'support':   int(supports[i]),
            } for i in range(len(present_class_names))
        },
        'macro': {
            'precision': macro_precision,
            'recall':    macro_recall,
            'f1':        macro_f1,
        },
        'weighted': {
            'precision': float(weighted_precision),
            'recall':    float(weighted_recall),
            'f1':        float(weighted_f1),
        },
        'confusion_matrix': cm.tolist(),
        'class_names':      present_class_names,
        'all_classes':      all_classes,
        'total_samples':    len(X_test),
        'dropped_predictions': dropped,
        'model_path':       model_path,
        'data_path':        data_path,
    }

    with open(Config.EVALUATION_RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to: {Config.EVALUATION_RESULTS_PATH}")

    generate_plots(cm, present_class_names, precision, recall, f1, accuracy,
                   macro_precision, macro_recall, macro_f1, supports,
                   y_test, predictions, all_classes)

    return results


def generate_plots(cm, classes, precision, recall, f1, accuracy,
                   macro_precision, macro_recall, macro_f1, supports,
                   y_test, predictions, all_classes):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax1 = axes[0, 0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax1)
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')

    ax2 = axes[0, 1]
    x = np.arange(len(classes))
    width = 0.25
    ax2.bar(x - width, precision, width, label='Precision', color='#3498db')
    ax2.bar(x,          recall,   width, label='Recall',    color='#2ecc71')
    ax2.bar(x + width,  f1,       width, label='F1-Score',  color='#e74c3c')
    ax2.set_xlabel('Activity Classes')
    ax2.set_ylabel('Score')
    ax2.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, rotation=45, ha='right')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[0, 2]
    metrics_values = [accuracy, macro_precision, macro_recall, macro_f1]
    metrics_names = ['Accuracy', 'Macro\nPrecision', 'Macro\nRecall', 'Macro\nF1']
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    bars = ax3.bar(metrics_names, metrics_values, color=colors)
    ax3.set_ylim(0, 1.1)
    ax3.set_ylabel('Score')
    ax3.set_title('Overall Performance Metrics', fontsize=14, fontweight='bold')
    ax3.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5,
                label='Target (80%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    for bar, val in zip(bars, metrics_values):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

    ax4 = axes[1, 0]
    correct = np.diag(cm)
    ax4.bar(classes, supports, color='#3498db', alpha=0.7,
            label='Total Samples')
    ax4.bar(classes, correct, color='#2ecc71', alpha=0.7,
            label='Correct Predictions')
    ax4.set_xlabel('Activity Classes')
    ax4.set_ylabel('Count')
    ax4.set_title('Class Distribution & Correct Predictions',
                  fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    ax5 = axes[1, 1]
    scatter = ax5.scatter(recall, precision, s=100, c=range(len(classes)),
                          cmap='viridis', alpha=0.7)
    for i, name in enumerate(classes):
        ax5.annotate(name, (recall[i], precision[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=10)
    ax5.plot([0, 1], [1, 0], 'r--', alpha=0.5)
    ax5.set_xlabel('Recall')
    ax5.set_ylabel('Precision')
    ax5.set_title('Precision-Recall per Class', fontsize=14, fontweight='bold')
    ax5.set_xlim(-0.05, 1.05)
    ax5.set_ylim(-0.05, 1.05)
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label='Classes')

    ax6 = axes[1, 2]
    ax6.axis('off')

    weighted_precision = precision_score(y_test, predictions,
                                         average='weighted', zero_division=0)
    weighted_recall = recall_score(y_test, predictions,
                                   average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_test, predictions,
                           average='weighted', zero_division=0)

    if len(classes) > 0:
        best_idx = int(np.argmax(f1))
        worst_idx = int(np.argmin(f1))
        best_class = classes[best_idx]
        worst_class = classes[worst_idx]
        best_f1 = f1[best_idx]
        worst_f1 = f1[worst_idx]
    else:
        best_class = worst_class = "N/A"
        best_f1 = worst_f1 = 0.0

    summary_text = f"""
    📊 PERFORMANCE SUMMARY

    Overall Metrics:
    ───────────────────
    Accuracy:  {accuracy:.2%}

    Macro Averages:
    ───────────────────
    Precision: {macro_precision:.2%}
    Recall:    {macro_recall:.2%}
    F1-Score:  {macro_f1:.2%}

    Weighted Averages:
    ───────────────────
    Precision: {weighted_precision:.2%}
    Recall:    {weighted_recall:.2%}
    F1-Score:  {weighted_f1:.2%}

    ⭐ Best performing class:
    {best_class} (F1: {best_f1:.2%})

    ⚠️  Class needing improvement:
    {worst_class} (F1: {worst_f1:.2%})

    📌 Test samples: {len(y_test)}
    📌 Classes present: {len(classes)}
    """

    ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes,
             fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

    plt.tight_layout()
    plt.savefig(Config.EVALUATION_PLOTS_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Visualization saved to: {Config.EVALUATION_PLOTS_PATH}")


if __name__ == "__main__":
    try:
        results = evaluate_model()
        if results:
            print("\n" + "=" * 70)
            print("✅ EVALUATION COMPLETE")
            print("=" * 70)
            print("📄 Files generated:")
            print(f"   - {Config.EVALUATION_RESULTS_PATH}")
            print(f"   - {Config.EVALUATION_PLOTS_PATH}")
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
