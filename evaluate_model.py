"""
evaluate_model.py - Model Performance Evaluation
Calculates Accuracy, Precision, Recall, F1-Score with visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from models.activity_recognizer import ActivityRecognizer

def evaluate_model(data_path='data/training_data.npz', 
                   model_path='saved_models/activity_model.pth'):
    """Evaluate trained model performance"""
    
    print("\n" + "="*70)
    print("📊 MODEL PERFORMANCE EVALUATION")
    print("="*70)
    
    # Load data
    if not os.path.exists(data_path):
        print(f"❌ Data not found: {data_path}")
        return None
    
    data = np.load(data_path)
    X = data['X']
    y = data['y']
    
    print(f"\n📁 Loaded {len(X)} samples")
    print(f"📐 Feature shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.TRAIN_TEST_SPLIT,
        random_state=Config.RANDOM_SEED, stratify=y
    )
    
    print(f"📚 Training: {len(X_train)} samples")
    print(f"📚 Testing: {len(X_test)} samples")
    
    # Load model
    recognizer = ActivityRecognizer(
        sequence_length=Config.SEQUENCE_LENGTH,
        num_keypoints=33,
        num_classes=len(Config.ACTIVITY_CLASSES)
    )
    
    if not os.path.exists(model_path):
        print(f"⚠️  Model not found: {model_path}")
        print("Using untrained model...")
    else:
        recognizer.load_model(model_path)
        print(f"✅ Model loaded from: {model_path}")
    
    # Predict
    print("\n🔮 Making predictions...")
    predictions = []
    confidences = []
    
    for seq in X_test:
        activity, confidence = recognizer.predict_activity(seq)
        activity_idx = recognizer.activity_labels.index(activity) if activity in recognizer.activity_labels else -1
        predictions.append(activity_idx)
        confidences.append(confidence)
    
    predictions = np.array(predictions)
    y_test = np.array(y_test)
    all_classes = recognizer.activity_labels
    
    # Get unique classes present in test data
    present_classes = np.unique(y_test)
    present_class_names = [all_classes[i] for i in present_classes]
    
    # Calculate metrics
    print("\n" + "="*70)
    print("📊 PERFORMANCE METRICS")
    print("="*70)
    
    # 1. Accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"\n🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 2. Per-class metrics - only for present classes
    precision = precision_score(y_test, predictions, average=None, zero_division=0, labels=present_classes)
    recall = recall_score(y_test, predictions, average=None, zero_division=0, labels=present_classes)
    f1 = f1_score(y_test, predictions, average=None, zero_division=0, labels=present_classes)
    
    print(f"\n📈 Per-Class Metrics (only classes present in test data):")
    print("-"*70)
    print(f"{'Class':<12} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>10}")
    print("-"*70)
    
    supports = []
    for i, name in enumerate(present_class_names):
        original_idx = all_classes.index(name)
        support = np.sum(y_test == original_idx)
        supports.append(support)
        print(f"{name:<12} {precision[i]:>12.4f} {recall[i]:>12.4f} {f1[i]:>12.4f} {support:>10}")
    
    # 3. Macro Average (unweighted)
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    print("-"*70)
    print(f"{'Macro Avg':<12} {macro_precision:>12.4f} {macro_recall:>12.4f} {macro_f1:>12.4f}")
    
    # 4. Weighted Average
    weighted_precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
    weighted_recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
    
    print(f"{'Weighted Avg':<12} {weighted_precision:>12.4f} {weighted_recall:>12.4f} {weighted_f1:>12.4f}")
    
    # 5. Confusion Matrix
    cm = confusion_matrix(y_test, predictions, labels=present_classes)
    print(f"\n📊 Confusion Matrix:")
    print("    " + " ".join([f"{c[:4]:>6}" for c in present_class_names]))
    for i, row in enumerate(cm):
        print(f"{present_class_names[i][:4]:<4} " + " ".join([f"{val:>6}" for val in row]))
    
    # 6. Classification Report - use labels parameter to match classes
    print(f"\n📋 Full Classification Report:")
    # Only include classes that are present
    report = classification_report(
        y_test, predictions, 
        labels=present_classes, 
        target_names=present_class_names, 
        zero_division=0
    )
    print(report)
    
    # 7. Also show report for all classes (with zeros for missing classes)
    print(f"\n📋 Classification Report (All Classes):")
    # Create a full report with all classes
    all_present = np.zeros(len(all_classes), dtype=bool)
    all_present[present_classes] = True
    
    # For classes not present, they won't appear in the report
    # So we'll just show the report for present classes
    report_all = classification_report(
        y_test, predictions, 
        labels=present_classes, 
        target_names=present_class_names, 
        zero_division=0
    )
    print(report_all)
    
    # Save results
    results = {
        'accuracy': float(accuracy),
        'per_class': {
            present_class_names[i]: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(supports[i])
            } for i in range(len(present_class_names))
        },
        'macro': {
            'precision': float(macro_precision),
            'recall': float(macro_recall),
            'f1': float(macro_f1)
        },
        'weighted': {
            'precision': float(weighted_precision),
            'recall': float(weighted_recall),
            'f1': float(weighted_f1)
        },
        'confusion_matrix': cm.tolist(),
        'class_names': present_class_names,
        'all_classes': all_classes,
        'total_samples': len(X_test),
        'model_path': model_path,
        'data_path': data_path
    }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to: evaluation_results.json")
    
    # Generate plots
    generate_plots(cm, present_class_names, precision, recall, f1, accuracy, 
                   macro_precision, macro_recall, macro_f1, supports, 
                   y_test, predictions, all_classes)
    
    return results

def generate_plots(cm, classes, precision, recall, f1, accuracy,
                   macro_precision, macro_recall, macro_f1, supports,
                   y_test, predictions, all_classes):
    """Generate visualization plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Confusion Matrix
    ax1 = axes[0, 0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax1)
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # 2. Per-class metrics bar chart
    ax2 = axes[0, 1]
    x = np.arange(len(classes))
    width = 0.25
    
    ax2.bar(x - width, precision, width, label='Precision', color='#3498db')
    ax2.bar(x, recall, width, label='Recall', color='#2ecc71')
    ax2.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')
    
    ax2.set_xlabel('Activity Classes')
    ax2.set_ylabel('Score')
    ax2.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, rotation=45, ha='right')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3)
    
    # 3. Overall metrics gauge
    ax3 = axes[0, 2]
    metrics_values = [accuracy, macro_precision, macro_recall, macro_f1]
    metrics_names = ['Accuracy', 'Macro\nPrecision', 'Macro\nRecall', 'Macro\nF1']
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    
    bars = ax3.bar(metrics_names, metrics_values, color=colors)
    ax3.set_ylim(0, 1.1)
    ax3.set_ylabel('Score')
    ax3.set_title('Overall Performance Metrics', fontsize=14, fontweight='bold')
    ax3.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Target (80%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    for bar, val in zip(bars, metrics_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Class distribution
    ax4 = axes[1, 0]
    correct = np.diag(cm)
    
    ax4.bar(classes, supports, color='#3498db', alpha=0.7, label='Total Samples')
    ax4.bar(classes, correct, color='#2ecc71', alpha=0.7, label='Correct Predictions')
    ax4.set_xlabel('Activity Classes')
    ax4.set_ylabel('Count')
    ax4.set_title('Class Distribution & Correct Predictions', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Precision-Recall scatter
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
    
    # 6. Performance summary text
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Calculate weighted averages for summary
    weighted_precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
    weighted_recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
    
    if len(classes) > 0:
        best_idx = np.argmax(f1)
        worst_idx = np.argmin(f1)
        best_class = classes[best_idx]
        worst_class = classes[worst_idx]
        best_f1 = f1[best_idx]
        worst_f1 = f1[worst_idx]
    else:
        best_class = "N/A"
        worst_class = "N/A"
        best_f1 = 0
        worst_f1 = 0
    
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
    plt.savefig('evaluation_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Visualization saved to: evaluation_plots.png")

if __name__ == "__main__":
    try:
        results = evaluate_model()
        if results:
            print("\n" + "="*70)
            print("✅ EVALUATION COMPLETE")
            print("="*70)
            print("📄 Files generated:")
            print("   - evaluation_results.json")
            print("   - evaluation_plots.png")
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()