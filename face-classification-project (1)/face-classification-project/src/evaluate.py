"""
Model Evaluation Script
Comprehensive evaluation with metrics and visualizations
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import pickle
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_loader import FaceDataLoader


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, model_path: str, encoder_path: str, test_data_dir: str):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to trained model
            encoder_path: Path to label encoder
            test_data_dir: Path to test dataset
        """
        print("=" * 70)
        print("Model Evaluation")
        print("=" * 70)
        
        # Load model
        print(f"\n📦 Loading model from {model_path}...")
        self.model = keras.models.load_model(model_path)
        self.input_size = self.model.input_shape[1:3]
        
        # Load label encoder
        print(f"📦 Loading label encoder from {encoder_path}...")
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        self.num_classes = len(self.label_encoder.classes_)
        
        # Load test data
        print(f"📦 Loading test data from {test_data_dir}...")
        self.loader = FaceDataLoader(
            data_dir=test_data_dir,
            target_size=self.input_size,
            min_images_per_person=1
        )
        
        images, labels, label_names = self.loader.load_dataset()
        self.X_test = self.loader.preprocess_data(images, normalization='imagenet')
        self.y_test = labels
        self.y_test_cat = to_categorical(self.y_test, self.num_classes)
        
        print(f"\n✅ Loaded {len(self.X_test)} test images")
        print(f"✅ Number of classes: {self.num_classes}")
        
        # Results storage
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        
        # Predictions
        self.y_pred = None
        self.y_pred_proba = None
    
    def predict(self):
        """Generate predictions"""
        print("\n🔮 Generating predictions...")
        
        self.y_pred_proba = self.model.predict(self.X_test, verbose=1)
        self.y_pred = np.argmax(self.y_pred_proba, axis=1)
        
        print("✅ Predictions complete")
    
    def evaluate_metrics(self):
        """Calculate evaluation metrics"""
        print("\n📊 Calculating metrics...")
        
        # Basic metrics
        results = self.model.evaluate(self.X_test, self.y_test_cat, verbose=0)
        
        metrics = {}
        for metric_name, value in zip(self.model.metrics_names, results):
            metrics[metric_name] = float(value)
            print(f"   {metric_name}: {value:.4f}")
        
        # Per-class metrics
        report = classification_report(
            self.y_test, self.y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True,
            zero_division=0
        )
        
        # Save metrics
        metrics_file = self.results_dir / 'evaluation_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump({
                'overall_metrics': metrics,
                'per_class_metrics': report
            }, f, indent=4)
        
        print(f"\n✅ Metrics saved to {metrics_file}")
        
        return metrics, report
    
    def plot_confusion_matrix(self, save_path=None):
        """Plot confusion matrix"""
        print("\n📊 Generating confusion matrix...")
        
        # Compute confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_,
                   ax=axes[0])
        axes[0].set_title('Confusion Matrix (Counts)', fontsize=14)
        axes[0].set_ylabel('True Label', fontsize=12)
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        
        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_,
                   ax=axes[1])
        axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14)
        axes[1].set_ylabel('True Label', fontsize=12)
        axes[1].set_xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.results_dir / 'confusion_matrix.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to {save_path}")
        
        plt.close()
    
    def plot_roc_curves(self, save_path=None, max_classes=10):
        """Plot ROC curves for top classes"""
        print(f"\n📊 Generating ROC curves (top {max_classes} classes)...")
        
        # Select classes with most samples
        unique, counts = np.unique(self.y_test, return_counts=True)
        top_classes = unique[np.argsort(counts)[-max_classes:]]
        
        plt.figure(figsize=(12, 10))
        
        for class_idx in top_classes:
            # Binary classification for this class
            y_true_binary = (self.y_test == class_idx).astype(int)
            y_score = self.y_pred_proba[:, class_idx]
            
            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            roc_auc = auc(fpr, tpr)
            
            # Plot
            class_name = self.label_encoder.inverse_transform([class_idx])[0]
            plt.plot(fpr, tpr, lw=2, 
                    label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves (Top Classes)', fontsize=14)
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        
        if save_path is None:
            save_path = self.results_dir / 'roc_curves.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ ROC curves saved to {save_path}")
        
        plt.close()
    
    def plot_top_k_accuracy(self, save_path=None, max_k=10):
        """Plot top-k accuracy"""
        print("\n📊 Calculating top-k accuracy...")
        
        top_k_accuracies = []
        k_values = range(1, min(max_k + 1, self.num_classes + 1))
        
        for k in k_values:
            # Get top-k predictions
            top_k_pred = np.argsort(self.y_pred_proba, axis=1)[:, -k:]
            
            # Check if true label is in top-k
            correct = np.any(top_k_pred == self.y_test[:, np.newaxis], axis=1)
            accuracy = np.mean(correct)
            top_k_accuracies.append(accuracy)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, top_k_accuracies, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('k', fontsize=12)
        plt.ylabel('Top-k Accuracy', fontsize=12)
        plt.title('Top-k Accuracy', fontsize=14)
        plt.grid(alpha=0.3)
        plt.xticks(k_values)
        
        # Add value labels
        for k, acc in zip(k_values, top_k_accuracies):
            plt.text(k, acc + 0.01, f'{acc:.3f}', ha='center', fontsize=9)
        
        if save_path is None:
            save_path = self.results_dir / 'top_k_accuracy.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Top-k accuracy plot saved to {save_path}")
        
        plt.close()
    
    def analyze_errors(self, save_path=None, top_n=10):
        """Analyze most common errors"""
        print(f"\n📊 Analyzing top {top_n} error patterns...")
        
        # Find incorrect predictions
        incorrect_mask = self.y_test != self.y_pred
        incorrect_indices = np.where(incorrect_mask)[0]
        
        if len(incorrect_indices) == 0:
            print("✅ No errors found!")
            return
        
        # Count error pairs
        error_pairs = {}
        for idx in incorrect_indices:
            true_label = self.label_encoder.inverse_transform([self.y_test[idx]])[0]
            pred_label = self.label_encoder.inverse_transform([self.y_pred[idx]])[0]
            key = (true_label, pred_label)
            error_pairs[key] = error_pairs.get(key, 0) + 1
        
        # Get top errors
        sorted_errors = sorted(error_pairs.items(), key=lambda x: x[1], reverse=True)
        top_errors = sorted_errors[:top_n]
        
        # Save error analysis
        error_report = {
            'total_errors': int(np.sum(incorrect_mask)),
            'error_rate': float(np.mean(incorrect_mask)),
            'top_error_pairs': [
                {
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'count': count
                }
                for (true_label, pred_label), count in top_errors
            ]
        }
        
        report_file = self.results_dir / 'error_analysis.json'
        with open(report_file, 'w') as f:
            json.dump(error_report, f, indent=4)
        
        print(f"\n✅ Error analysis saved to {report_file}")
        print(f"\nTop {top_n} Error Patterns:")
        for (true_label, pred_label), count in top_errors:
            print(f"   {true_label} → {pred_label}: {count} times")
    
    def run_full_evaluation(self):
        """Run complete evaluation pipeline"""
        print("\n" + "=" * 70)
        print("Running Full Evaluation")
        print("=" * 70)
        
        # Predict
        self.predict()
        
        # Calculate metrics
        self.evaluate_metrics()
        
        # Generate visualizations
        self.plot_confusion_matrix()
        self.plot_roc_curves()
        self.plot_top_k_accuracy()
        self.analyze_errors()
        
        print("\n" + "=" * 70)
        print("Evaluation Complete!")
        print("=" * 70)
        print(f"\n📁 Results saved to: {self.results_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate face classification model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--encoder', type=str, required=True,
                       help='Path to label encoder')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test dataset')
    
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.model).exists():
        print(f"❌ Error: Model file not found: {args.model}")
        return
    
    if not Path(args.encoder).exists():
        print(f"❌ Error: Encoder file not found: {args.encoder}")
        return
    
    if not Path(args.test_data).exists():
        print(f"❌ Error: Test data directory not found: {args.test_data}")
        return
    
    # Run evaluation
    evaluator = ModelEvaluator(args.model, args.encoder, args.test_data)
    evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()
