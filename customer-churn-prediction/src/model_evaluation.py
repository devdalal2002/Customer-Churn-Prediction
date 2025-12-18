"""
Evaluation helpers: metrics, plotting helpers, and convenience wrappers.
"""
from typing import Any, Dict
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve,
                             average_precision_score, accuracy_score, precision_score, recall_score, f1_score)


def evaluate_model(model: Any, X_test, y_test, out_dir: str = 'reports/figures') -> Dict[str, float]:
    """Evaluate model on test set, save plots to out_dir, return metric dict."""
    os.makedirs(out_dir, exist_ok=True)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred)),
    }

    # classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(cm, cmap='Blues')
    ax.set_title('Confusion matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha='center', va='center')
    fig_path = os.path.join(out_dir, 'confusion_matrix.png')
    fig.savefig(fig_path)
    plt.close(fig)

    # ROC curve
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
        ax.plot([0,1],[0,1], linestyle='--', color='gray')
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title('ROC Curve')
        ax.legend(loc='lower right')
        fig.savefig(os.path.join(out_dir, 'roc_curve.png'))
        plt.close(fig)

        # PR curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        fig, ax = plt.subplots()
        ax.plot(recall, precision, label=f'AP = {ap:.3f}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        fig.savefig(os.path.join(out_dir, 'pr_curve.png'))
        plt.close(fig)

    # include classification report in metrics
    metrics.update({f'report_{k}': v for k, v in report.items() if isinstance(v, dict)})
    return metrics


def save_metrics(metrics: Dict[str, float], path: str = 'reports/figures/metrics.txt') -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
