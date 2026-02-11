
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix
)
from typing import Optional


def compute_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: Optional[float] = None
) -> dict:

    if threshold is None:
        threshold = np.percentile(y_scores, 95)
    
    y_pred = (y_scores >= threshold).astype(int)
    
    metrics = {
        'auc_roc': roc_auc_score(y_true, y_scores),
        'auc_pr': average_precision_score(y_true, y_scores),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'threshold': threshold
    }
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics['true_positives'] = int(tp)
    metrics['false_positives'] = int(fp)
    metrics['true_negatives'] = int(tn)
    metrics['false_negatives'] = int(fn)
    
    return metrics


def find_optimal_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> float:

    thresholds = np.percentile(y_scores, np.arange(80, 100, 0.5))
    best_f1 = 0
    best_threshold = thresholds[0]
    
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    
    return best_threshold


def compare_models(
    y_true: np.ndarray,
    teacher_scores: np.ndarray,
    student_scores: np.ndarray
) -> dict:

    teacher_metrics = compute_metrics(y_true, teacher_scores)
    student_metrics = compute_metrics(y_true, student_scores)
    
    comparison = {
        'teacher': teacher_metrics,
        'student': student_metrics,
        'auc_roc_delta': student_metrics['auc_roc'] - teacher_metrics['auc_roc'],
        'f1_delta': student_metrics['f1'] - teacher_metrics['f1'],
        'retention_auc': student_metrics['auc_roc'] / teacher_metrics['auc_roc'] * 100,
        'retention_f1': student_metrics['f1'] / max(teacher_metrics['f1'], 1e-10) * 100
    }
    
    return comparison


def score_correlation(teacher_scores: np.ndarray, student_scores: np.ndarray) -> dict:

    pearson = np.corrcoef(teacher_scores, student_scores)[0, 1]
    
    teacher_rank = np.argsort(np.argsort(teacher_scores))
    student_rank = np.argsort(np.argsort(student_scores))
    n = len(teacher_scores)
    spearman = 1 - 6 * np.sum((teacher_rank - student_rank) ** 2) / (n * (n**2 - 1))
    
    return {
        'pearson': pearson,
        'spearman': spearman,
        'mse': np.mean((teacher_scores - student_scores) ** 2),
        'mae': np.mean(np.abs(teacher_scores - student_scores))
    }
