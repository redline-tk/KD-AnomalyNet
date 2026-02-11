import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Optional
from dataclasses import dataclass
from scipy.stats import entropy


@dataclass 
class DiffusionDiagnostics:
    # Stability metrics
    mean_prediction_stability: float
    boundary_stability: float
    feature_sensitivity: np.ndarray
    
    # Confidence metrics
    confidence_calibration: float
    uncertainty_correlation: float
    
    # Detailed results
    per_sample_stability: np.ndarray
    noise_level_results: dict
    
    # Interpretations
    findings: list[str]
    recommendations: list[str]


class DiffusionProbe:    
    def __init__(
        self,
        noise_schedule: list[float] = None,
        n_samples_per_level: int = 20,
        feature_probe_dims: int = 5
    ):
        self.noise_schedule = noise_schedule or [0.01, 0.05, 0.1, 0.2, 0.3]
        self.n_samples_per_level = n_samples_per_level
        self.feature_probe_dims = feature_probe_dims

    def diagnose(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        X: np.ndarray,
        y_true: np.ndarray,
        original_scores: np.ndarray
    ) -> DiffusionDiagnostics:
        
        n_samples, n_features = X.shape
        feature_std = X.std(axis=0) + 1e-10
        
        noise_results = {}
        all_stabilities = []
        
        for sigma in self.noise_schedule:
            level_result = self._analyze_noise_level(
                predict_fn, X, original_scores, feature_std, sigma
            )
            noise_results[sigma] = level_result
            all_stabilities.append(level_result['per_sample_std'])
        
        # Aggregate stability across noise levels
        per_sample_stability = np.mean(all_stabilities, axis=0)
        mean_stability = 1.0 - per_sample_stability.mean()  
        
        boundary_mask = np.abs(original_scores - 0.5) < 0.2
        if boundary_mask.sum() > 0:
            boundary_stability = 1.0 - per_sample_stability[boundary_mask].mean()
        else:
            boundary_stability = mean_stability
        
        feature_sensitivity = self._analyze_feature_sensitivity(
            predict_fn, X, original_scores, feature_std
        )
        
        confidence = np.abs(original_scores - 0.5) * 2 
        stability_scores = 1 - per_sample_stability
        
        if len(confidence) > 1:
            uncertainty_corr = np.corrcoef(confidence, stability_scores)[0, 1]
        else:
            uncertainty_corr = 0.0
        
        confident_mask = confidence > 0.5
        if confident_mask.sum() > 0:
            confident_preds = (original_scores[confident_mask] > 0.5).astype(int)
            confident_labels = y_true[confident_mask]
            confidence_calibration = (confident_preds == confident_labels).mean()
        else:
            confidence_calibration = 0.5
        
        findings, recommendations = self._generate_insights(
            mean_stability=mean_stability,
            boundary_stability=boundary_stability,
            uncertainty_corr=uncertainty_corr,
            confidence_calibration=confidence_calibration,
            feature_sensitivity=feature_sensitivity,
            noise_results=noise_results
        )
        
        return DiffusionDiagnostics(
            mean_prediction_stability=float(mean_stability),
            boundary_stability=float(boundary_stability),
            feature_sensitivity=feature_sensitivity,
            confidence_calibration=float(confidence_calibration),
            uncertainty_correlation=float(uncertainty_corr) if not np.isnan(uncertainty_corr) else 0.0,
            per_sample_stability=per_sample_stability,
            noise_level_results=noise_results,
            findings=findings,
            recommendations=recommendations
        )

    def _analyze_noise_level(
        self,
        predict_fn: Callable,
        X: np.ndarray,
        original_scores: np.ndarray,
        feature_std: np.ndarray,
        sigma: float
    ) -> dict:
        n_samples = len(X)
        perturbed_scores = []
        
        for _ in range(self.n_samples_per_level):
            noise = np.random.randn(*X.shape) * feature_std * sigma
            X_noisy = X + noise
            scores = predict_fn(X_noisy)
            perturbed_scores.append(scores)
        
        perturbed_scores = np.array(perturbed_scores)
        
        # Statistics
        score_mean = perturbed_scores.mean(axis=0)
        score_std = perturbed_scores.std(axis=0)
        score_range = perturbed_scores.max(axis=0) - perturbed_scores.min(axis=0)
        
        # Prediction consistency
        original_preds = (original_scores >= 0.5)
        flip_counts = np.zeros(n_samples)
        for scores in perturbed_scores:
            flipped = (scores >= 0.5) != original_preds
            flip_counts += flipped
        flip_rate = flip_counts / self.n_samples_per_level
        
        mean_shift = score_mean - original_scores
        
        return {
            'sigma': sigma,
            'per_sample_std': score_std,
            'mean_std': float(score_std.mean()),
            'max_std': float(score_std.max()),
            'mean_range': float(score_range.mean()),
            'flip_rate': float(flip_rate.mean()),
            'samples_unstable': int((score_std > 0.1).sum()),
            'mean_bias': float(mean_shift.mean()),
            'bias_towards_anomaly': float((mean_shift > 0).mean())
        }

    def _analyze_feature_sensitivity(
        self,
        predict_fn: Callable,
        X: np.ndarray,
        original_scores: np.ndarray,
        feature_std: np.ndarray
    ) -> np.ndarray:

        n_samples, n_features = X.shape
        feature_sensitivity = np.zeros(n_features)
        
        sigma = 0.1  # Fixed noise level for comparison
        
        for f in range(n_features):
            score_changes = []
            
            for _ in range(self.n_samples_per_level // 2):
                X_perturbed = X.copy()
                noise = np.random.randn(n_samples) * feature_std[f] * sigma
                X_perturbed[:, f] += noise
                
                new_scores = predict_fn(X_perturbed)
                change = np.abs(new_scores - original_scores)
                score_changes.append(change.mean())
            
            feature_sensitivity[f] = np.mean(score_changes)
        
        total = feature_sensitivity.sum()
        if total > 0:
            feature_sensitivity /= total
        
        return feature_sensitivity

    def _generate_insights(
        self,
        mean_stability: float,
        boundary_stability: float,
        uncertainty_corr: float,
        confidence_calibration: float,
        feature_sensitivity: np.ndarray,
        noise_results: dict
    ) -> tuple[list[str], list[str]]:
        findings = []
        recommendations = []
        
        if mean_stability > 0.9:
            findings.append(
                f"Model shows high prediction stability ({mean_stability:.1%}). "
                "Predictions are robust to input perturbations."
            )
        elif mean_stability > 0.7:
            findings.append(
                f"Model shows moderate stability ({mean_stability:.1%}). "
                "Some sensitivity to noise exists."
            )
        else:
            findings.append(
                f"Model shows LOW stability ({mean_stability:.1%}). "
                "Predictions are highly sensitive to noise."
            )
            recommendations.append(
                "Consider noise augmentation during training or "
                "increasing model capacity for more robust representations."
            )
        
        if boundary_stability < mean_stability - 0.1:
            findings.append(
                f"Boundary regions are less stable ({boundary_stability:.1%}) "
                f"than average ({mean_stability:.1%}). Decision boundary is uncertain."
            )
            recommendations.append(
                "Focus on boundary-aware training or calibration to "
                "improve predictions near the decision threshold."
            )

        if uncertainty_corr > 0.5:
            findings.append(
                f"Good uncertainty-stability correlation ({uncertainty_corr:.2f}). "
                "Model is appropriately uncertain where predictions are unstable."
            )
        elif uncertainty_corr < 0.2:
            findings.append(
                f"Poor uncertainty correlation ({uncertainty_corr:.2f}). "
                "Model confidence does not reflect prediction stability."
            )
            recommendations.append(
                "Model may be overconfident. Consider calibration techniques "
                "or uncertainty-aware training."
            )
        
        if confidence_calibration > 0.85:
            findings.append(
                f"Good calibration ({confidence_calibration:.1%}). "
                "Confident predictions are usually correct."
            )
        elif confidence_calibration < 0.7:
            findings.append(
                f"Poor calibration ({confidence_calibration:.1%}). "
                "High-confidence predictions are often wrong."
            )
            recommendations.append(
                "Apply temperature scaling or other calibration methods."
            )
        
        top_features = np.argsort(feature_sensitivity)[-3:][::-1]
        top_sensitivity = feature_sensitivity[top_features]
        
        if top_sensitivity[0] > 0.3:
            findings.append(
                f"Feature {top_features[0]} dominates predictions "
                f"({top_sensitivity[0]:.1%} sensitivity). "
                "Model may be over-relying on single feature."
            )
        
        low_noise = noise_results.get(0.01, {})
        high_noise = noise_results.get(0.2, {})
        
        if low_noise and high_noise:
            flip_increase = high_noise.get('flip_rate', 0) - low_noise.get('flip_rate', 0)
            if flip_increase > 0.2:
                findings.append(
                    f"Prediction flips increase sharply with noise "
                    f"(+{flip_increase:.1%} from σ=0.01 to σ=0.2). "
                    "Model may lack robustness margin."
                )
        
        return findings, recommendations

    def compare_models(
        self,
        predict_fn_a: Callable,
        predict_fn_b: Callable,
        X: np.ndarray,
        y_true: np.ndarray,
        name_a: str = "Model A",
        name_b: str = "Model B"
    ) -> dict:

        scores_a = predict_fn_a(X)
        scores_b = predict_fn_b(X)
        
        diag_a = self.diagnose(predict_fn_a, X, y_true, scores_a)
        diag_b = self.diagnose(predict_fn_b, X, y_true, scores_b)
        
        comparison = {
            name_a: {
                'stability': diag_a.mean_prediction_stability,
                'boundary_stability': diag_a.boundary_stability,
                'calibration': diag_a.confidence_calibration,
                'uncertainty_correlation': diag_a.uncertainty_correlation
            },
            name_b: {
                'stability': diag_b.mean_prediction_stability,
                'boundary_stability': diag_b.boundary_stability,
                'calibration': diag_b.confidence_calibration,
                'uncertainty_correlation': diag_b.uncertainty_correlation
            },
            'differences': {
                'stability': diag_b.mean_prediction_stability - diag_a.mean_prediction_stability,
                'boundary_stability': diag_b.boundary_stability - diag_a.boundary_stability,
                'calibration': diag_b.confidence_calibration - diag_a.confidence_calibration
            },
            'interpretation': []
        }
        
        if comparison['differences']['stability'] < -0.1:
            comparison['interpretation'].append(
                f"{name_b} is less stable than {name_a} under perturbation. "
                "Knowledge transfer may have reduced robustness."
            )
        elif comparison['differences']['stability'] > 0.1:
            comparison['interpretation'].append(
                f"{name_b} is MORE stable than {name_a}. "
                "Distillation may have smoothed the decision boundary."
            )
        
        if comparison['differences']['calibration'] < -0.1:
            comparison['interpretation'].append(
                f"{name_b} is less calibrated than {name_a}. "
                "Consider post-hoc calibration."
            )
        
        return comparison


def create_diffusion_diagnostic_report(
    teacher_predict_fn: Callable,
    student_predict_fn: Callable,
    X: np.ndarray,
    y_true: np.ndarray
) -> dict:

    probe = DiffusionProbe()
    
    teacher_scores = teacher_predict_fn(X)
    student_scores = student_predict_fn(X)
    
    # Individual diagnostics
    teacher_diag = probe.diagnose(teacher_predict_fn, X, y_true, teacher_scores)
    student_diag = probe.diagnose(student_predict_fn, X, y_true, student_scores)
    
    
    comparison = probe.compare_models(
        teacher_predict_fn, student_predict_fn,
        X, y_true,
        name_a="Teacher", name_b="Student"
    )
    
    return {
        'teacher': {
            'stability': teacher_diag.mean_prediction_stability,
            'boundary_stability': teacher_diag.boundary_stability,
            'calibration': teacher_diag.confidence_calibration,
            'findings': teacher_diag.findings,
            'noise_levels': teacher_diag.noise_level_results
        },
        'student': {
            'stability': student_diag.mean_prediction_stability,
            'boundary_stability': student_diag.boundary_stability,
            'calibration': student_diag.confidence_calibration,
            'findings': student_diag.findings,
            'recommendations': student_diag.recommendations,
            'noise_levels': student_diag.noise_level_results
        },
        'comparison': comparison,
        'feature_sensitivity': {
            'teacher': teacher_diag.feature_sensitivity.tolist(),
            'student': student_diag.feature_sensitivity.tolist()
        }
    }
