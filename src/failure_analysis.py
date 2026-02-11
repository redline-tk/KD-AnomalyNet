import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.stats import entropy
from typing import Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class FailureCase:
    """A single failure case."""
    index: int
    failure_type: str
    teacher_score: float
    student_score: float
    true_label: int
    severity: float  
    features: Optional[np.ndarray] = None
    explanation: str = ""


@dataclass
class FailureModeReport:

    total_samples: int
    total_failures: int
    failure_rate: float
    

    collapsed_anomalies: list[FailureCase] = field(default_factory=list)
    hallucinated_anomalies: list[FailureCase] = field(default_factory=list)
    boundary_confusions: list[FailureCase] = field(default_factory=list)
    confidence_failures: list[FailureCase] = field(default_factory=list)
    

    collapse_rate: float = 0.0
    hallucination_rate: float = 0.0
    boundary_confusion_rate: float = 0.0
    

    mean_severity: float = 0.0
    critical_failures: int = 0  # Severity > 0.8


class FailureModeAnalyzer:

    def __init__(
        self,
        decision_threshold: float = 0.5,
        boundary_width: float = 0.15,
        severity_threshold: float = 0.8
    ):
        self.decision_threshold = decision_threshold
        self.boundary_width = boundary_width
        self.severity_threshold = severity_threshold

    def analyze(
        self,
        teacher_scores: np.ndarray,
        student_scores: np.ndarray,
        y_true: np.ndarray,
        X: Optional[np.ndarray] = None
    ) -> FailureModeReport:

        n_samples = len(y_true)
        

        teacher_pred = (teacher_scores >= self.decision_threshold).astype(int)
        student_pred = (student_scores >= self.decision_threshold).astype(int)
        

        report = FailureModeReport(
            total_samples=n_samples,
            total_failures=0,
            failure_rate=0.0
        )
        

        for i in range(n_samples):
            failures = self._analyze_sample(
                index=i,
                teacher_score=teacher_scores[i],
                student_score=student_scores[i],
                teacher_pred=teacher_pred[i],
                student_pred=student_pred[i],
                true_label=y_true[i],
                features=X[i] if X is not None else None
            )
            
            for failure in failures:
                if failure.failure_type == 'collapsed_anomaly':
                    report.collapsed_anomalies.append(failure)
                elif failure.failure_type == 'hallucinated_anomaly':
                    report.hallucinated_anomalies.append(failure)
                elif failure.failure_type == 'boundary_confusion':
                    report.boundary_confusions.append(failure)
                elif failure.failure_type == 'confidence_failure':
                    report.confidence_failures.append(failure)
        

        all_failures = (
            report.collapsed_anomalies + 
            report.hallucinated_anomalies + 
            report.boundary_confusions +
            report.confidence_failures
        )
        
        report.total_failures = len(all_failures)
        report.failure_rate = report.total_failures / n_samples if n_samples > 0 else 0
        
        n_true_anomalies = y_true.sum()
        n_true_normal = n_samples - n_true_anomalies
        
        report.collapse_rate = len(report.collapsed_anomalies) / max(n_true_anomalies, 1)
        report.hallucination_rate = len(report.hallucinated_anomalies) / max(n_true_normal, 1)
        report.boundary_confusion_rate = len(report.boundary_confusions) / n_samples
        
        if all_failures:
            severities = [f.severity for f in all_failures]
            report.mean_severity = np.mean(severities)
            report.critical_failures = sum(1 for s in severities if s > self.severity_threshold)
        
        return report

    def _analyze_sample(
        self,
        index: int,
        teacher_score: float,
        student_score: float,
        teacher_pred: int,
        student_pred: int,
        true_label: int,
        features: Optional[np.ndarray]
    ) -> list[FailureCase]:

        failures = []
        

        if true_label == 1 and teacher_pred == 1 and student_pred == 0:
            severity = teacher_score - student_score  
            failures.append(FailureCase(
                index=index,
                failure_type='collapsed_anomaly',
                teacher_score=teacher_score,
                student_score=student_score,
                true_label=true_label,
                severity=min(severity, 1.0),
                features=features,
                explanation=f"Anomaly collapsed: teacher={teacher_score:.3f}, student={student_score:.3f}"
            ))
        

        if true_label == 0 and student_pred == 1:
   
            severity = student_score - self.decision_threshold
            

            if teacher_pred == 0:
                severity *= 1.5
            
            failures.append(FailureCase(
                index=index,
                failure_type='hallucinated_anomaly',
                teacher_score=teacher_score,
                student_score=student_score,
                true_label=true_label,
                severity=min(severity, 1.0),
                features=features,
                explanation=f"False positive: student={student_score:.3f} on normal sample"
            ))
        

        near_boundary = (
            abs(teacher_score - self.decision_threshold) < self.boundary_width or
            abs(student_score - self.decision_threshold) < self.boundary_width
        )
        
        if near_boundary and teacher_pred != student_pred:

            if true_label == teacher_pred:
                severity = abs(student_score - teacher_score)
            else:
                severity = 0.3 
            
            failures.append(FailureCase(
                index=index,
                failure_type='boundary_confusion',
                teacher_score=teacher_score,
                student_score=student_score,
                true_label=true_label,
                severity=min(severity, 1.0),
                features=features,
                explanation=f"Boundary confusion at t={teacher_score:.3f}, s={student_score:.3f}"
            ))
        

        student_confident = abs(student_score - 0.5) > 0.3 
        student_wrong = student_pred != true_label
        
        if student_confident and student_wrong:
            severity = abs(student_score - 0.5) * 2 
            
            failures.append(FailureCase(
                index=index,
                failure_type='confidence_failure',
                teacher_score=teacher_score,
                student_score=student_score,
                true_label=true_label,
                severity=min(severity, 1.0),
                features=features,
                explanation=f"Confident but wrong: score={student_score:.3f}, label={true_label}"
            ))
        
        return failures

    def get_failure_patterns(self, report: FailureModeReport) -> dict:

        patterns = {}
        

        if report.collapsed_anomalies:
            collapsed_teacher = [f.teacher_score for f in report.collapsed_anomalies]
            collapsed_student = [f.student_score for f in report.collapsed_anomalies]
            patterns['collapsed_anomalies'] = {
                'count': len(report.collapsed_anomalies),
                'teacher_score_mean': np.mean(collapsed_teacher),
                'teacher_score_std': np.std(collapsed_teacher),
                'student_score_mean': np.mean(collapsed_student),
                'student_score_std': np.std(collapsed_student),
                'mean_score_drop': np.mean(np.array(collapsed_teacher) - np.array(collapsed_student))
            }
        
        if report.hallucinated_anomalies:
            halluc_student = [f.student_score for f in report.hallucinated_anomalies]
            halluc_teacher = [f.teacher_score for f in report.hallucinated_anomalies]
            patterns['hallucinated_anomalies'] = {
                'count': len(report.hallucinated_anomalies),
                'student_score_mean': np.mean(halluc_student),
                'teacher_score_mean': np.mean(halluc_teacher),
                'mean_overestimation': np.mean(np.array(halluc_student) - np.array(halluc_teacher))
            }
        

        if report.boundary_confusions:
            boundary_teacher = [f.teacher_score for f in report.boundary_confusions]
            boundary_student = [f.student_score for f in report.boundary_confusions]
            patterns['boundary_confusion'] = {
                'count': len(report.boundary_confusions),
                'in_boundary_region': sum(
                    1 for t, s in zip(boundary_teacher, boundary_student)
                    if abs(t - 0.5) < 0.2 and abs(s - 0.5) < 0.2
                ),
                'mean_disagreement': np.mean(np.abs(np.array(boundary_teacher) - np.array(boundary_student)))
            }
        
        return patterns

    def generate_report_summary(self, report: FailureModeReport) -> dict:

        patterns = self.get_failure_patterns(report)
        
        summary = {
            'overview': {
                'total_samples': report.total_samples,
                'total_failures': report.total_failures,
                'failure_rate_percent': report.failure_rate * 100,
                'critical_failures': report.critical_failures
            },
            'failure_breakdown': {
                'collapsed_anomalies': {
                    'count': len(report.collapsed_anomalies),
                    'rate_of_anomalies': report.collapse_rate * 100,
                    'description': 'Anomalies detected by teacher but missed by student'
                },
                'hallucinated_anomalies': {
                    'count': len(report.hallucinated_anomalies),
                    'rate_of_normals': report.hallucination_rate * 100,
                    'description': 'Normal samples incorrectly flagged as anomalies by student'
                },
                'boundary_confusions': {
                    'count': len(report.boundary_confusions),
                    'rate': report.boundary_confusion_rate * 100,
                    'description': 'Disagreements near decision boundary'
                },
                'confidence_failures': {
                    'count': len(report.confidence_failures),
                    'description': 'High-confidence incorrect predictions'
                }
            },
            'severity': {
                'mean_severity': report.mean_severity,
                'critical_rate': report.critical_failures / max(report.total_failures, 1) * 100
            },
            'patterns': patterns,
            'recommendations': self._generate_recommendations(report, patterns)
        }
        
        return summary

    def _generate_recommendations(self, report: FailureModeReport, patterns: dict) -> list[str]:

        recommendations = []
        
        if report.collapse_rate > 0.2:
            recommendations.append(
                "High anomaly collapse rate ({:.1f}%). Consider: "
                "1) Increasing student capacity, "
                "2) Adjusting distillation alpha to emphasize anomaly scores, "
                "3) Using harder mining for anomaly samples.".format(report.collapse_rate * 100)
            )
        
        if report.hallucination_rate > 0.1:
            recommendations.append(
                "High hallucination rate ({:.1f}%). Consider: "
                "1) Adding regularization, "
                "2) Reducing temperature to sharpen distributions, "
                "3) Validating teacher labels on edge cases.".format(report.hallucination_rate * 100)
            )
        
        if report.boundary_confusion_rate > 0.15:
            recommendations.append(
                "High boundary confusion ({:.1f}%). Consider: "
                "1) Boundary-aware loss weighting, "
                "2) Calibration fine-tuning, "
                "3) Ensemble smoothing.".format(report.boundary_confusion_rate * 100)
            )
        
        if report.mean_severity > 0.6:
            recommendations.append(
                "High failure severity ({:.2f}). Failures are not just edge cases. "
                "Consider major architectural or training changes.".format(report.mean_severity)
            )
        
        if not recommendations:
            recommendations.append(
                "Failure rates within acceptable bounds. "
                "Focus on incremental improvements and edge case handling."
            )
        
        return recommendations


class BoundaryStabilityAnalyzer:

    
    def __init__(
        self,
        noise_levels: list[float] = None,
        n_perturbations: int = 10
    ):
        self.noise_levels = noise_levels or [0.01, 0.05, 0.1, 0.2]
        self.n_perturbations = n_perturbations

    def analyze_stability(
        self,
        model_predict_fn,
        X: np.ndarray,
        original_scores: np.ndarray
    ) -> dict:

        n_samples = len(X)
        feature_std = X.std(axis=0) + 1e-10
        
        stability_results = {
            'per_noise_level': {},
            'per_sample_stability': np.zeros(n_samples),
            'unstable_samples': [],
            'stable_samples': []
        }
        
        for noise_level in self.noise_levels:
            perturbation_scores = []
            
            for _ in range(self.n_perturbations):
                # Add noise scaled by feature std
                noise = np.random.randn(*X.shape) * feature_std * noise_level
                X_perturbed = X + noise
                
                perturbed_scores = model_predict_fn(X_perturbed)
                perturbation_scores.append(perturbed_scores)
            
            perturbation_scores = np.array(perturbation_scores)
            

            score_std = perturbation_scores.std(axis=0)
            score_range = perturbation_scores.max(axis=0) - perturbation_scores.min(axis=0)
            mean_score = perturbation_scores.mean(axis=0)
            

            pred_flips = np.zeros(n_samples)
            for scores in perturbation_scores:
                original_pred = (original_scores >= 0.5)
                perturbed_pred = (scores >= 0.5)
                pred_flips += (original_pred != perturbed_pred)
            flip_rate = pred_flips / self.n_perturbations
            
            stability_results['per_noise_level'][noise_level] = {
                'mean_score_std': float(score_std.mean()),
                'max_score_std': float(score_std.max()),
                'mean_score_range': float(score_range.mean()),
                'mean_flip_rate': float(flip_rate.mean()),
                'samples_with_flips': int((flip_rate > 0).sum())
            }
            

            stability_results['per_sample_stability'] += score_std
        

        stability_results['per_sample_stability'] /= len(self.noise_levels)
        

        stability_threshold = np.percentile(stability_results['per_sample_stability'], 80)
        
        for i in range(n_samples):
            sample_info = {
                'index': i,
                'stability_score': float(stability_results['per_sample_stability'][i]),
                'original_score': float(original_scores[i]),
                'near_boundary': abs(original_scores[i] - 0.5) < 0.15
            }
            
            if stability_results['per_sample_stability'][i] > stability_threshold:
                stability_results['unstable_samples'].append(sample_info)
            else:
                stability_results['stable_samples'].append(sample_info)
        

        stability_results['summary'] = {
            'mean_stability': float(stability_results['per_sample_stability'].mean()),
            'n_unstable': len(stability_results['unstable_samples']),
            'unstable_rate': len(stability_results['unstable_samples']) / n_samples,
            'boundary_samples_unstable': sum(
                1 for s in stability_results['unstable_samples'] if s['near_boundary']
            )
        }
        
        return stability_results


def create_failure_analysis_report(
    teacher_scores: np.ndarray,
    student_scores: np.ndarray,
    y_true: np.ndarray,
    X: Optional[np.ndarray] = None,
    student_predict_fn=None
) -> dict:

    analyzer = FailureModeAnalyzer()
    report = analyzer.analyze(teacher_scores, student_scores, y_true, X)
    summary = analyzer.generate_report_summary(report)
    

    if student_predict_fn is not None and X is not None:
        stability_analyzer = BoundaryStabilityAnalyzer()
        stability = stability_analyzer.analyze_stability(
            student_predict_fn, X, student_scores
        )
        summary['boundary_stability'] = stability['summary']
        summary['stability_by_noise'] = stability['per_noise_level']
    
    return summary
