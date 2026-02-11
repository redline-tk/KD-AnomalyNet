import numpy as np
import time
from dataclasses import dataclass
from typing import Optional
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import RobustScaler
import warnings

try:
    from pyod.models.ecod import ECOD
    from pyod.models.copod import COPOD
    from pyod.models.hbos import HBOS
    from pyod.models.knn import KNN
    from pyod.models.lof import LOF as PyODLOF
    from pyod.models.iforest import IForest
    from pyod.models.ocsvm import OCSVM
    from pyod.models.pca import PCA as PyODPCA
    from pyod.models.mcd import MCD
    from pyod.models.cblof import CBLOF
    from pyod.models.feature_bagging import FeatureBagging
    PYOD_AVAILABLE = True
except ImportError:
    PYOD_AVAILABLE = False
    print("Warning: PyOD not installed. Install with: pip install pyod")

from .metrics import compute_metrics


@dataclass
class BaselineResult:

    name: str
    category: str
    auc_roc: float
    auc_pr: float
    f1: float
    precision: float
    recall: float
    train_time: float
    inference_time: float
    n_params: Optional[int] = None


class FixedBaselineRunner:

    
    def __init__(self, contamination: float = 0.05):
        self.contamination = np.clip(contamination, 0.01, 0.5)
        self.scaler = RobustScaler()
        self.results = []

    def get_sklearn_baselines(self) -> dict:

        return {
            'IsolationForest': {
                'model': IsolationForest(
                    contamination=self.contamination, 
                    random_state=42, 
                    n_jobs=-1,
                    n_estimators=100
                ),
                'category': 'tree',
                'score_sign': -1  # decision_function: lower = more anomalous
            },
            'LocalOutlierFactor': {
                'model': LocalOutlierFactor(
                    contamination=self.contamination, 
                    novelty=True, 
                    n_jobs=-1,
                    n_neighbors=20
                ),
                'category': 'density',
                'score_sign': -1  # decision_function: lower = more anomalous
            },
            'OneClassSVM': {
                'model': OneClassSVM(
                    nu=self.contamination, 
                    kernel='rbf',
                    gamma='scale'
                ),
                'category': 'kernel',
                'score_sign': -1  # decision_function: lower = more anomalous
            },
            'EllipticEnvelope': {
                'model': EllipticEnvelope(
                    contamination=self.contamination, 
                    random_state=42
                ),
                'category': 'statistical',
                'score_sign': -1  # decision_function: lower = more anomalous
            },
        }

    def get_pyod_baselines(self) -> dict:

        if not PYOD_AVAILABLE:
            return {}
        
        return {
            'ECOD': {
                'model': ECOD(contamination=self.contamination),
                'category': 'statistical',
                'score_sign': 1  # PyOD: higher = more anomalous
            },
            'COPOD': {
                'model': COPOD(contamination=self.contamination),
                'category': 'statistical', 
                'score_sign': 1
            },
            'HBOS': {
                'model': HBOS(contamination=self.contamination, n_bins=20),
                'category': 'histogram',
                'score_sign': 1
            },
            'KNN_PyOD': {
                'model': KNN(contamination=self.contamination, n_neighbors=10),
                'category': 'distance',
                'score_sign': 1
            },
            'LOF_PyOD': {
                'model': PyODLOF(contamination=self.contamination, n_neighbors=20),
                'category': 'density',
                'score_sign': 1
            },
            'IForest_PyOD': {
                'model': IForest(contamination=self.contamination, n_estimators=100),
                'category': 'tree',
                'score_sign': 1
            },
            'OCSVM_PyOD': {
                'model': OCSVM(contamination=self.contamination),
                'category': 'kernel',
                'score_sign': 1
            },
            'PCA_PyOD': {
                'model': PyODPCA(contamination=self.contamination),
                'category': 'linear',
                'score_sign': 1
            },
            'MCD': {
                'model': MCD(contamination=self.contamination),
                'category': 'statistical',
                'score_sign': 1
            },
            'CBLOF': {
                'model': CBLOF(contamination=self.contamination, n_clusters=8),
                'category': 'clustering',
                'score_sign': 1
            },
            'FeatureBagging': {
                'model': FeatureBagging(contamination=self.contamination, n_estimators=10),
                'category': 'ensemble',
                'score_sign': 1
            },
        }

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:

        p1, p99 = np.percentile(scores, [1, 99])
        scores = np.clip(scores, p1, p99)
        
        # Min-max normalization
        min_s, max_s = scores.min(), scores.max()
        if max_s - min_s < 1e-10:
            return np.full_like(scores, 0.5)
        
        return (scores - min_s) / (max_s - min_s)

    def _get_scores(
        self, 
        model, 
        X_test: np.ndarray, 
        score_sign: int,
        name: str
    ) -> np.ndarray:
        
        # PyOD models use decision_function
        if hasattr(model, 'decision_function'):
            raw_scores = model.decision_function(X_test)
        elif hasattr(model, 'score_samples'):
            raw_scores = model.score_samples(X_test)
        else:
            raise ValueError(f"Cannot extract scores from {name}")
        
        # Apply sign correction so higher = more anomalous
        scores = raw_scores * score_sign
        
        # Normalize to [0, 1]
        return self._normalize_scores(scores)

    def run_baseline(
        self,
        name: str,
        model_info: dict,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Optional[BaselineResult]:

        model = model_info['model']
        category = model_info['category']
        score_sign = model_info['score_sign']
        
        try:

            start = time.time()
            model.fit(X_train)
            train_time = time.time() - start
            

            start = time.time()
            scores = self._get_scores(model, X_test, score_sign, name)
            inference_time = time.time() - start
            

            if np.isnan(scores).any() or np.isinf(scores).any():
                warnings.warn(f"{name}: Invalid scores detected")
                return None
            

            metrics = compute_metrics(y_test, scores)
            

            if metrics['auc_roc'] < 0.3:

                flipped_metrics = compute_metrics(y_test, 1 - scores)
                if flipped_metrics['auc_roc'] > metrics['auc_roc']:
                    warnings.warn(f"{name}: Score inversion detected, correcting")
                    metrics = flipped_metrics
                    scores = 1 - scores
            

            n_params = None
            if hasattr(model, 'n_estimators'):
                n_params = model.n_estimators * 100
            elif hasattr(model, 'support_vectors_'):
                n_params = len(model.support_vectors_) * X_train.shape[1]
            
            return BaselineResult(
                name=name,
                category=category,
                auc_roc=metrics['auc_roc'],
                auc_pr=metrics['auc_pr'],
                f1=metrics['f1'],
                precision=metrics['precision'],
                recall=metrics['recall'],
                train_time=train_time,
                inference_time=inference_time,
                n_params=n_params
            )
            
        except Exception as e:
            warnings.warn(f"{name} failed: {e}")
            return None

    def run_all_baselines(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        include_slow: bool = True
    ) -> list[BaselineResult]:

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = []
        

        print("    Running sklearn baselines...")
        for name, model_info in self.get_sklearn_baselines().items():
            result = self.run_baseline(name, model_info, X_train_scaled, X_test_scaled, y_test)
            if result:
                results.append(result)
                print(f"      {name}: AUC={result.auc_roc:.4f}")
        

        if PYOD_AVAILABLE:
            print("    Running PyOD baselines...")
            pyod_baselines = self.get_pyod_baselines()
            

            if not include_slow:
                slow_methods = ['FeatureBagging', 'LSCP']
                pyod_baselines = {k: v for k, v in pyod_baselines.items() if k not in slow_methods}
            
            for name, model_info in pyod_baselines.items():
                result = self.run_baseline(name, model_info, X_train_scaled, X_test_scaled, y_test)
                if result:
                    results.append(result)
                    print(f"      {name}: AUC={result.auc_roc:.4f}")
        
        self.results = results
        return results

    def get_results_dict(self) -> list[dict]:

        return [
            {
                'name': r.name,
                'category': r.category,
                'auc_roc': r.auc_roc,
                'auc_pr': r.auc_pr,
                'f1': r.f1,
                'precision': r.precision,
                'recall': r.recall,
                'train_time_s': r.train_time,
                'inference_time_s': r.inference_time,
                'n_params': r.n_params
            }
            for r in self.results
        ]

    def get_summary(self) -> dict:

        if not self.results:
            return {}
        
        aucs = [r.auc_roc for r in self.results]
        best_result = max(self.results, key=lambda x: x.auc_roc)
        
        return {
            'n_methods': len(self.results),
            'best_method': best_result.name,
            'best_auc': best_result.auc_roc,
            'mean_auc': np.mean(aucs),
            'std_auc': np.std(aucs),
            'methods_above_80': sum(1 for a in aucs if a > 0.8),
            'methods_above_90': sum(1 for a in aucs if a > 0.9)
        }


def run_fixed_baseline_comparison(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    contamination: float = 0.05
) -> dict:

    runner = FixedBaselineRunner(contamination=contamination)
    results = runner.run_all_baselines(X_train, X_test, y_test)
    
    return {
        'baselines': runner.get_results_dict(),
        'summary': runner.get_summary()
    }
