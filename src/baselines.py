import numpy as np
import time
from dataclasses import dataclass
from typing import Callable, Optional
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import RobustScaler

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
    from pyod.models.lscp import LSCP
    from pyod.models.suod import SUOD
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


class BaselineRunner:

    
    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.scaler = RobustScaler()
        self.results = []

    def get_sklearn_baselines(self) -> dict:

        return {
            'IsolationForest': (
                IsolationForest(contamination=self.contamination, random_state=42, n_jobs=-1),
                'tree'
            ),
            'LocalOutlierFactor': (
                LocalOutlierFactor(contamination=self.contamination, novelty=True, n_jobs=-1),
                'density'
            ),
            'OneClassSVM': (
                OneClassSVM(nu=self.contamination, kernel='rbf'),
                'kernel'
            ),
            'EllipticEnvelope': (
                EllipticEnvelope(contamination=self.contamination, random_state=42),
                'statistical'
            ),
        }

    def get_pyod_baselines(self) -> dict:

        if not PYOD_AVAILABLE:
            return {}
        
        base_detectors = [
            IForest(contamination=self.contamination),
            PyODLOF(contamination=self.contamination),
            HBOS(contamination=self.contamination)
        ]
        
        return {
            'ECOD': (ECOD(contamination=self.contamination), 'statistical'),
            'COPOD': (COPOD(contamination=self.contamination), 'statistical'),
            'HBOS': (HBOS(contamination=self.contamination, n_bins=20), 'histogram'),
            'KNN': (KNN(contamination=self.contamination, n_neighbors=10), 'distance'),
            'PyOD_LOF': (PyODLOF(contamination=self.contamination), 'density'),
            'PyOD_IForest': (IForest(contamination=self.contamination), 'tree'),
            'PyOD_OCSVM': (OCSVM(contamination=self.contamination), 'kernel'),
            'PyOD_PCA': (PyODPCA(contamination=self.contamination), 'linear'),
            'MCD': (MCD(contamination=self.contamination), 'statistical'),
            'CBLOF': (CBLOF(contamination=self.contamination, n_clusters=8), 'clustering'),
            'FeatureBagging': (
                FeatureBagging(contamination=self.contamination, n_estimators=10),
                'ensemble'
            ),
            'LSCP': (
                LSCP(detector_list=base_detectors, contamination=self.contamination),
                'ensemble'
            ),
        }

    def _get_scores(self, model, X_train: np.ndarray, X_test: np.ndarray, name: str) -> np.ndarray:

        if hasattr(model, 'decision_function'):
            scores = -model.decision_function(X_test)
        elif hasattr(model, 'score_samples'):
            scores = -model.score_samples(X_test)
        else:
            scores = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        if scores is None:
            raise ValueError(f"Cannot extract scores from {name}")
        
        # Normalize to [0, 1]
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        return scores

    def run_baseline(
        self,
        name: str,
        model,
        category: str,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> BaselineResult:
                       
        start = time.time()
        model.fit(X_train)
        train_time = time.time() - start
        

        start = time.time()
        scores = self._get_scores(model, X_train, X_test, name)
        inference_time = time.time() - start
        

        metrics = compute_metrics(y_test, scores)
        

        n_params = None
        if hasattr(model, 'n_estimators'):
            n_params = model.n_estimators * 100  # rough estimate
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
        

        for name, (model, category) in self.get_sklearn_baselines().items():
            try:
                print(f"    Running {name}...")
                result = self.run_baseline(
                    name, model, category, X_train_scaled, X_test_scaled, y_test
                )
                results.append(result)
            except Exception as e:
                print(f"    {name} failed: {e}")
        

        for name, (model, category) in self.get_pyod_baselines().items():
            if not include_slow and name in ['LSCP', 'FeatureBagging', 'SUOD']:
                continue
            try:
                print(f"    Running {name}...")
                result = self.run_baseline(
                    name, model, category, X_train_scaled, X_test_scaled, y_test
                )
                results.append(result)
            except Exception as e:
                print(f"    {name} failed: {e}")
        
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


class DistillationBaselines:

    
    @staticmethod
    def hinton_distillation(
        teacher_scores: np.ndarray,
        X_train: np.ndarray,
        X_test: np.ndarray,
        temperature: float = 3.0,
        epochs: int = 100
    ) -> np.ndarray:

        import torch
        import torch.nn as nn
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_dim = X_train.shape[1]
        

        student = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        ).to(device)
        
        X_tensor = torch.FloatTensor(X_train).to(device)
        y_tensor = torch.FloatTensor(teacher_scores).to(device)
        
        # Soft targets
        soft_targets = torch.sigmoid(y_tensor / temperature)
        
        optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
        dataset = torch.utils.data.TensorDataset(X_tensor, soft_targets)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
        
        student.train()
        for _ in range(epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred = student(batch_x).squeeze()
                loss = nn.functional.binary_cross_entropy(pred, batch_y)
                loss.backward()
                optimizer.step()
        
        student.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            scores = student(X_test_tensor).squeeze().cpu().numpy()
        
        return scores

    @staticmethod
    def feature_matching_distillation(
        teacher_embeddings: np.ndarray,
        X_train: np.ndarray,
        X_test: np.ndarray,
        epochs: int = 100
    ) -> np.ndarray:

        import torch
        import torch.nn as nn
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_dim = X_train.shape[1]
        embed_dim = teacher_embeddings.shape[1] if len(teacher_embeddings.shape) > 1 else 16
        
        class FeatureMatchStudent(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, embed_dim)
                )
                self.scorer = nn.Sequential(
                    nn.Linear(embed_dim, 8),
                    nn.ReLU(),
                    nn.Linear(8, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                embed = self.encoder(x)
                score = self.scorer(embed).squeeze()
                return embed, score
        
        student = FeatureMatchStudent().to(device)
        
        X_tensor = torch.FloatTensor(X_train).to(device)
        embed_tensor = torch.FloatTensor(teacher_embeddings).to(device)
        
        optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
        dataset = torch.utils.data.TensorDataset(X_tensor, embed_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
        
        student.train()
        for _ in range(epochs):
            for batch_x, batch_embed in loader:
                optimizer.zero_grad()
                pred_embed, _ = student(batch_x)
                loss = nn.functional.mse_loss(pred_embed, batch_embed)
                loss.backward()
                optimizer.step()
        
        student.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            _, scores = student(X_test_tensor)
            scores = scores.cpu().numpy()
        
        return scores


def run_baseline_comparison(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    contamination: float = 0.05
) -> dict:

    runner = BaselineRunner(contamination=contamination)
    results = runner.run_all_baselines(X_train, X_test, y_test)
    
    return {
        'baselines': runner.get_results_dict(),
        'best_method': max(results, key=lambda x: x.auc_roc).name,
        'best_auc': max(r.auc_roc for r in results)
    }
