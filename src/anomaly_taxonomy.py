import numpy as np
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Optional
from dataclasses import dataclass
from collections import Counter


@dataclass
class AnomalyProfile:

    index: int
    true_label: int
    predicted_score: float
    anomaly_type: str
    global_distance: float  # Distance to global center
    local_density: float    # Local neighborhood density
    nearest_cluster_dist: float  # Distance to nearest cluster
    isolation_depth: float  # Isolation forest path length
    subspace_score: float   # Anomaly in feature subspaces
    confidence: float       # How confident is the type assignment


class AnomalyTaxonomist:

    ANOMALY_TYPES = [
        'global_outlier',      # Far from everything
        'local_outlier',       # Locally anomalous, globally normal
        'cluster_boundary',    # Near decision boundaries
        'subspace_outlier',    # Anomalous in feature subsets
        'collective_outlier',  # Part of a small anomalous cluster
        'ambiguous'            # Mixed signals, hard to classify
    ]
    
    def __init__(
        self,
        n_neighbors: int = 20,
        n_clusters: int = 5,
        contamination: float = 0.1
    ):
        self.n_neighbors = n_neighbors
        self.n_clusters = n_clusters
        self.contamination = contamination
        self.scaler = StandardScaler()
        
        # Fitted components
        self._global_center = None
        self._cluster_model = None
        self._cluster_centers = None
        self._nn_model = None
        self._lof_model = None
        self._iforest = None
        self._normal_std = None
        self._fitted = False

    def fit(self, X_normal: np.ndarray):

        X_scaled = self.scaler.fit_transform(X_normal)
        
        # Global statistics
        self._global_center = X_scaled.mean(axis=0)
        self._normal_std = X_scaled.std(axis=0) + 1e-10
        
        # Clustering for boundary detection
        n_clusters = min(self.n_clusters, len(X_normal) // 10)
        self._cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self._cluster_model.fit(X_scaled)
        self._cluster_centers = self._cluster_model.cluster_centers_
        
        # Nearest neighbors for local density
        self._nn_model = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(X_normal) - 1))
        self._nn_model.fit(X_scaled)
        
        # LOF for local outlier detection
        self._lof_model = LocalOutlierFactor(
            n_neighbors=min(self.n_neighbors, len(X_normal) - 1),
            contamination=self.contamination,
            novelty=True
        )
        self._lof_model.fit(X_scaled)
        
        # Isolation Forest for isolation depth
        self._iforest = IsolationForest(contamination=self.contamination, random_state=42)
        self._iforest.fit(X_scaled)
        
        self._fitted = True
        return self

    def classify_anomalies(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        scores: Optional[np.ndarray] = None
    ) -> list[AnomalyProfile]:

        if not self._fitted:
            raise RuntimeError("Must call fit() first")
        
        X_scaled = self.scaler.transform(X)
        n_samples = len(X)
        
        # Compute features for classification
        global_distances = self._compute_global_distance(X_scaled)
        local_densities = self._compute_local_density(X_scaled)
        cluster_distances = self._compute_cluster_distance(X_scaled)
        isolation_depths = self._compute_isolation_depth(X_scaled)
        subspace_scores = self._compute_subspace_anomaly(X_scaled)
        lof_scores = -self._lof_model.decision_function(X_scaled)  # Higher = more anomalous
        
        def normalize(arr):
            min_v, max_v = arr.min(), arr.max()
            if max_v - min_v < 1e-10:
                return np.zeros_like(arr)
            return (arr - min_v) / (max_v - min_v)
        
        global_norm = normalize(global_distances)
        local_norm = normalize(lof_scores)
        boundary_norm = 1 - normalize(cluster_distances)  # Closer to boundary = higher
        subspace_norm = normalize(subspace_scores)
        isolation_norm = normalize(-isolation_depths)  # Shorter path = more anomalous
        
        profiles = []
        
        for i in range(n_samples):

            anomaly_type, confidence = self._determine_type(
                global_score=global_norm[i],
                local_score=local_norm[i],
                boundary_score=boundary_norm[i],
                subspace_score=subspace_norm[i],
                isolation_score=isolation_norm[i],
                is_anomaly=(y_true[i] == 1)
            )
            
            profile = AnomalyProfile(
                index=i,
                true_label=int(y_true[i]),
                predicted_score=float(scores[i]) if scores is not None else 0.0,
                anomaly_type=anomaly_type,
                global_distance=float(global_distances[i]),
                local_density=float(local_densities[i]),
                nearest_cluster_dist=float(cluster_distances[i]),
                isolation_depth=float(isolation_depths[i]),
                subspace_score=float(subspace_scores[i]),
                confidence=float(confidence)
            )
            profiles.append(profile)
        
        return profiles

    def _compute_global_distance(self, X_scaled: np.ndarray) -> np.ndarray:

        diff = X_scaled - self._global_center
        return np.sqrt(np.sum((diff / self._normal_std) ** 2, axis=1))

    def _compute_local_density(self, X_scaled: np.ndarray) -> np.ndarray:

        distances, _ = self._nn_model.kneighbors(X_scaled)
        # Average distance to k neighbors (lower = denser)
        return distances.mean(axis=1)

    def _compute_cluster_distance(self, X_scaled: np.ndarray) -> np.ndarray:

        distances = np.zeros((len(X_scaled), len(self._cluster_centers)))
        for i, center in enumerate(self._cluster_centers):
            distances[:, i] = np.linalg.norm(X_scaled - center, axis=1)
        return distances.min(axis=1)

    def _compute_isolation_depth(self, X_scaled: np.ndarray) -> np.ndarray:

        return self._iforest.decision_function(X_scaled)

    def _compute_subspace_anomaly(self, X_scaled: np.ndarray) -> np.ndarray:

        n_features = X_scaled.shape[1]
        n_projections = min(10, n_features * (n_features - 1) // 2)
        
        if n_features < 2:
            return np.zeros(len(X_scaled))
        
        subspace_scores = np.zeros(len(X_scaled))
        
        np.random.seed(42)
        for _ in range(n_projections):
            # Random 2D projection
            dims = np.random.choice(n_features, size=min(2, n_features), replace=False)
            X_proj = X_scaled[:, dims]
            
            # Compute local outlier score in this subspace
            if len(X_proj) > self.n_neighbors:
                nn = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(X_proj) - 1))
                nn.fit(X_proj)
                distances, _ = nn.kneighbors(X_proj)
                subspace_scores += distances.mean(axis=1)
        
        return subspace_scores / n_projections

    def _determine_type(
        self,
        global_score: float,
        local_score: float,
        boundary_score: float,
        subspace_score: float,
        isolation_score: float,
        is_anomaly: bool
    ) -> tuple[str, float]:

        if not is_anomaly:
            return ('normal', 1.0)
        
        # Score thresholds
        HIGH = 0.7
        MEDIUM = 0.4
        
        scores = {
            'global_outlier': global_score * 0.5 + isolation_score * 0.5,
            'local_outlier': local_score * 0.6 + (1 - global_score) * 0.4,
            'cluster_boundary': boundary_score * 0.7 + (1 - isolation_score) * 0.3,
            'subspace_outlier': subspace_score * 0.6 + (1 - global_score) * 0.4,
        }
        

        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        

        high_count = sum(1 for s in scores.values() if s > MEDIUM)
        if high_count >= 3 or best_score < MEDIUM:
            return ('ambiguous', best_score)
        
        return (best_type, best_score)

    def get_type_distribution(self, profiles: list[AnomalyProfile]) -> dict:

        anomaly_profiles = [p for p in profiles if p.true_label == 1]
        type_counts = Counter(p.anomaly_type for p in anomaly_profiles)
        total = len(anomaly_profiles) if anomaly_profiles else 1
        
        return {
            'counts': dict(type_counts),
            'percentages': {k: v / total * 100 for k, v in type_counts.items()},
            'total_anomalies': len(anomaly_profiles)
        }

    def analyze_transfer_by_type(
        self,
        profiles: list[AnomalyProfile],
        teacher_scores: np.ndarray,
        student_scores: np.ndarray,
        threshold: float = 0.5
    ) -> dict:

        results = {}
        
        for atype in self.ANOMALY_TYPES:
            type_profiles = [p for p in profiles if p.anomaly_type == atype and p.true_label == 1]
            
            if not type_profiles:
                continue
            
            indices = [p.index for p in type_profiles]
            t_scores = teacher_scores[indices]
            s_scores = student_scores[indices]
            

            teacher_detected = t_scores > threshold
            student_detected = s_scores > threshold
            

            teacher_rate = teacher_detected.mean()
            student_rate = student_detected.mean()
            

            if teacher_detected.sum() > 0:
                transfer_rate = (teacher_detected & student_detected).sum() / teacher_detected.sum()
            else:
                transfer_rate = 0.0
            

            if len(t_scores) > 1:
                correlation = np.corrcoef(t_scores, s_scores)[0, 1]
            else:
                correlation = 0.0
            

            score_diff = s_scores - t_scores
            mean_degradation = score_diff.mean()
            
            results[atype] = {
                'count': len(type_profiles),
                'teacher_detection_rate': float(teacher_rate),
                'student_detection_rate': float(student_rate),
                'transfer_rate': float(transfer_rate),
                'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
                'mean_score_degradation': float(mean_degradation),
                'detection_loss': float(teacher_rate - student_rate)
            }
        
        return results


def create_taxonomy_report(
    X_train_normal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    teacher_scores: np.ndarray,
    student_scores: np.ndarray
) -> dict:

    taxonomist = AnomalyTaxonomist()
    taxonomist.fit(X_train_normal)
    
    profiles = taxonomist.classify_anomalies(X_test, y_test, student_scores)
    
    distribution = taxonomist.get_type_distribution(profiles)
    transfer_analysis = taxonomist.analyze_transfer_by_type(
        profiles, teacher_scores, student_scores
    )
    

    if transfer_analysis:
        best_type = max(transfer_analysis.items(), key=lambda x: x[1]['transfer_rate'])
        worst_type = min(transfer_analysis.items(), key=lambda x: x[1]['transfer_rate'])
    else:
        best_type = worst_type = (None, {})
    
    return {
        'distribution': distribution,
        'transfer_by_type': transfer_analysis,
        'best_transfer_type': {
            'type': best_type[0],
            'transfer_rate': best_type[1].get('transfer_rate', 0)
        },
        'worst_transfer_type': {
            'type': worst_type[0],
            'transfer_rate': worst_type[1].get('transfer_rate', 0)
        },
        'profiles': profiles
    }
