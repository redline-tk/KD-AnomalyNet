
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from typing import Optional
import urllib.request
import os

BENCHMARK_URLS = {
    # Standard ODDS datasets
    'cardio': 'https://www.dropbox.com/s/galg3ihvxklf0qi/cardio.mat?dl=1',
    'satellite': 'https://www.dropbox.com/s/dpzxp8jyr9h93k5/satellite.mat?dl=1',
    'thyroid': 'https://www.dropbox.com/s/bih0e15a0fukftb/thyroid.mat?dl=1',
    'mammography': 'https://www.dropbox.com/s/tq2v4hhwyv17hlk/mammography.mat?dl=1',
    'shuttle': 'https://www.dropbox.com/s/mk8ozgisimfn3dw/shuttle.mat?dl=1',
    'pendigits': 'https://www.dropbox.com/s/1x8rzb4a0lia6t1/pendigits.mat?dl=1',
    'annthyroid': 'https://www.dropbox.com/s/aifk51owxbogwav/annthyroid.mat?dl=1',
    'breastw': 'https://www.dropbox.com/s/g3hlnucj71kfvq4/breastw.mat?dl=1',
    'cover': 'https://www.dropbox.com/s/awx8iuzbu8dkxf1/cover.mat?dl=1',
    'glass': 'https://www.dropbox.com/s/iq3hjxw77gpbl7u/glass.mat?dl=1',
    'ionosphere': 'https://www.dropbox.com/s/lpn4z73fico4uup/ionosphere.mat?dl=1',
    'letter': 'https://www.dropbox.com/s/rt9i95h9jywrtiy/letter.mat?dl=1',
    'lympho': 'https://www.dropbox.com/s/ag469ssk0lmctco/lympho.mat?dl=1',
    'mnist': 'https://www.dropbox.com/s/n3wurjt8v9qi6nc/mnist.mat?dl=1',
    'musk': 'https://www.dropbox.com/s/we6aqhb0m38i60t/musk.mat?dl=1',
    'optdigits': 'https://www.dropbox.com/s/w52ndgz5k75s514/optdigits.mat?dl=1',
    'pima': 'https://www.dropbox.com/s/mvlwu7p0nyk2a2r/pima.mat?dl=1',
    'speech': 'https://www.dropbox.com/s/w6xv51ctea6uzdz/speech.mat?dl=1',
    'vertebral': 'https://www.dropbox.com/s/5kuqb387sgvwmrb/vertebral.mat?dl=1',
    'vowels': 'https://www.dropbox.com/s/pa26odoq6atq9vx/vowels.mat?dl=1',
    'wbc': 'https://www.dropbox.com/s/ebz9v9kdnvykzcb/wbc.mat?dl=1',
    'wine': 'https://www.dropbox.com/s/uvjaudt2uto7zal/wine.mat?dl=1',
}


MAIN_DATASETS = [
    'cardio', 'satellite', 'thyroid', 'mammography', 
    'annthyroid', 'pima',
    'pendigits', 'optdigits', 'vowels', 'musk'
]


def load_dataset(
    name: str,
    data_dir: str = 'data',
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    data_path = Path(data_dir) / f"{name}.npz"
    mat_path = Path(data_dir) / f"{name}.mat"
    
    if data_path.exists():
        data = np.load(data_path)
        X, y = data['X'], data['y']
    elif mat_path.exists():
        X, y = _load_mat(mat_path)
    elif name in BENCHMARK_URLS:
        print(f"Downloading {name} dataset...")
        X, y = _download_and_load(name, data_dir)
    else:
        print(f"Dataset '{name}' not found. Generating realistic synthetic data...")
        X, y = _generate_synthetic(name)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def _load_mat(path: Path) -> tuple[np.ndarray, np.ndarray]:

    from scipy.io import loadmat
    data = loadmat(str(path))
    X = data['X'].astype(np.float32)
    y = data['y'].astype(np.int32).ravel()
    return X, y


def _download_and_load(name: str, data_dir: str) -> tuple[np.ndarray, np.ndarray]:

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    mat_path = Path(data_dir) / f"{name}.mat"
    
    try:
        urllib.request.urlretrieve(BENCHMARK_URLS[name], mat_path)
        X, y = _load_mat(mat_path)
        np.savez(Path(data_dir) / f"{name}.npz", X=X, y=y)
        return X, y
    except Exception as e:
        print(f"Download failed: {e}. Using synthetic data.")
        return _generate_synthetic(name)


def load_csv(
    path: str,
    label_col: Optional[str] = None,
    drop_cols: Optional[list] = None
) -> tuple[np.ndarray, Optional[np.ndarray]]:

    df = pd.read_csv(path)
    
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')
    
    y = None
    if label_col and label_col in df.columns:
        y = df[label_col].values
        df = df.drop(columns=[label_col])
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_cols].fillna(df[numeric_cols].median()).values
    
    return X, y


def _generate_synthetic(
    name: str, 
    n_samples: int = 5000, 
    n_features: int = 20,
    contamination: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:

    np.random.seed(hash(name) % 2**32)
    
    n_anomalies = int(n_samples * contamination)
    n_normal = n_samples - n_anomalies
    n_clusters = 5
    
    X_normal, cluster_labels = make_blobs(
        n_samples=n_normal,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=1.0,
        random_state=42
    )
    
    anomaly_types = np.random.choice(4, size=n_anomalies)
    X_anomaly = np.zeros((n_anomalies, n_features))
    
    for i, atype in enumerate(anomaly_types):
        if atype == 0:  # Local: slightly outside cluster boundary
            cluster_id = np.random.randint(n_clusters)
            cluster_mask = cluster_labels == cluster_id
            cluster_center = X_normal[cluster_mask].mean(axis=0)
            cluster_std = X_normal[cluster_mask].std(axis=0)
            direction = np.random.randn(n_features)
            direction /= np.linalg.norm(direction)
            X_anomaly[i] = cluster_center + direction * cluster_std * np.random.uniform(2.5, 4.0)
        elif atype == 1:  # Global: far from all clusters
            X_anomaly[i] = np.random.randn(n_features) * 4 + np.random.choice([-6, 6], n_features)
        elif atype == 2:  # Cluster boundary: between clusters
            c1, c2 = np.random.choice(n_clusters, 2, replace=False)
            center1 = X_normal[cluster_labels == c1].mean(axis=0)
            center2 = X_normal[cluster_labels == c2].mean(axis=0)
            alpha = np.random.uniform(0.3, 0.7)
            X_anomaly[i] = alpha * center1 + (1 - alpha) * center2 + np.random.randn(n_features) * 0.5
        else:  # Sparse subspace: anomaly in subset of features
            base = X_normal[np.random.randint(n_normal)].copy()
            n_corrupt = max(2, n_features // 4)
            corrupt_dims = np.random.choice(n_features, n_corrupt, replace=False)
            base[corrupt_dims] += np.random.randn(n_corrupt) * 5
            X_anomaly[i] = base
    
    X = np.vstack([X_normal, X_anomaly]).astype(np.float32)
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anomalies)]).astype(np.int32)
    
    shuffle_idx = np.random.permutation(len(X))
    return X[shuffle_idx], y[shuffle_idx]


def get_dataset_info(X: np.ndarray, y: Optional[np.ndarray] = None) -> dict:

    info = {
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'feature_means': X.mean(axis=0).tolist(),
        'feature_stds': X.std(axis=0).tolist()
    }
    if y is not None:
        info['contamination'] = float(y.mean())
        info['n_anomalies'] = int(y.sum())
    return info
