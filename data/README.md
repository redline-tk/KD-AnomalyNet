# Data Directory

Place benchmark datasets here in .npz format:
- cardio.npz
- satellite.npz
- thyroid.npz
- mammography.npz

Each file should contain:
- X: feature matrix (n_samples, n_features)
- y: labels (n_samples,) where 1=anomaly, 0=normal
