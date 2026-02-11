import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from scipy.stats import zscore


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims[:len(hidden_dims)//2 + 1]:
            encoder_layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU(), nn.BatchNorm1d(h_dim)])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers[:-1])
        
        decoder_layers = []
        for h_dim in hidden_dims[len(hidden_dims)//2 + 1:]:
            decoder_layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU(), nn.BatchNorm1d(h_dim)])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var


class TeacherEnsemble:

    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.scaler = RobustScaler()
        self.ae = None
        self.vae = None
        self.iforest = None
        self.lof = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._fitted = False

    def fit(self, X: np.ndarray, epochs: int = 50, batch_size: int = 256) -> 'TeacherEnsemble':
        X_scaled = self.scaler.fit_transform(X)
        input_dim = X_scaled.shape[1]
        
        hidden_dims = self.config.get('ae_hidden_dims', [64, 32, 16, 32, 64])
        self.ae = Autoencoder(input_dim, hidden_dims).to(self.device)
        self.vae = VAE(input_dim, latent_dim=self.config.get('vae_latent_dim', 8)).to(self.device)
        
        self._train_nn(self.ae, X_scaled, epochs, batch_size, is_vae=False)
        self._train_nn(self.vae, X_scaled, epochs, batch_size, is_vae=True)
        
        self.iforest = IsolationForest(
            n_estimators=self.config.get('if_n_estimators', 100),
            contamination='auto', random_state=42, n_jobs=-1
        )
        self.iforest.fit(X_scaled)
        
        self.lof = LocalOutlierFactor(
            n_neighbors=self.config.get('lof_n_neighbors', 20),
            contamination='auto', novelty=True, n_jobs=-1
        )
        self.lof.fit(X_scaled)
        
        self._fitted = True
        return self

    def _train_nn(self, model: nn.Module, X: np.ndarray, epochs: int, batch_size: int, is_vae: bool):
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        
        for _ in range(epochs):
            for (batch,) in loader:
                optimizer.zero_grad()
                if is_vae:
                    recon, mu, log_var = model(batch)
                    recon_loss = nn.functional.mse_loss(recon, batch)
                    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                    loss = recon_loss + 0.1 * kl_loss
                else:
                    recon, _ = model(batch)
                    loss = nn.functional.mse_loss(recon, batch)
                loss.backward()
                optimizer.step()

    def get_soft_labels(self, X: np.ndarray, temperature: float = 1.0) -> dict:

        if not self._fitted:
            raise RuntimeError("Teacher must be fitted before generating labels")
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.ae.eval()
        self.vae.eval()
        
        with torch.no_grad():
            ae_recon, _ = self.ae(X_tensor)
            ae_scores = torch.mean((X_tensor - ae_recon) ** 2, dim=1).cpu().numpy()
            
            vae_recon, mu, log_var = self.vae(X_tensor)
            vae_scores = torch.mean((X_tensor - vae_recon) ** 2, dim=1).cpu().numpy()
        
        if_scores = -self.iforest.score_samples(X_scaled)
        lof_scores = -self.lof.score_samples(X_scaled)
        
        scores = np.column_stack([
            self._normalize_scores(ae_scores),
            self._normalize_scores(vae_scores),
            self._normalize_scores(if_scores),
            self._normalize_scores(lof_scores)
        ])
        
        aggregated_scores = np.mean(scores, axis=1)
        soft_labels = self._softmax_with_temperature(scores, temperature)
        confidence = 1 - np.std(scores, axis=1)
        
        return {
            'soft_labels': soft_labels,
            'aggregated_scores': aggregated_scores,
            'individual_scores': scores,
            'confidence': confidence,
            'reconstruction_target': ae_recon.cpu().numpy()
        }

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        scores = np.clip(scores, np.percentile(scores, 1), np.percentile(scores, 99))
        min_s, max_s = scores.min(), scores.max()
        if max_s - min_s < 1e-10:
            return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)

    def _softmax_with_temperature(self, scores: np.ndarray, temperature: float) -> np.ndarray:
        scaled = scores / temperature
        exp_scores = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        labels = self.get_soft_labels(X)
        return labels['aggregated_scores']
