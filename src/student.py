import numpy as np
import torch
import torch.nn as nn


class StudentNetwork(nn.Module):

    
    def __init__(self, input_dim: int, hidden_dims: list[int] = None, dropout: float = 0.1):
        super().__init__()
        hidden_dims = hidden_dims or [32, 16]
        
        shared_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        self.shared = nn.Sequential(*shared_layers)
        
        self.recon_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim)
        )
        
        self.score_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()
        )
        
        self.input_dim = input_dim
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        h = self.shared(x)
        recon = self.recon_head(h)
        score = self.score_head(h).squeeze(-1)
        return recon, score

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.shared(x)

    def predict(self, X: np.ndarray, device: torch.device = None) -> np.ndarray:

        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.eval()
        
        X_tensor = torch.FloatTensor(X).to(device)
        with torch.no_grad():
            _, scores = self.forward(X_tensor)
        return scores.cpu().numpy()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
