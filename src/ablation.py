"""Ablation study framework for KD-AnomalyNet."""
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional
from itertools import product
import json
from pathlib import Path
from datetime import datetime

from .teacher import TeacherEnsemble
from .student import StudentNetwork
from .distiller import AnomalyDistiller
from .losses import AnomalyDistillationLoss, TemperatureCurriculum
from .metrics import compute_metrics


@dataclass
class AblationConfig:
    name: str
    use_temperature_curriculum: bool = True
    temperature_init: float = 5.0
    temperature_final: float = 1.0
    use_boundary_aware_loss: bool = True
    boundary_margin: float = 0.1
    boundary_weight: float = 2.0
    use_dual_head: bool = True
    alpha: float = 0.7
    hidden_dims: list = field(default_factory=lambda: [32, 16])
    epochs: int = 150
    batch_size: int = 256
    learning_rate: float = 1e-3


class SingleHeadStudent(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dims: list[int] = None, dropout: float = 0.1):
        super().__init__()
        hidden_dims = hidden_dims or [32, 16]
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        
        layers.extend([
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()
        ])
        
        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        score = self.network(x).squeeze(-1)
        return x, score 

    def predict(self, X: np.ndarray, device: torch.device = None) -> np.ndarray:
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            _, scores = self.forward(X_tensor)
        return scores.cpu().numpy()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class StandardMSELoss(nn.Module):
    
    def __init__(self, alpha: float = 0.7):
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        student_recon: torch.Tensor,
        student_scores: torch.Tensor,
        teacher_recon: torch.Tensor,
        teacher_scores: torch.Tensor,
        teacher_confidence: torch.Tensor,
        temperature: float = 1.0
    ) -> tuple[torch.Tensor, dict]:
        recon_loss = nn.functional.mse_loss(student_recon, teacher_recon)
        score_loss = nn.functional.mse_loss(student_scores, teacher_scores)
        
        total_loss = self.alpha * score_loss + (1 - self.alpha) * recon_loss
        
        return total_loss, {
            'recon_loss': recon_loss.item(),
            'score_loss': score_loss.item(),
            'total_loss': total_loss.item(),
            'boundary_samples': 0.0
        }


class FixedTemperature:

    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def get_temperature(self, epoch: int) -> float:
        return self.temperature


class AblationDistiller(AnomalyDistiller):
    
    def __init__(
        self,
        teacher: TeacherEnsemble,
        student: nn.Module,
        config: AblationConfig
    ):
        super().__init__(teacher, student, {})
        self.ablation_config = config

    def distill(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> dict:
        cfg = self.ablation_config
        
        if X_val is None:
            split_idx = int(len(X_train) * 0.85)
            indices = np.random.permutation(len(X_train))
            X_train, X_val = X_train[indices[:split_idx]], X_train[indices[split_idx:]]
        
        train_labels = self.teacher.get_soft_labels(X_train)
        val_labels = self.teacher.get_soft_labels(X_val)
        
        train_loader = self._prepare_dataloader(X_train, train_labels, cfg.batch_size, shuffle=True)
        val_loader = self._prepare_dataloader(X_val, val_labels, cfg.batch_size, shuffle=False)
        
        if cfg.use_boundary_aware_loss:
            loss_fn = AnomalyDistillationLoss(
                alpha=cfg.alpha,
                boundary_margin=cfg.boundary_margin,
                boundary_weight=cfg.boundary_weight
            )
        else:
            loss_fn = StandardMSELoss(alpha=cfg.alpha)
        
        if cfg.use_temperature_curriculum:
            curriculum = TemperatureCurriculum(
                initial_temp=cfg.temperature_init,
                final_temp=cfg.temperature_final,
                total_epochs=cfg.epochs,
                decay_type='exponential'
            )
        else:
            curriculum = FixedTemperature(temperature=1.0)
        
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=cfg.learning_rate,
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        best_state = None
        
        for epoch in range(cfg.epochs):
            temperature = curriculum.get_temperature(epoch)
            self.history['temperature'].append(temperature)
            
            train_loss = self._train_epoch(train_loader, loss_fn, optimizer, temperature)
            val_loss = self._validate_epoch(val_loader, loss_fn, temperature)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            scheduler.step()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.student.state_dict().items()}
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        if best_state:
            self.student.load_state_dict(best_state)
        
        return {
            'final_train_loss': self.history['train_loss'][-1],
            'final_val_loss': self.history['val_loss'][-1],
            'best_val_loss': best_val_loss,
            'epochs_trained': len(self.history['train_loss']),
            'student_params': self.student.count_parameters(),
            'config_name': cfg.name
        }


def get_ablation_configs() -> list[AblationConfig]:
    configs = []
    

    configs.append(AblationConfig(
        name="full_model",
        use_temperature_curriculum=True,
        use_boundary_aware_loss=True,
        use_dual_head=True,
        alpha=0.7
    ))
    
    configs.append(AblationConfig(
        name="no_temp_curriculum",
        use_temperature_curriculum=False,
        use_boundary_aware_loss=True,
        use_dual_head=True,
        alpha=0.7
    ))
    
    configs.append(AblationConfig(
        name="high_init_temp",
        use_temperature_curriculum=True,
        temperature_init=10.0,
        use_boundary_aware_loss=True,
        use_dual_head=True,
        alpha=0.7
    ))
    
    configs.append(AblationConfig(
        name="low_init_temp",
        use_temperature_curriculum=True,
        temperature_init=2.0,
        use_boundary_aware_loss=True,
        use_dual_head=True,
        alpha=0.7
    ))
    
    configs.append(AblationConfig(
        name="no_boundary_loss",
        use_temperature_curriculum=True,
        use_boundary_aware_loss=False,
        use_dual_head=True,
        alpha=0.7
    ))
    
    configs.append(AblationConfig(
        name="high_boundary_weight",
        use_temperature_curriculum=True,
        use_boundary_aware_loss=True,
        boundary_weight=4.0,
        use_dual_head=True,
        alpha=0.7
    ))
    
    configs.append(AblationConfig(
        name="single_head",
        use_temperature_curriculum=True,
        use_boundary_aware_loss=True,
        use_dual_head=False,
        alpha=0.7
    ))
    
    configs.append(AblationConfig(
        name="deeper_student",
        use_temperature_curriculum=True,
        use_boundary_aware_loss=True,
        use_dual_head=True,
        hidden_dims=[64, 32, 16],
        alpha=0.7
    ))
    
    configs.append(AblationConfig(
        name="wider_student",
        use_temperature_curriculum=True,
        use_boundary_aware_loss=True,
        use_dual_head=True,
        hidden_dims=[64, 32],
        alpha=0.7
    ))
    
    # Alpha ablations
    for alpha in [0.3, 0.5, 0.7, 0.9]:
        configs.append(AblationConfig(
            name=f"alpha_{alpha}",
            use_temperature_curriculum=True,
            use_boundary_aware_loss=True,
            use_dual_head=True,
            alpha=alpha
        ))
    
    return configs


def run_ablation_study(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    teacher: TeacherEnsemble,
    output_dir: Path,
    dataset_name: str
) -> list[dict]:

    configs = get_ablation_configs()
    results = []
    
    teacher_scores = teacher.predict_scores(X_test)
    teacher_metrics = compute_metrics(y_test, teacher_scores)
    
    for cfg in configs:
        print(f"  Running ablation: {cfg.name}")
        
        if cfg.use_dual_head:
            student = StudentNetwork(
                input_dim=X_train.shape[1],
                hidden_dims=cfg.hidden_dims
            )
        else:
            student = SingleHeadStudent(
                input_dim=X_train.shape[1],
                hidden_dims=cfg.hidden_dims
            )
        
        distiller = AblationDistiller(teacher, student, cfg)
        distill_result = distiller.distill(X_train)
        
        student_scores = distiller.predict(X_test)
        student_metrics = compute_metrics(y_test, student_scores)
        
        result = {
            'dataset': dataset_name,
            'config': cfg.name,
            'config_details': {
                'use_temperature_curriculum': cfg.use_temperature_curriculum,
                'use_boundary_aware_loss': cfg.use_boundary_aware_loss,
                'use_dual_head': cfg.use_dual_head,
                'alpha': cfg.alpha,
                'hidden_dims': cfg.hidden_dims
            },
            'teacher_auc': teacher_metrics['auc_roc'],
            'student_auc': student_metrics['auc_roc'],
            'student_f1': student_metrics['f1'],
            'student_pr_auc': student_metrics['auc_pr'],
            'retention_auc': student_metrics['auc_roc'] / teacher_metrics['auc_roc'] * 100,
            'epochs_trained': distill_result['epochs_trained'],
            'student_params': distill_result['student_params']
        }
        results.append(result)
    
    return results
