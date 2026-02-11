import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional
import warnings

from .teacher import TeacherEnsemble
from .student import StudentNetwork
from .losses import AnomalyDistillationLoss, TemperatureCurriculum


class ImprovedDistiller:

    MIN_SAMPLES = 300
    MIN_SAMPLES_WARN = 500
    
    def __init__(
        self,
        teacher: TeacherEnsemble,
        student: StudentNetwork,
        config: dict = None
    ):
        self.teacher = teacher
        self.student = student
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.student.to(self.device)
        self.history = {'train_loss': [], 'val_loss': [], 'temperature': [], 'lr': []}
        self._fitted = False

    def distill(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        epochs: int = 150,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        patience: int = 20
    ) -> dict:
        n_samples = len(X_train)
        n_features = X_train.shape[1]
        

        if n_samples < self.MIN_SAMPLES:
            warnings.warn(f"Dataset very small ({n_samples} samples). Results may be unreliable.")
        elif n_samples < self.MIN_SAMPLES_WARN:
            warnings.warn(f"Dataset small ({n_samples} samples). Consider using simpler student.")
        

        batch_size = min(batch_size, max(32, n_samples // 10))
        
        # Adaptive learning rate based on dataset size
        if n_samples < 500:
            learning_rate = learning_rate * 0.5
        
        if X_val is None:
            val_ratio = 0.15 if n_samples > 1000 else 0.2
            split_idx = int(len(X_train) * (1 - val_ratio))
            indices = np.random.permutation(len(X_train))
            X_train, X_val = X_train[indices[:split_idx]], X_train[indices[split_idx:]]
        
        # Generate soft labels with error handling
        try:
            train_labels = self.teacher.get_soft_labels(X_train)
            val_labels = self.teacher.get_soft_labels(X_val)
        except Exception as e:
            raise RuntimeError(f"Failed to generate teacher labels: {e}")
        

        teacher_scores = train_labels['aggregated_scores']
        if np.std(teacher_scores) < 1e-6:
            warnings.warn("Teacher scores have near-zero variance. Distillation may fail.")
        
        train_loader = self._prepare_dataloader(X_train, train_labels, batch_size, shuffle=True)
        val_loader = self._prepare_dataloader(X_val, val_labels, batch_size, shuffle=False)
        

        loss_fn = AnomalyDistillationLoss(
            alpha=self.config.get('alpha', 0.5),
            boundary_margin=self.config.get('boundary_margin', 0.1),
            boundary_weight=self.config.get('boundary_weight', 2.0)
        )
        

        temp_init = self.config.get('temperature_init', 5.0)
        temp_final = self.config.get('temperature_final', 1.0)
        

        temp_init = np.clip(temp_init, 1.0, 10.0)
        temp_final = np.clip(temp_final, 0.5, 5.0)
        
        curriculum = TemperatureCurriculum(
            initial_temp=temp_init,
            final_temp=temp_final,
            total_epochs=epochs,
            decay_type='exponential'
        )
        

        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=learning_rate,
            weight_decay=self.config.get('weight_decay', 1e-4),
            betas=(0.9, 0.999)
        )
        

        warmup_epochs = min(10, epochs // 10)
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            temperature = curriculum.get_temperature(epoch)
            current_lr = optimizer.param_groups[0]['lr']
            
            self.history['temperature'].append(temperature)
            self.history['lr'].append(current_lr)
            
            train_loss, train_metrics = self._train_epoch(
                train_loader, loss_fn, optimizer, temperature
            )
            val_loss, val_metrics = self._validate_epoch(
                val_loader, loss_fn, temperature
            )
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            scheduler.step()
            
            # Early stopping with improvement threshold
            if val_loss < best_val_loss * 0.999:  # Require 0.1% improvement
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.student.state_dict().items()}
            else:
                patience_counter += 1
            
            # Check for training collapse
            if epoch > warmup_epochs and train_loss > self.history['train_loss'][warmup_epochs] * 2:
                warnings.warn(f"Training may be diverging at epoch {epoch}")
                break
            
            if patience_counter >= patience:
                break
        
        if best_state:
            self.student.load_state_dict(best_state)
        
        self._fitted = True
        
        return {
            'final_train_loss': self.history['train_loss'][-1],
            'final_val_loss': self.history['val_loss'][-1],
            'best_val_loss': best_val_loss,
            'epochs_trained': len(self.history['train_loss']),
            'student_params': self.student.count_parameters(),
            'converged': patience_counter >= patience,
            'batch_size_used': batch_size,
            'final_lr': self.history['lr'][-1]
        }

    def _prepare_dataloader(
        self,
        X: np.ndarray,
        labels: dict,
        batch_size: int,
        shuffle: bool
    ) -> DataLoader:
        X_scaled = self.teacher.scaler.transform(X)
        
        # Ensure float32 for stability
        X_tensor = torch.FloatTensor(X_scaled.astype(np.float32))
        recon_tensor = torch.FloatTensor(labels['reconstruction_target'].astype(np.float32))
        score_tensor = torch.FloatTensor(labels['aggregated_scores'].astype(np.float32))
        conf_tensor = torch.FloatTensor(labels['confidence'].astype(np.float32))
        
        dataset = TensorDataset(X_tensor, recon_tensor, score_tensor, conf_tensor)
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            drop_last=shuffle and len(dataset) > batch_size 
        )

    def _train_epoch(
        self,
        loader: DataLoader,
        loss_fn: AnomalyDistillationLoss,
        optimizer: torch.optim.Optimizer,
        temperature: float
    ) -> tuple[float, dict]:
        self.student.train()
        total_loss = 0.0
        total_metrics = {}
        n_batches = 0
        
        for batch in loader:
            x, teacher_recon, teacher_scores, confidence = [b.to(self.device) for b in batch]
            
            optimizer.zero_grad()
            student_recon, student_scores = self.student(x)
            
            loss, metrics = loss_fn(
                student_recon, student_scores,
                teacher_recon, teacher_scores,
                confidence, temperature
            )
            
            # Check for NaN
            if torch.isnan(loss):
                warnings.warn("NaN loss detected, skipping batch")
                continue
            
            loss.backward()
            

            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item() * len(x)
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v
            n_batches += 1
        
        avg_metrics = {k: v / n_batches for k, v in total_metrics.items()} if n_batches > 0 else {}
        return total_loss / len(loader.dataset), avg_metrics

    def _validate_epoch(
        self,
        loader: DataLoader,
        loss_fn: AnomalyDistillationLoss,
        temperature: float
    ) -> tuple[float, dict]:
        self.student.eval()
        total_loss = 0.0
        total_metrics = {}
        n_batches = 0
        
        with torch.no_grad():
            for batch in loader:
                x, teacher_recon, teacher_scores, confidence = [b.to(self.device) for b in batch]
                student_recon, student_scores = self.student(x)
                
                loss, metrics = loss_fn(
                    student_recon, student_scores,
                    teacher_recon, teacher_scores,
                    confidence, temperature
                )
                
                total_loss += loss.item() * len(x)
                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0) + v
                n_batches += 1
        
        avg_metrics = {k: v / n_batches for k, v in total_metrics.items()} if n_batches > 0 else {}
        return total_loss / len(loader.dataset), avg_metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            warnings.warn("Distiller not fitted. Results may be random.")
        X_scaled = self.teacher.scaler.transform(X)
        return self.student.predict(X_scaled, self.device)

    def compare_inference_speed(self, X: np.ndarray, n_runs: int = 100) -> dict:

        import time
        
        X_scaled = self.teacher.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        

        for _ in range(10):
            _ = self.teacher.predict_scores(X)
            with torch.no_grad():
                _ = self.student(X_tensor)
        

        start = time.time()
        for _ in range(n_runs):
            _ = self.teacher.predict_scores(X)
        teacher_time = (time.time() - start) / n_runs
        

        self.student.eval()
        start = time.time()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = self.student(X_tensor)
        student_time = (time.time() - start) / n_runs
        
        return {
            'teacher_time_ms': teacher_time * 1000,
            'student_time_ms': student_time * 1000,
            'speedup': teacher_time / max(student_time, 1e-10)
        }

    def save(self, path: str):
        torch.save({
            'student_state': self.student.state_dict(),
            'history': self.history,
            'config': self.config,
            'fitted': self._fitted
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.student.load_state_dict(checkpoint['student_state'])
        self.history = checkpoint['history']
        self.config = checkpoint['config']
        self._fitted = checkpoint.get('fitted', True)


class AdaptiveStudentNetwork(nn.Module):

    
    def __init__(
        self, 
        input_dim: int, 
        n_samples: int,
        hidden_dims: list[int] = None,
        dropout: float = 0.1,
        use_reconstruction_head: bool = True
    ):
        super().__init__()
        

        if hidden_dims is None:
            if n_samples < 500:
                hidden_dims = [32, 16]
                dropout = 0.2 
            elif n_samples < 2000:
                hidden_dims = [64, 32]
            else:
                hidden_dims = [128, 64]
        
        self.use_reconstruction_head = use_reconstruction_head
        
        # Shared encoder
        shared_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),  # LayerNorm instead of BatchNorm for small batches
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        self.shared = nn.Sequential(*shared_layers)
        

        self.score_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()
        )
        

        if use_reconstruction_head:
            self.recon_head = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], input_dim)
            )
        else:
            self.recon_head = None
        
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
        score = self.score_head(h).squeeze(-1)
        
        if self.recon_head is not None:
            recon = self.recon_head(h)
        else:
            recon = x  # Pass-through for compatibility
        
        return recon, score

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
