import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional
from tqdm import tqdm

from .teacher import TeacherEnsemble
from .student import StudentNetwork
from .losses import AnomalyDistillationLoss, TemperatureCurriculum


class AnomalyDistiller:

    
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
        self.history = {'train_loss': [], 'val_loss': [], 'temperature': []}

    def distill(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        patience: int = 15
    ) -> dict:
        if X_val is None:
            split_idx = int(len(X_train) * 0.85)
            indices = np.random.permutation(len(X_train))
            X_train, X_val = X_train[indices[:split_idx]], X_train[indices[split_idx:]]
        
        train_labels = self.teacher.get_soft_labels(X_train)
        val_labels = self.teacher.get_soft_labels(X_val)
        
        train_loader = self._prepare_dataloader(X_train, train_labels, batch_size, shuffle=True)
        val_loader = self._prepare_dataloader(X_val, val_labels, batch_size, shuffle=False)
        
        loss_fn = AnomalyDistillationLoss(
            alpha=self.config.get('alpha', 0.7),
            boundary_margin=self.config.get('boundary_margin', 0.1),
            boundary_weight=self.config.get('boundary_weight', 2.0)
        )
        
        curriculum = TemperatureCurriculum(
            initial_temp=self.config.get('temperature_init', 5.0),
            final_temp=self.config.get('temperature_final', 1.0),
            total_epochs=epochs,
            decay_type=self.config.get('temperature_decay', 'exponential')
        )
        
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=learning_rate,
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
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
            'student_params': self.student.count_parameters()
        }

    def _prepare_dataloader(
        self,
        X: np.ndarray,
        labels: dict,
        batch_size: int,
        shuffle: bool
    ) -> DataLoader:
        X_scaled = self.teacher.scaler.transform(X)
        dataset = TensorDataset(
            torch.FloatTensor(X_scaled),
            torch.FloatTensor(labels['reconstruction_target']),
            torch.FloatTensor(labels['aggregated_scores']),
            torch.FloatTensor(labels['confidence'])
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _train_epoch(
        self,
        loader: DataLoader,
        loss_fn: AnomalyDistillationLoss,
        optimizer: torch.optim.Optimizer,
        temperature: float
    ) -> float:
        self.student.train()
        total_loss = 0.0
        
        for batch in loader:
            x, teacher_recon, teacher_scores, confidence = [b.to(self.device) for b in batch]
            
            optimizer.zero_grad()
            student_recon, student_scores = self.student(x)
            loss, _ = loss_fn(
                student_recon, student_scores,
                teacher_recon, teacher_scores,
                confidence, temperature
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * len(x)
        
        return total_loss / len(loader.dataset)

    def _validate_epoch(
        self,
        loader: DataLoader,
        loss_fn: AnomalyDistillationLoss,
        temperature: float
    ) -> float:
        self.student.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in loader:
                x, teacher_recon, teacher_scores, confidence = [b.to(self.device) for b in batch]
                student_recon, student_scores = self.student(x)
                loss, _ = loss_fn(
                    student_recon, student_scores,
                    teacher_recon, teacher_scores,
                    confidence, temperature
                )
                total_loss += loss.item() * len(x)
        
        return total_loss / len(loader.dataset)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.teacher.scaler.transform(X)
        return self.student.predict(X_scaled, self.device)

    def compare_inference_speed(self, X: np.ndarray, n_runs: int = 100) -> dict:

        import time
        
        X_scaled = self.teacher.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        for _ in range(10):
            _ = self.teacher.predict_scores(X)
            _ = self.student.predict(X_scaled, self.device)
        
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
            'speedup': teacher_time / student_time
        }

    def save(self, path: str):
        torch.save({
            'student_state': self.student.state_dict(),
            'history': self.history,
            'config': self.config
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.student.load_state_dict(checkpoint['student_state'])
        self.history = checkpoint['history']
        self.config = checkpoint['config']
