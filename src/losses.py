import torch
import torch.nn as nn
import torch.nn.functional as F


class AnomalyDistillationLoss(nn.Module):

    
    def __init__(
        self,
        alpha: float = 0.7,
        boundary_margin: float = 0.1,
        boundary_weight: float = 2.0
    ):
        super().__init__()
        self.alpha = alpha
        self.boundary_margin = boundary_margin
        self.boundary_weight = boundary_weight

    def forward(
        self,
        student_recon: torch.Tensor,
        student_scores: torch.Tensor,
        teacher_recon: torch.Tensor,
        teacher_scores: torch.Tensor,
        teacher_confidence: torch.Tensor,
        temperature: float = 1.0
    ) -> tuple[torch.Tensor, dict]:
        recon_loss = F.mse_loss(student_recon, teacher_recon, reduction='none')
        recon_loss = recon_loss.mean(dim=1)
        
        teacher_soft = torch.sigmoid(teacher_scores / temperature)
        student_soft = torch.sigmoid(student_scores / temperature)
        score_loss = F.binary_cross_entropy(student_soft, teacher_soft, reduction='none')
        
        boundary_mask = self._compute_boundary_weights(teacher_scores)
        weighted_score_loss = score_loss * boundary_mask * teacher_confidence
        
        combined_recon = (recon_loss * teacher_confidence).mean()
        combined_score = weighted_score_loss.mean() * (temperature ** 2)
        
        total_loss = self.alpha * combined_score + (1 - self.alpha) * combined_recon
        
        metrics = {
            'recon_loss': combined_recon.item(),
            'score_loss': combined_score.item(),
            'total_loss': total_loss.item(),
            'boundary_samples': (boundary_mask > 1.0).float().mean().item()
        }
        
        return total_loss, metrics

    def _compute_boundary_weights(self, scores: torch.Tensor) -> torch.Tensor:
        threshold = 0.5
        distance_to_boundary = torch.abs(scores - threshold)
        is_boundary = distance_to_boundary < self.boundary_margin
        weights = torch.where(
            is_boundary,
            torch.full_like(scores, self.boundary_weight),
            torch.ones_like(scores)
        )
        return weights


class TemperatureCurriculum:

    
    def __init__(
        self,
        initial_temp: float = 5.0,
        final_temp: float = 1.0,
        total_epochs: int = 100,
        decay_type: str = 'exponential'
    ):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.total_epochs = total_epochs
        self.decay_type = decay_type

    def get_temperature(self, epoch: int) -> float:
        progress = min(epoch / self.total_epochs, 1.0)
        
        if self.decay_type == 'exponential':
            log_init = torch.log(torch.tensor(self.initial_temp))
            log_final = torch.log(torch.tensor(self.final_temp))
            log_temp = log_init + progress * (log_final - log_init)
            return torch.exp(log_temp).item()
        elif self.decay_type == 'linear':
            return self.initial_temp + progress * (self.final_temp - self.initial_temp)
        elif self.decay_type == 'cosine':
            cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
            return self.final_temp + cosine_decay * (self.initial_temp - self.final_temp)
        else:
            return self.initial_temp


class ContrastiveDistillationLoss(nn.Module):

    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        student_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor,
        anomaly_labels: torch.Tensor
    ) -> torch.Tensor:
        cos_sim = F.cosine_similarity(student_embeddings, teacher_embeddings)
        normal_mask = anomaly_labels < 0.5
        anomaly_mask = ~normal_mask
        
        normal_loss = (1 - cos_sim[normal_mask]).mean() if normal_mask.any() else torch.tensor(0.0)
        anomaly_loss = F.relu(cos_sim[anomaly_mask] - self.margin).mean() if anomaly_mask.any() else torch.tensor(0.0)
        
        return normal_loss + anomaly_loss
