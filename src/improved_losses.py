import torch
import torch.nn as nn
import torch.nn.functional as F


class StableDistillationLoss(nn.Module):

    def __init__(
        self,
        alpha: float = 0.7,
        boundary_margin: float = 0.1,
        boundary_weight: float = 2.0,
        label_smoothing: float = 0.1,
        use_huber: bool = True
    ):
        super().__init__()
        self.alpha = alpha
        self.boundary_margin = boundary_margin
        self.boundary_weight = boundary_weight
        self.label_smoothing = label_smoothing
        self.use_huber = use_huber
        
        if use_huber:
            self.recon_loss_fn = nn.HuberLoss(reduction='none', delta=1.0)
        else:
            self.recon_loss_fn = nn.MSELoss(reduction='none')

    def forward(
        self,
        student_recon: torch.Tensor,
        student_scores: torch.Tensor,
        teacher_recon: torch.Tensor,
        teacher_scores: torch.Tensor,
        teacher_confidence: torch.Tensor,
        temperature: float = 1.0
    ) -> tuple[torch.Tensor, dict]:
        

        student_scores = student_scores.view(-1)
        teacher_scores = teacher_scores.view(-1)
        teacher_confidence = teacher_confidence.view(-1)
        

        student_scores = torch.clamp(student_scores, 1e-7, 1 - 1e-7)
        teacher_scores = torch.clamp(teacher_scores, 1e-7, 1 - 1e-7)
        teacher_confidence = torch.clamp(teacher_confidence, 0.1, 1.0)
        

        recon_loss = self.recon_loss_fn(student_recon, teacher_recon)
        recon_loss = recon_loss.mean(dim=1)  # Per-sample loss
        

        smoothed_teacher = teacher_scores * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        

        teacher_soft = torch.sigmoid(torch.logit(smoothed_teacher) / temperature)
        student_soft = student_scores 
        
        score_loss = F.binary_cross_entropy(
            student_soft, 
            teacher_soft,
            reduction='none'
        )
        

        boundary_weights = self._compute_boundary_weights(teacher_scores)
        weighted_score_loss = score_loss * boundary_weights * teacher_confidence
        

        combined_recon = (recon_loss * teacher_confidence).mean()
        combined_score = weighted_score_loss.mean() * (temperature ** 2)
        
        total_loss = self.alpha * combined_score + (1 - self.alpha) * combined_recon
        

        if torch.isnan(total_loss) or torch.isinf(total_loss):

            total_loss = F.mse_loss(student_scores, teacher_scores)
        
        metrics = {
            'recon_loss': combined_recon.item(),
            'score_loss': combined_score.item(),
            'total_loss': total_loss.item(),
            'boundary_ratio': (boundary_weights > 1.0).float().mean().item(),
            'mean_confidence': teacher_confidence.mean().item()
        }
        
        return total_loss, metrics

    def _compute_boundary_weights(self, scores: torch.Tensor) -> torch.Tensor:

        threshold = 0.5
        distance_to_boundary = torch.abs(scores - threshold)
        

        weights = 1.0 + (self.boundary_weight - 1.0) * torch.exp(
            -distance_to_boundary / self.boundary_margin
        )
        
        return weights


class ScoreOnlyLoss(nn.Module):

    def __init__(
        self,
        boundary_margin: float = 0.1,
        boundary_weight: float = 2.0
    ):
        super().__init__()
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
        
        student_scores = student_scores.view(-1)
        teacher_scores = teacher_scores.view(-1)
        teacher_confidence = teacher_confidence.view(-1)
        

        student_scores = torch.clamp(student_scores, 1e-7, 1 - 1e-7)
        teacher_scores = torch.clamp(teacher_scores, 1e-7, 1 - 1e-7)
        

        teacher_soft = torch.sigmoid(torch.logit(teacher_scores) / temperature)
        

        student_logits = torch.logit(student_scores)
        teacher_logits = torch.logit(teacher_soft)
        
        score_loss = F.mse_loss(student_logits, teacher_logits, reduction='none')
        

        boundary_weights = self._compute_boundary_weights(teacher_scores)
        weighted_loss = score_loss * boundary_weights * teacher_confidence
        
        total_loss = weighted_loss.mean() * (temperature ** 2)
        
        metrics = {
            'recon_loss': 0.0,
            'score_loss': total_loss.item(),
            'total_loss': total_loss.item(),
            'boundary_ratio': (boundary_weights > 1.0).float().mean().item()
        }
        
        return total_loss, metrics

    def _compute_boundary_weights(self, scores: torch.Tensor) -> torch.Tensor:
        threshold = 0.5
        distance = torch.abs(scores - threshold)
        weights = 1.0 + (self.boundary_weight - 1.0) * torch.exp(
            -distance / self.boundary_margin
        )
        return weights


class RankingDistillationLoss(nn.Module):

    
    def __init__(self, margin: float = 0.1, n_pairs: int = 100):
        super().__init__()
        self.margin = margin
        self.n_pairs = n_pairs

    def forward(
        self,
        student_recon: torch.Tensor,
        student_scores: torch.Tensor,
        teacher_recon: torch.Tensor,
        teacher_scores: torch.Tensor,
        teacher_confidence: torch.Tensor,
        temperature: float = 1.0
    ) -> tuple[torch.Tensor, dict]:
        
        batch_size = student_scores.shape[0]
        student_scores = student_scores.view(-1)
        teacher_scores = teacher_scores.view(-1)
        
        # Sample pairs
        n_pairs = min(self.n_pairs, batch_size * (batch_size - 1) // 2)
        
        if n_pairs < 10:

            loss = F.mse_loss(student_scores, teacher_scores)
            return loss, {'total_loss': loss.item(), 'recon_loss': 0, 'score_loss': loss.item()}
        
        idx_i = torch.randint(0, batch_size, (n_pairs,), device=student_scores.device)
        idx_j = torch.randint(0, batch_size, (n_pairs,), device=student_scores.device)
        

        mask = idx_i != idx_j
        idx_i = idx_i[mask]
        idx_j = idx_j[mask]
        
        if len(idx_i) == 0:
            loss = F.mse_loss(student_scores, teacher_scores)
            return loss, {'total_loss': loss.item(), 'recon_loss': 0, 'score_loss': loss.item()}
        

        teacher_diff = teacher_scores[idx_i] - teacher_scores[idx_j]
        teacher_sign = torch.sign(teacher_diff)
        

        student_diff = student_scores[idx_i] - student_scores[idx_j]
        

        ranking_loss = F.relu(self.margin - teacher_sign * student_diff).mean()
        

        mse_loss = F.mse_loss(student_scores, teacher_scores)
        
        total_loss = 0.5 * ranking_loss + 0.5 * mse_loss
        
        metrics = {
            'recon_loss': 0.0,
            'score_loss': total_loss.item(),
            'total_loss': total_loss.item(),
            'ranking_loss': ranking_loss.item(),
            'mse_loss': mse_loss.item()
        }
        
        return total_loss, metrics


class TemperatureCurriculum:

    
    def __init__(
        self,
        initial_temp: float = 5.0,
        final_temp: float = 1.0,
        total_epochs: int = 100,
        decay_type: str = 'exponential',
        warmup_epochs: int = 10
    ):
        self.initial_temp = max(1.0, min(initial_temp, 20.0))  
        self.final_temp = max(0.5, min(final_temp, self.initial_temp))
        self.total_epochs = max(1, total_epochs)
        self.decay_type = decay_type
        self.warmup_epochs = min(warmup_epochs, total_epochs // 5)

    def get_temperature(self, epoch: int) -> float:

        if epoch < self.warmup_epochs:
            return self.initial_temp
        

        adjusted_epoch = epoch - self.warmup_epochs
        adjusted_total = self.total_epochs - self.warmup_epochs
        progress = min(adjusted_epoch / max(adjusted_total, 1), 1.0)
        
        if self.decay_type == 'exponential':

            log_init = torch.log(torch.tensor(self.initial_temp))
            log_final = torch.log(torch.tensor(self.final_temp))
            log_temp = log_init + progress * (log_final - log_init)
            return torch.exp(log_temp).item()
        elif self.decay_type == 'linear':
            return self.initial_temp + progress * (self.final_temp - self.initial_temp)
        elif self.decay_type == 'cosine':
            cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
            return self.final_temp + cosine_decay.item() * (self.initial_temp - self.final_temp)
        elif self.decay_type == 'step':
            # Step decay at 1/3 and 2/3
            if progress < 0.33:
                return self.initial_temp
            elif progress < 0.66:
                return (self.initial_temp + self.final_temp) / 2
            else:
                return self.final_temp
        else:
            return self.initial_temp
