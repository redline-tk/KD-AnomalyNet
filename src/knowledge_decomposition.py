
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import pearsonr, spearmanr
from typing import Optional
from dataclasses import dataclass


@dataclass
class KnowledgeComponent:

    name: str
    description: str
    scores: np.ndarray
    auc_roc: float
    auc_pr: float
    weight_in_ensemble: float


@dataclass
class TransferAnalysis:

    component_name: str
    teacher_auc: float
    student_correlation: float  
    knowledge_retained: float   
    importance: float          


class KnowledgeDecomposer:


    
    def __init__(self, teacher_ensemble):
        """
        Args:
            teacher_ensemble: Fitted TeacherEnsemble instance
        """
        self.teacher = teacher_ensemble
        self.components = {}
        self._decomposed = False

    def decompose(self, X: np.ndarray, y_true: np.ndarray) -> dict[str, KnowledgeComponent]:

        X_scaled = self.teacher.scaler.transform(X)
        
        # 1. Reconstruction knowledge (Autoencoder)
        if hasattr(self.teacher, 'autoencoder') and self.teacher.autoencoder is not None:
            ae_scores = self._get_autoencoder_scores(X_scaled)
            ae_auc = self._safe_auc(y_true, ae_scores)
            self.components['reconstruction'] = KnowledgeComponent(
                name='reconstruction',
                description='Anomaly detection via reconstruction error (Autoencoder)',
                scores=ae_scores,
                auc_roc=ae_auc,
                auc_pr=self._safe_ap(y_true, ae_scores),
                weight_in_ensemble=0.25
            )
        
        # 2. Density knowledge (VAE)
        if hasattr(self.teacher, 'vae') and self.teacher.vae is not None:
            vae_scores = self._get_vae_scores(X_scaled)
            vae_auc = self._safe_auc(y_true, vae_scores)
            self.components['density'] = KnowledgeComponent(
                name='density',
                description='Anomaly detection via density estimation (VAE ELBO)',
                scores=vae_scores,
                auc_roc=vae_auc,
                auc_pr=self._safe_ap(y_true, vae_scores),
                weight_in_ensemble=0.25
            )
        
        # 3. Isolation knowledge (Isolation Forest)
        if hasattr(self.teacher, 'iforest') and self.teacher.iforest is not None:
            if_scores = self._get_isolation_scores(X_scaled)
            if_auc = self._safe_auc(y_true, if_scores)
            self.components['isolation'] = KnowledgeComponent(
                name='isolation',
                description='Anomaly detection via isolation path length',
                scores=if_scores,
                auc_roc=if_auc,
                auc_pr=self._safe_ap(y_true, if_scores),
                weight_in_ensemble=0.25
            )
        
        # 4. Local structure knowledge (LOF)
        if hasattr(self.teacher, 'lof') and self.teacher.lof is not None:
            lof_scores = self._get_lof_scores(X_scaled)
            lof_auc = self._safe_auc(y_true, lof_scores)
            self.components['local_structure'] = KnowledgeComponent(
                name='local_structure',
                description='Anomaly detection via local outlier factor',
                scores=lof_scores,
                auc_roc=lof_auc,
                auc_pr=self._safe_ap(y_true, lof_scores),
                weight_in_ensemble=0.25
            )
        
        # 5. Ensemble knowledge (combined)
        ensemble_scores = self.teacher.predict_scores(
            self.teacher.scaler.inverse_transform(X_scaled)
        )
        self.components['ensemble'] = KnowledgeComponent(
            name='ensemble',
            description='Combined ensemble prediction',
            scores=ensemble_scores,
            auc_roc=self._safe_auc(y_true, ensemble_scores),
            auc_pr=self._safe_ap(y_true, ensemble_scores),
            weight_in_ensemble=1.0
        )
        
        self._decomposed = True
        return self.components

    def _get_autoencoder_scores(self, X_scaled: np.ndarray) -> np.ndarray:

        import torch
        self.teacher.autoencoder.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.teacher.device)
            recon = self.teacher.autoencoder(X_tensor)
            errors = torch.mean((X_tensor - recon) ** 2, dim=1)
        scores = errors.cpu().numpy()
        return self._normalize_scores(scores)

    def _get_vae_scores(self, X_scaled: np.ndarray) -> np.ndarray:

        import torch
        self.teacher.vae.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.teacher.device)
            recon, mu, logvar = self.teacher.vae(X_tensor)

            recon_loss = torch.mean((X_tensor - recon) ** 2, dim=1)

            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            scores = (recon_loss + 0.1 * kl_loss).cpu().numpy()
        return self._normalize_scores(scores)

    def _get_isolation_scores(self, X_scaled: np.ndarray) -> np.ndarray:

        scores = -self.teacher.iforest.decision_function(X_scaled)
        return self._normalize_scores(scores)

    def _get_lof_scores(self, X_scaled: np.ndarray) -> np.ndarray:
        """Get LOF scores."""
        scores = -self.teacher.lof.decision_function(X_scaled)
        return self._normalize_scores(scores)

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1]."""
        min_s, max_s = scores.min(), scores.max()
        if max_s - min_s < 1e-10:
            return np.full_like(scores, 0.5)
        return (scores - min_s) / (max_s - min_s)

    def _safe_auc(self, y_true: np.ndarray, scores: np.ndarray) -> float:
        """Safely compute AUC-ROC."""
        try:
            return float(roc_auc_score(y_true, scores))
        except:
            return 0.5

    def _safe_ap(self, y_true: np.ndarray, scores: np.ndarray) -> float:
        """Safely compute average precision."""
        try:
            return float(average_precision_score(y_true, scores))
        except:
            return 0.0

    def analyze_student_transfer(
        self,
        student_scores: np.ndarray,
        y_true: np.ndarray
    ) -> dict[str, TransferAnalysis]:

        if not self._decomposed:
            raise RuntimeError("Must call decompose() first")
        
        student_auc = self._safe_auc(y_true, student_scores)
        transfer_results = {}
        
        for name, component in self.components.items():

            if len(component.scores) > 1:
                try:
                    pearson_corr, _ = pearsonr(student_scores, component.scores)
                    spearman_corr, _ = spearmanr(student_scores, component.scores)
                    correlation = (pearson_corr + spearman_corr) / 2
                except:
                    correlation = 0.0
            else:
                correlation = 0.0

            if component.auc_roc > 0.5:
                auc_retention = (student_auc - 0.5) / (component.auc_roc - 0.5)
                auc_retention = np.clip(auc_retention, 0, 1.5)  # Allow > 1 if student does better
            else:
                auc_retention = 0.0
            

            knowledge_retained = 0.5 * max(0, correlation) + 0.5 * auc_retention

            importance = (component.auc_roc - 0.5) * 2  # Scale 0.5-1.0 to 0-1
            
            transfer_results[name] = TransferAnalysis(
                component_name=name,
                teacher_auc=component.auc_roc,
                student_correlation=float(correlation) if not np.isnan(correlation) else 0.0,
                knowledge_retained=float(knowledge_retained),
                importance=float(np.clip(importance, 0, 1))
            )
        
        return transfer_results

    def get_component_contributions(self, y_true: np.ndarray) -> dict:

        if not self._decomposed:
            raise RuntimeError("Must call decompose() first")
        
        ensemble_auc = self.components['ensemble'].auc_roc
        contributions = {}
        

        component_names = [k for k in self.components.keys() if k != 'ensemble']
        
        for exclude_name in component_names:

            other_scores = []
            for name, comp in self.components.items():
                if name != exclude_name and name != 'ensemble':
                    other_scores.append(comp.scores)
            
            if other_scores:
                combined = np.mean(other_scores, axis=0)
                reduced_auc = self._safe_auc(y_true, combined)
                contribution = ensemble_auc - reduced_auc
            else:
                contribution = 0.0
            
            contributions[exclude_name] = {
                'standalone_auc': self.components[exclude_name].auc_roc,
                'contribution_to_ensemble': float(contribution),
                'is_beneficial': contribution > 0
            }
        
        return contributions

    def generate_report(
        self,
        student_scores: np.ndarray,
        y_true: np.ndarray
    ) -> dict:

        if not self._decomposed:
            raise RuntimeError("Must call decompose() first")
        
        transfer = self.analyze_student_transfer(student_scores, y_true)
        contributions = self.get_component_contributions(y_true)
        

        component_aucs = {k: v.auc_roc for k, v in self.components.items()}
        transfer_rates = {k: v.knowledge_retained for k, v in transfer.items()}
        
 
        non_ensemble_transfer = {k: v for k, v in transfer.items() if k != 'ensemble'}
        if non_ensemble_transfer:
            best_transfer = max(non_ensemble_transfer.items(), key=lambda x: x[1].knowledge_retained)
            worst_transfer = min(non_ensemble_transfer.items(), key=lambda x: x[1].knowledge_retained)
        else:
            best_transfer = worst_transfer = (None, None)
        
        return {
            'components': {
                name: {
                    'description': comp.description,
                    'auc_roc': comp.auc_roc,
                    'auc_pr': comp.auc_pr
                }
                for name, comp in self.components.items()
            },
            'transfer_analysis': {
                name: {
                    'correlation': t.student_correlation,
                    'knowledge_retained': t.knowledge_retained,
                    'importance': t.importance
                }
                for name, t in transfer.items()
            },
            'contributions': contributions,
            'summary': {
                'best_component': max(component_aucs.items(), key=lambda x: x[1] if x[0] != 'ensemble' else 0),
                'worst_component': min(component_aucs.items(), key=lambda x: x[1] if x[0] != 'ensemble' else 1),
                'best_transfer': (best_transfer[0], best_transfer[1].knowledge_retained if best_transfer[1] else 0),
                'worst_transfer': (worst_transfer[0], worst_transfer[1].knowledge_retained if worst_transfer[1] else 0),
                'mean_transfer_rate': np.mean([v.knowledge_retained for k, v in transfer.items() if k != 'ensemble'])
            }
        }


def create_knowledge_report(
    teacher_ensemble,
    X_test: np.ndarray,
    y_test: np.ndarray,
    student_scores: np.ndarray
) -> dict:

    decomposer = KnowledgeDecomposer(teacher_ensemble)
    decomposer.decompose(X_test, y_test)
    return decomposer.generate_report(student_scores, y_test)
