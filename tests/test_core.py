import pytest
import numpy as np
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.teacher import TeacherEnsemble, Autoencoder, VAE
from src.student import StudentNetwork
from src.distiller import AnomalyDistiller
from src.losses import AnomalyDistillationLoss, TemperatureCurriculum
from src.metrics import compute_metrics, find_optimal_threshold


@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.randn(500, 10)
    y = np.concatenate([np.zeros(475), np.ones(25)])
    return X, y


@pytest.fixture
def trained_teacher(sample_data):
    X, _ = sample_data
    teacher = TeacherEnsemble()
    teacher.fit(X[:400], epochs=5, batch_size=64)
    return teacher


class TestAutoencoder:
    def test_forward_shape(self):
        model = Autoencoder(input_dim=10, hidden_dims=[8, 4, 8])
        x = torch.randn(32, 10)
        recon, z = model(x)
        assert recon.shape == x.shape
        assert z.shape[0] == 32

    def test_reconstruction(self):
        model = Autoencoder(input_dim=10, hidden_dims=[8, 4, 8])
        x = torch.randn(32, 10)
        recon, _ = model(x)
        assert not torch.allclose(recon, x)


class TestVAE:
    def test_forward_shape(self):
        model = VAE(input_dim=10, hidden_dim=8, latent_dim=4)
        x = torch.randn(32, 10)
        recon, mu, log_var = model(x)
        assert recon.shape == x.shape
        assert mu.shape == (32, 4)
        assert log_var.shape == (32, 4)


class TestTeacherEnsemble:
    def test_fit(self, sample_data):
        X, _ = sample_data
        teacher = TeacherEnsemble()
        teacher.fit(X, epochs=2, batch_size=64)
        assert teacher._fitted

    def test_soft_labels(self, trained_teacher, sample_data):
        X, _ = sample_data
        labels = trained_teacher.get_soft_labels(X[:100])
        assert 'soft_labels' in labels
        assert 'aggregated_scores' in labels
        assert labels['aggregated_scores'].shape == (100,)

    def test_predict_scores(self, trained_teacher, sample_data):
        X, _ = sample_data
        scores = trained_teacher.predict_scores(X[:100])
        assert scores.shape == (100,)
        assert scores.min() >= 0
        assert scores.max() <= 1


class TestStudentNetwork:
    def test_forward(self):
        model = StudentNetwork(input_dim=10, hidden_dims=[8, 4])
        x = torch.randn(32, 10)
        recon, score = model(x)
        assert recon.shape == x.shape
        assert score.shape == (32,)
        assert (score >= 0).all() and (score <= 1).all()

    def test_predict(self):
        model = StudentNetwork(input_dim=10)
        X = np.random.randn(32, 10).astype(np.float32)
        scores = model.predict(X)
        assert scores.shape == (32,)

    def test_parameter_count(self):
        model = StudentNetwork(input_dim=10, hidden_dims=[8, 4])
        params = model.count_parameters()
        assert params > 0


class TestAnomalyDistillationLoss:
    def test_forward(self):
        loss_fn = AnomalyDistillationLoss()
        student_recon = torch.randn(32, 10)
        student_scores = torch.rand(32)
        teacher_recon = torch.randn(32, 10)
        teacher_scores = torch.rand(32)
        confidence = torch.rand(32)
        
        loss, metrics = loss_fn(
            student_recon, student_scores,
            teacher_recon, teacher_scores,
            confidence, temperature=2.0
        )
        assert loss.item() > 0
        assert 'recon_loss' in metrics
        assert 'score_loss' in metrics


class TestTemperatureCurriculum:
    def test_exponential_decay(self):
        curriculum = TemperatureCurriculum(5.0, 1.0, 100, 'exponential')
        t0 = curriculum.get_temperature(0)
        t50 = curriculum.get_temperature(50)
        t100 = curriculum.get_temperature(100)
        assert t0 > t50 > t100
        assert abs(t0 - 5.0) < 0.1
        assert abs(t100 - 1.0) < 0.1

    def test_linear_decay(self):
        curriculum = TemperatureCurriculum(5.0, 1.0, 100, 'linear')
        t50 = curriculum.get_temperature(50)
        assert abs(t50 - 3.0) < 0.1


class TestDistiller:
    def test_distill(self, trained_teacher, sample_data):
        X, _ = sample_data
        student = StudentNetwork(input_dim=10, hidden_dims=[8, 4])
        distiller = AnomalyDistiller(trained_teacher, student)
        result = distiller.distill(X[:400], epochs=3, batch_size=64)
        assert 'final_train_loss' in result
        assert result['epochs_trained'] > 0

    def test_predict(self, trained_teacher, sample_data):
        X, _ = sample_data
        student = StudentNetwork(input_dim=10)
        distiller = AnomalyDistiller(trained_teacher, student)
        distiller.distill(X[:400], epochs=2, batch_size=64)
        scores = distiller.predict(X[400:])
        assert scores.shape == (100,)


class TestMetrics:
    def test_compute_metrics(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        metrics = compute_metrics(y_true, y_scores, threshold=0.5)
        assert 'auc_roc' in metrics
        assert 'f1' in metrics
        assert metrics['auc_roc'] > 0.5

    def test_optimal_threshold(self):
        y_true = np.array([0]*90 + [1]*10)
        y_scores = np.concatenate([np.random.rand(90)*0.5, np.random.rand(10)*0.5 + 0.5])
        threshold = find_optimal_threshold(y_true, y_scores)
        assert 0 < threshold < 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
