import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pathlib import Path
from typing import Optional
import warnings


plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9


def plot_roc_curves(
    results: list[dict],
    save_path: Optional[Path] = None
) -> plt.Figure:

    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for i, result in enumerate(results):
        if 'fpr' in result and 'tpr' in result:
            ax.plot(
                result['fpr'], result['tpr'],
                color=colors[i],
                lw=2,
                label=f"{result['name']} (AUC={result['auc_roc']:.3f})"
            )
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves Comparison')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    
    return fig


def plot_score_distributions(
    teacher_scores: np.ndarray,
    student_scores: np.ndarray,
    y_true: np.ndarray,
    save_path: Optional[Path] = None
) -> plt.Figure:

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Teacher
    ax = axes[0]
    ax.hist(teacher_scores[y_true == 0], bins=50, alpha=0.7, 
            label='Normal', color='steelblue', density=True)
    ax.hist(teacher_scores[y_true == 1], bins=50, alpha=0.7, 
            label='Anomaly', color='crimson', density=True)
    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Density')
    ax.set_title('Teacher Score Distribution')
    ax.legend()
    
    # Student
    ax = axes[1]
    ax.hist(student_scores[y_true == 0], bins=50, alpha=0.7,
            label='Normal', color='steelblue', density=True)
    ax.hist(student_scores[y_true == 1], bins=50, alpha=0.7,
            label='Anomaly', color='crimson', density=True)
    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Density')
    ax.set_title('Student Score Distribution')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    
    return fig


def plot_tsne_embeddings(
    X: np.ndarray,
    y: np.ndarray,
    scores: Optional[np.ndarray] = None,
    title: str = 't-SNE Visualization',
    save_path: Optional[Path] = None,
    perplexity: int = 30
) -> plt.Figure:

    max_samples = 5000
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X = X[idx]
        y = y[idx]
        if scores is not None:
            scores = scores[idx]

    if X.shape[1] > 50:
        pca = PCA(n_components=50)
        X = pca.fit_transform(X)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
        X_embedded = tsne.fit_transform(X)
    
    if scores is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # By label
        ax = axes[0]
        scatter = ax.scatter(
            X_embedded[y == 0, 0], X_embedded[y == 0, 1],
            c='steelblue', alpha=0.5, s=20, label='Normal'
        )
        ax.scatter(
            X_embedded[y == 1, 0], X_embedded[y == 1, 1],
            c='crimson', alpha=0.8, s=30, label='Anomaly', marker='x'
        )
        ax.set_title(f'{title} - By Label')
        ax.legend()
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        
        # By score
        ax = axes[1]
        scatter = ax.scatter(
            X_embedded[:, 0], X_embedded[:, 1],
            c=scores, cmap='RdYlBu_r', alpha=0.6, s=20
        )
        plt.colorbar(scatter, ax=ax, label='Anomaly Score')
        ax.set_title(f'{title} - By Score')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            X_embedded[y == 0, 0], X_embedded[y == 0, 1],
            c='steelblue', alpha=0.5, s=20, label='Normal'
        )
        ax.scatter(
            X_embedded[y == 1, 0], X_embedded[y == 1, 1],
            c='crimson', alpha=0.8, s=30, label='Anomaly', marker='x'
        )
        ax.set_title(title)
        ax.legend()
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    
    return fig


def plot_critical_difference_diagram(
    cd_data: dict,
    save_path: Optional[Path] = None
) -> plt.Figure:

    methods = cd_data['methods']
    ranks = cd_data['ranks']
    cd = cd_data['critical_difference']
    n_methods = len(methods)
    
    fig, ax = plt.subplots(figsize=(10, max(3, n_methods * 0.4)))
    
    # Plot axis
    ax.set_xlim(0.5, n_methods + 0.5)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color='black', linewidth=1)
    

    for i in range(1, n_methods + 1):
        ax.axvline(i, ymin=0.45, ymax=0.55, color='black', linewidth=1)
        ax.text(i, 0.4, str(i), ha='center', va='top', fontsize=10)
    

    ax.plot([1, 1 + cd], [0.85, 0.85], 'k-', linewidth=2)
    ax.plot([1, 1], [0.83, 0.87], 'k-', linewidth=2)
    ax.plot([1 + cd, 1 + cd], [0.83, 0.87], 'k-', linewidth=2)
    ax.text(1 + cd/2, 0.9, f'CD={cd:.2f}', ha='center', va='bottom', fontsize=9)
    

    left_methods = [(m, r) for m, r in zip(methods, ranks) if r <= (n_methods + 1) / 2]
    right_methods = [(m, r) for m, r in zip(methods, ranks) if r > (n_methods + 1) / 2]
    

    for i, (method, rank) in enumerate(left_methods):
        y_pos = 0.5 - (i + 1) * 0.08
        ax.plot([rank, rank], [0.5, y_pos + 0.02], 'k-', linewidth=0.5)
        ax.plot([0.5, rank], [y_pos, y_pos], 'k-', linewidth=0.5)
        ax.text(0.4, y_pos, method, ha='right', va='center', fontsize=9)
    

    for i, (method, rank) in enumerate(right_methods):
        y_pos = 0.5 - (i + 1) * 0.08
        ax.plot([rank, rank], [0.5, y_pos + 0.02], 'k-', linewidth=0.5)
        ax.plot([rank, n_methods + 0.5], [y_pos, y_pos], 'k-', linewidth=0.5)
        ax.text(n_methods + 0.6, y_pos, method, ha='left', va='center', fontsize=9)
    

    for clique in cd_data.get('cliques', []):
        clique_methods = clique['methods']
        clique_ranks = [ranks[methods.index(m)] for m in clique_methods]
        y_pos = 0.5 + 0.1 * (cd_data['cliques'].index(clique) + 1)
        ax.plot([min(clique_ranks), max(clique_ranks)], [y_pos, y_pos], 
                'k-', linewidth=3)
    
    ax.set_title('Critical Difference Diagram', fontsize=12)
    ax.text((n_methods + 1) / 2, 0.3, 'Average Rank', ha='center', fontsize=10)
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    
    return fig


def plot_ablation_heatmap(
    ablation_results: list[dict],
    metric: str = 'student_auc',
    save_path: Optional[Path] = None
) -> plt.Figure:

    datasets = sorted(set(r['dataset'] for r in ablation_results))
    configs = sorted(set(r['config'] for r in ablation_results))
    
    matrix = np.zeros((len(datasets), len(configs)))
    
    for r in ablation_results:
        i = datasets.index(r['dataset'])
        j = configs.index(r['config'])
        matrix[i, j] = r[metric]
    
    fig, ax = plt.subplots(figsize=(max(10, len(configs) * 0.8), max(6, len(datasets) * 0.5)))
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')
    
    ax.set_xticks(np.arange(len(configs)))
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.set_yticklabels(datasets)
    

    for i in range(len(datasets)):
        for j in range(len(configs)):
            text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                          ha='center', va='center', fontsize=8)
    
    ax.set_title(f'Ablation Study: {metric}')
    plt.colorbar(im, ax=ax, label=metric)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    
    return fig


def plot_hyperparameter_sensitivity(
    results: list[dict],
    param_name: str,
    metric: str = 'auc_roc',
    save_path: Optional[Path] = None
) -> plt.Figure:

    param_values = sorted(set(r[param_name] for r in results))
    datasets = sorted(set(r['dataset'] for r in results))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
    
    for i, dataset in enumerate(datasets):
        dataset_results = [r for r in results if r['dataset'] == dataset]
        values = []
        scores = []
        for pv in param_values:
            matching = [r for r in dataset_results if r[param_name] == pv]
            if matching:
                values.append(pv)
                scores.append(matching[0][metric])
        
        ax.plot(values, scores, 'o-', color=colors[i], label=dataset, markersize=6)
    
    ax.set_xlabel(param_name)
    ax.set_ylabel(metric)
    ax.set_title(f'Sensitivity Analysis: {param_name}')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    
    return fig


def plot_training_curves(
    history: dict,
    save_path: Optional[Path] = None
) -> plt.Figure:

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss curves
    ax = axes[0]
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    

    ax = axes[1]
    if 'temperature' in history and history['temperature']:
        ax.plot(epochs, history['temperature'], 'g-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Temperature')
        ax.set_title('Temperature Schedule')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No temperature data', ha='center', va='center')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    
    return fig


def plot_baseline_comparison_bar(
    baseline_results: list[dict],
    our_result: dict,
    metric: str = 'auc_roc',
    save_path: Optional[Path] = None
) -> plt.Figure:

    all_results = baseline_results + [our_result]
    all_results = sorted(all_results, key=lambda x: x[metric], reverse=True)
    
    names = [r['name'] for r in all_results]
    scores = [r[metric] for r in all_results]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['crimson' if r['name'] == our_result['name'] else 'steelblue' for r in all_results]
    
    bars = ax.barh(names, scores, color=colors, edgecolor='black', linewidth=0.5)
    
    for bar, score in zip(bars, scores):
        ax.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', fontsize=9)
    
    ax.set_xlabel(metric.upper().replace('_', '-'))
    ax.set_title(f'Method Comparison: {metric}')
    ax.set_xlim(0, max(scores) * 1.15)
    

    legend_elements = [
        mpatches.Patch(facecolor='crimson', edgecolor='black', label='Ours'),
        mpatches.Patch(facecolor='steelblue', edgecolor='black', label='Baselines')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    
    return fig


def plot_speedup_comparison(
    results: list[dict],
    save_path: Optional[Path] = None
) -> plt.Figure:

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    names = [r['name'] for r in results]
    times = [r.get('inference_time_s', r.get('inference_time', 0)) * 1000 for r in results]
    

    ax = axes[0]
    ax.barh(names, times, color='steelblue', edgecolor='black')
    ax.set_xlabel('Inference Time (ms)')
    ax.set_title('Inference Time Comparison')
    

    ax = axes[1]
    if len(results) > 1:
        teacher_time = times[0]  # Assume first is teacher
        speedups = [teacher_time / t if t > 0 else 0 for t in times]
        colors = ['crimson' if s > 1 else 'steelblue' for s in speedups]
        ax.barh(names, speedups, color=colors, edgecolor='black')
        ax.axvline(1, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Speedup (x)')
        ax.set_title('Speedup vs Teacher')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    
    return fig


def create_all_visualizations(
    experiment_results: dict,
    output_dir: Path
) -> dict:

    output_dir.mkdir(parents=True, exist_ok=True)
    
    figures = {}

    return figures
