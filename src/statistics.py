import numpy as np
from scipy import stats
from scipy.stats import wilcoxon, friedmanchisquare, rankdata
from typing import Optional
import warnings


def wilcoxon_signed_rank_test(
    scores1: np.ndarray,
    scores2: np.ndarray,
    alternative: str = 'two-sided'
) -> dict:

    if len(scores1) != len(scores2):
        raise ValueError("Score arrays must have same length")
    
    if len(scores1) < 5:
        warnings.warn("Sample size < 5, results may be unreliable")
    
    differences = scores1 - scores2
    

    non_zero_mask = differences != 0
    if not np.any(non_zero_mask):
        return {
            'statistic': np.nan,
            'p_value': 1.0,
            'significant': False,
            'effect_size': 0.0,
            'interpretation': 'No difference between methods'
        }
    
    try:
        statistic, p_value = wilcoxon(
            scores1, scores2, 
            alternative=alternative,
            zero_method='wilcox'
        )
    except ValueError as e:
        return {
            'statistic': np.nan,
            'p_value': 1.0,
            'significant': False,
            'effect_size': 0.0,
            'interpretation': f'Test failed: {e}'
        }
    

    n = len(scores1)
    z_score = stats.norm.ppf(1 - p_value/2) if p_value < 1 else 0
    effect_size = abs(z_score) / np.sqrt(n)
    
    # Interpretation
    if effect_size < 0.1:
        effect_interp = 'negligible'
    elif effect_size < 0.3:
        effect_interp = 'small'
    elif effect_size < 0.5:
        effect_interp = 'medium'
    else:
        effect_interp = 'large'
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'effect_size': effect_size,
        'effect_interpretation': effect_interp,
        'mean_diff': np.mean(differences),
        'median_diff': np.median(differences),
        'interpretation': f"{'Significant' if p_value < 0.05 else 'No significant'} difference (p={p_value:.4f}, r={effect_size:.3f} {effect_interp})"
    }


def friedman_test(score_matrix: np.ndarray, method_names: list[str]) -> dict:

    n_datasets, n_methods = score_matrix.shape
    
    if n_datasets < 3:
        return {
            'statistic': np.nan,
            'p_value': 1.0,
            'significant': False,
            'interpretation': 'Need at least 3 datasets for Friedman test'
        }
    

    try:
        statistic, p_value = friedmanchisquare(*[score_matrix[:, i] for i in range(n_methods)])
    except Exception as e:
        return {
            'statistic': np.nan,
            'p_value': 1.0,
            'significant': False,
            'interpretation': f'Test failed: {e}'
        }
    

    ranks = np.zeros_like(score_matrix)
    for i in range(n_datasets):
        ranks[i] = rankdata(-score_matrix[i])  # Higher score = lower rank (better)
    
    avg_ranks = ranks.mean(axis=0)
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'average_ranks': {name: rank for name, rank in zip(method_names, avg_ranks)},
        'best_method': method_names[np.argmin(avg_ranks)],
        'interpretation': f"{'Significant' if p_value < 0.05 else 'No significant'} difference among methods (p={p_value:.4f})"
    }


def nemenyi_post_hoc(score_matrix: np.ndarray, method_names: list[str], alpha: float = 0.05) -> dict:

    n_datasets, n_methods = score_matrix.shape
    

    ranks = np.zeros_like(score_matrix)
    for i in range(n_datasets):
        ranks[i] = rankdata(-score_matrix[i])
    
    avg_ranks = ranks.mean(axis=0)
    
    # Critical difference (Nemenyi)
    # q_alpha values for alpha=0.05
    q_alpha = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102,
        10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313,
        14: 3.354, 15: 3.391, 16: 3.426, 17: 3.458,
        18: 3.489, 19: 3.517, 20: 3.544
    }
    
    q = q_alpha.get(n_methods, 3.5)  # default for large k
    cd = q * np.sqrt(n_methods * (n_methods + 1) / (6 * n_datasets))
    

    comparisons = []
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            diff = abs(avg_ranks[i] - avg_ranks[j])
            significant = diff > cd
            comparisons.append({
                'method1': method_names[i],
                'method2': method_names[j],
                'rank_diff': diff,
                'significant': significant
            })
    
    return {
        'critical_difference': cd,
        'average_ranks': {name: rank for name, rank in zip(method_names, avg_ranks)},
        'pairwise_comparisons': comparisons,
        'n_significant_pairs': sum(1 for c in comparisons if c['significant'])
    }


def generate_cd_diagram_data(
    score_matrix: np.ndarray,
    method_names: list[str],
    alpha: float = 0.05
) -> dict:

    n_datasets, n_methods = score_matrix.shape
    

    ranks = np.zeros_like(score_matrix)
    for i in range(n_datasets):
        ranks[i] = rankdata(-score_matrix[i])
    
    avg_ranks = ranks.mean(axis=0)
    

    sorted_indices = np.argsort(avg_ranks)
    sorted_names = [method_names[i] for i in sorted_indices]
    sorted_ranks = avg_ranks[sorted_indices]
    

    nemenyi = nemenyi_post_hoc(score_matrix, method_names, alpha)
    cd = nemenyi['critical_difference']
    

    cliques = []
    for i, name1 in enumerate(sorted_names):
        clique = [name1]
        for j, name2 in enumerate(sorted_names[i+1:], i+1):
            if abs(sorted_ranks[i] - sorted_ranks[j]) <= cd:
                clique.append(name2)
            else:
                break
        if len(clique) > 1:
            cliques.append({
                'methods': clique,
                'start_rank': sorted_ranks[i],
                'end_rank': sorted_ranks[sorted_names.index(clique[-1])]
            })
    

    unique_cliques = []
    for clique in cliques:
        is_subset = False
        for other in cliques:
            if clique != other and set(clique['methods']).issubset(set(other['methods'])):
                is_subset = True
                break
        if not is_subset:
            unique_cliques.append(clique)
    
    return {
        'methods': sorted_names,
        'ranks': sorted_ranks.tolist(),
        'critical_difference': cd,
        'cliques': unique_cliques,
        'n_datasets': n_datasets,
        'n_methods': n_methods
    }


def bootstrap_confidence_interval(
    scores: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 1000
) -> tuple[float, float]:

    np.random.seed(42)
    n = len(scores)
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return lower, upper


def compute_all_statistical_tests(
    results_matrix: np.ndarray,
    method_names: list[str],
    proposed_method_idx: int = 0
) -> dict:

    n_datasets, n_methods = results_matrix.shape
    proposed_scores = results_matrix[:, proposed_method_idx]
    

    friedman = friedman_test(results_matrix, method_names)
    

    nemenyi = nemenyi_post_hoc(results_matrix, method_names)
    

    pairwise_tests = {}
    for i, name in enumerate(method_names):
        if i != proposed_method_idx:
            test = wilcoxon_signed_rank_test(proposed_scores, results_matrix[:, i])
            pairwise_tests[name] = test
    

    confidence_intervals = {}
    for i, name in enumerate(method_names):
        lower, upper = bootstrap_confidence_interval(results_matrix[:, i])
        confidence_intervals[name] = {
            'mean': results_matrix[:, i].mean(),
            'std': results_matrix[:, i].std(),
            'ci_lower': lower,
            'ci_upper': upper
        }
    

    cd_data = generate_cd_diagram_data(results_matrix, method_names)
    

    wins, ties, losses = 0, 0, 0
    for i in range(n_methods):
        if i == proposed_method_idx:
            continue
        diff = proposed_scores - results_matrix[:, i]
        wins += np.sum(diff > 0.001)  # threshold for tie
        ties += np.sum(np.abs(diff) <= 0.001)
        losses += np.sum(diff < -0.001)
    
    return {
        'friedman_test': friedman,
        'nemenyi_post_hoc': nemenyi,
        'pairwise_wilcoxon': pairwise_tests,
        'confidence_intervals': confidence_intervals,
        'cd_diagram_data': cd_data,
        'win_tie_loss': {
            'wins': int(wins),
            'ties': int(ties),
            'losses': int(losses),
            'win_rate': wins / (wins + ties + losses) if (wins + ties + losses) > 0 else 0
        }
    }
