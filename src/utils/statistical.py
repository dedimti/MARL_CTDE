"""
src/utils/statistical.py
Statistical Analysis for MARL-CTDE Evaluation

Implements:
    - Wilcoxon signed-rank test (non-parametric comparison)
    - 95% Confidence Intervals (bootstrap method)
    - Cohen's d effect size
    - Jain's Fairness Index

Used for Table 7 (Statistical Analysis) in the paper.

References:
    [40] Adler et al. (2025) - Optimized resource allocation RL
    [46] Jiang et al. (2024) - Distributional RL batch jobs cloud
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


def wilcoxon_test(
    scores_a: List[float],
    scores_b: List[float],
    alternative: str = 'greater'
) -> Dict:
    """
    Wilcoxon signed-rank test for paired samples.
    Non-parametric — does not assume normal distribution.

    H0: Median difference = 0
    H1: Median of A > Median of B (alternative='greater')

    Args:
        scores_a: Performance scores of method A (our MARL)
        scores_b: Performance scores of method B (baseline)
        alternative: 'greater', 'less', or 'two-sided'

    Returns:
        results: Dict with statistic, p_value, significant, effect_size
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(f"Samples must be paired. Got {len(scores_a)} vs {len(scores_b)}")

    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)

    stat, p_value = stats.wilcoxon(scores_a, scores_b, alternative=alternative)

    return {
        'statistic': float(stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'n_samples': len(scores_a),
        'median_a': float(np.median(scores_a)),
        'median_b': float(np.median(scores_b)),
        'median_improvement': float(np.median(scores_a) - np.median(scores_b)),
        'test': 'Wilcoxon signed-rank'
    }


def bootstrap_ci(
    data: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval estimation.

    Args:
        data: Sample data
        confidence: Confidence level (default 0.95)
        n_bootstrap: Number of bootstrap iterations
        seed: Random seed for reproducibility

    Returns:
        (mean, lower_ci, upper_ci)
    """
    rng = np.random.default_rng(seed)
    data = np.array(data)
    n = len(data)

    bootstrap_means = np.array([
        np.mean(rng.choice(data, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return float(np.mean(data)), float(lower), float(upper)


def cohens_d(
    group_a: List[float],
    group_b: List[float]
) -> Dict:
    """
    Compute Cohen's d effect size.

    Interpretation:
        |d| < 0.2  : negligible
        |d| < 0.5  : small
        |d| < 0.8  : medium
        |d| >= 0.8 : large

    Args:
        group_a: Scores from method A
        group_b: Scores from method B

    Returns:
        Dict with d, magnitude, interpretation
    """
    a = np.array(group_a)
    b = np.array(group_b)

    mean_diff = np.mean(a) - np.mean(b)
    pooled_std = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)

    if pooled_std == 0:
        d = 0.0
    else:
        d = mean_diff / pooled_std

    # Interpret magnitude
    abs_d = abs(d)
    if abs_d < 0.2:
        magnitude = "negligible"
    elif abs_d < 0.5:
        magnitude = "small"
    elif abs_d < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"

    return {
        'cohens_d': float(d),
        'abs_d': float(abs_d),
        'magnitude': magnitude,
        'mean_a': float(np.mean(a)),
        'mean_b': float(np.mean(b)),
        'std_a': float(np.std(a, ddof=1)),
        'std_b': float(np.std(b, ddof=1))
    }


def full_statistical_report(
    our_method: Dict[str, List[float]],
    baselines: Dict[str, Dict[str, List[float]]],
    metrics: List[str] = ['makespan', 'energy', 'sla_violation_rate',
                           'resource_utilization', 'convergence_speed']
) -> Dict:
    """
    Generate full statistical report comparing our method vs all baselines.
    Produces data for Table 7 in the paper.

    Args:
        our_method: Dict[metric_name -> list of scores] for MARL-CTDE
        baselines: Dict[baseline_name -> Dict[metric_name -> list of scores]]
        metrics: List of metric names to compare

    Returns:
        report: Full statistical report with all comparisons
    """
    report = {}

    for baseline_name, baseline_scores in baselines.items():
        report[baseline_name] = {}

        for metric in metrics:
            if metric not in our_method or metric not in baseline_scores:
                continue

            a = our_method[metric]
            b = baseline_scores[metric]

            # For metrics where lower is better (makespan, energy, SLA violation)
            # Our method should have LOWER values → H1: A < B
            lower_is_better = metric in ['makespan', 'energy',
                                          'sla_violation_rate', 'convergence_speed']
            alternative = 'less' if lower_is_better else 'greater'

            # Wilcoxon test
            wilcoxon = wilcoxon_test(a, b, alternative=alternative)

            # Confidence intervals
            mean_a, ci_low_a, ci_high_a = bootstrap_ci(a)
            mean_b, ci_low_b, ci_high_b = bootstrap_ci(b)

            # Effect size
            effect = cohens_d(a, b)

            # Improvement percentage
            if np.mean(b) != 0:
                improvement_pct = (np.mean(b) - np.mean(a)) / abs(np.mean(b)) * 100
                if not lower_is_better:
                    improvement_pct = -improvement_pct
            else:
                improvement_pct = 0.0

            report[baseline_name][metric] = {
                'our_mean': mean_a,
                'our_ci': (ci_low_a, ci_high_a),
                'baseline_mean': mean_b,
                'baseline_ci': (ci_low_b, ci_high_b),
                'improvement_pct': float(improvement_pct),
                'wilcoxon': wilcoxon,
                'cohens_d': effect,
                'significant': wilcoxon['significant']
            }

    return report


def format_statistical_table(report: Dict) -> str:
    """
    Format statistical report as LaTeX table for paper.
    Produces Table 7.
    """
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Statistical Comparison: MARL-CTDE vs Baselines (Wilcoxon Test, $p < 0.05$)}",
        r"\label{tab:statistical}",
        r"\begin{tabular}{llrrrrr}",
        r"\hline",
        r"Baseline & Metric & Our Mean & Baseline Mean & Improv. (\%) & Cohen's $d$ & $p$-value \\",
        r"\hline"
    ]

    for baseline_name, metrics in report.items():
        first = True
        for metric, results in metrics.items():
            sig_marker = "**" if results['significant'] else ""
            lines.append(
                f"{baseline_name if first else ''} & "
                f"{metric.replace('_', ' ')} & "
                f"{results['our_mean']:.3f} & "
                f"{results['baseline_mean']:.3f} & "
                f"{results['improvement_pct']:+.1f}\\% & "
                f"{results['cohens_d']['cohens_d']:.3f} & "
                f"{results['wilcoxon']['p_value']:.4f}{sig_marker} \\\\"
            )
            first = False
        lines.append(r"\hline")

    lines += [
        r"\end{tabular}",
        r"\footnotesize{** $p < 0.05$, Wilcoxon signed-rank test, 5 independent runs}",
        r"\end{table}"
    ]

    return "\n".join(lines)
