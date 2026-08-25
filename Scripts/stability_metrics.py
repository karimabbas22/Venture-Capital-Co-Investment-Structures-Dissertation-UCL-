"""
stability_metrics.py — canonical stability metrics for perturbation experiments.

evaluate_stability() is the single entry point: given a model's predicted
probabilities on the clean graph and on a perturbed graph (same startups,
same order), it returns one row of metrics. Both the tabular perturbation
runner (07) and the GNN perturbation runner (11) call this same function,
so every perturbation experiment — regardless of model architecture —
produces an identical CSV schema.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve


def prediction_flip_rate(prob_orig, prob_pert, threshold: float = 0.5) -> float:
    """Fraction of startups whose binary prediction (at `threshold`) changes
    between the clean and perturbed graph.

    NOTE: `threshold` here (from experiment_config.json's `flip_threshold`,
    fixed at 0.5) is deliberately independent of each model's own tuned
    F1-maximizing threshold (see pipeline_core.tune_threshold, used by 03/
    05/10 for classification metrics). Flip rate needs one fixed reference
    point shared across all models/perturbations/budgets for the metric to
    be comparable at all -- letting it float per model would conflate
    "prediction instability" with "different models have different
    thresholds," which is not what this metric is meant to measure."""
    pred_orig = (np.asarray(prob_orig) >= threshold).astype(int)
    pred_pert = (np.asarray(prob_pert) >= threshold).astype(int)
    return float(np.mean(pred_orig != pred_pert))


def jaccard_topk(prob_orig, prob_pert, k: int) -> float:
    """Overlap between the top-k highest-probability startups before and
    after perturbation."""
    top_a = set(np.argsort(prob_orig)[::-1][:k])
    top_b = set(np.argsort(prob_pert)[::-1][:k])
    if not top_a and not top_b:
        return 1.0
    return len(top_a & top_b) / len(top_a | top_b)


def calibration_drift(y_true, prob_orig, prob_pert, n_bins: int = 10) -> dict:
    """Change in Brier score, and L2 distance between the two reliability
    curves, as a measure of whether perturbation degrades calibration
    (not just discrimination)."""
    delta_brier = brier_score_loss(y_true, prob_pert) - brier_score_loss(y_true, prob_orig)
    try:
        frac_o, _ = calibration_curve(y_true, prob_orig, n_bins=n_bins, strategy="uniform")
        frac_p, _ = calibration_curve(y_true, prob_pert, n_bins=n_bins, strategy="uniform")
        n = min(len(frac_o), len(frac_p))
        reliability_l2 = float(np.sqrt(np.mean((frac_o[:n] - frac_p[:n]) ** 2))) if n > 0 else float("nan")
    except ValueError:
        reliability_l2 = float("nan")
    return {
        "delta_brier": round(float(delta_brier), 6),
        "reliability_curve_l2": round(reliability_l2, 6),
    }


def evaluate_stability(y_true, prob_orig, prob_pert,
                       threshold: float = 0.5,
                       topk_values: tuple[int, ...] = (25, 50, 100)) -> dict:
    """Single canonical stability-metrics row for one (model, perturbation,
    budget, seed) combination."""
    prob_orig = np.asarray(prob_orig)
    prob_pert = np.asarray(prob_pert)
    delta_p = np.abs(prob_orig - prob_pert)
    sp_rho, _ = spearmanr(prob_orig, prob_pert)

    try:
        roc_orig = roc_auc_score(y_true, prob_orig)
        roc_pert = roc_auc_score(y_true, prob_pert)
    except ValueError:
        roc_orig = roc_pert = float("nan")
    try:
        pr_orig = average_precision_score(y_true, prob_orig)
        pr_pert = average_precision_score(y_true, prob_pert)
    except ValueError:
        pr_orig = pr_pert = float("nan")

    row = {
        "roc_auc_clean": round(float(roc_orig), 4),
        "roc_auc_pert": round(float(roc_pert), 4),
        "delta_roc_auc": round(float(roc_pert - roc_orig), 4),
        "pr_auc_clean": round(float(pr_orig), 4),
        "pr_auc_pert": round(float(pr_pert), 4),
        "delta_pr_auc": round(float(pr_pert - pr_orig), 4),
        "flip_rate": round(prediction_flip_rate(prob_orig, prob_pert, threshold), 4),
        "mean_abs_delta_p": round(float(delta_p.mean()), 6),
        "median_abs_delta_p": round(float(np.median(delta_p)), 6),
        "p90_abs_delta_p": round(float(np.percentile(delta_p, 90)), 6),
        "spearman_rho": round(float(sp_rho), 6),
    }
    for k in topk_values:
        row[f"jaccard_top{k}"] = round(jaccard_topk(prob_orig, prob_pert, k), 4)
    row.update(calibration_drift(y_true, prob_orig, prob_pert))
    return row
