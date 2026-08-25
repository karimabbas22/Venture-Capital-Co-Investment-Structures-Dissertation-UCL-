"""
Experiments/significance_tests.py -- formal statistical inference on the
headline ROC-AUC comparisons: every delta reported elsewhere (baseline vs
network per model class, GNN vs best tabular, etc.) was a bare point
estimate with no significance test. With
test n~3,280 (~730 positives), Hanley-McNeil back-of-envelope AUC standard
errors are plausibly 0.01-0.02 -- comparable to or larger than several of
the reported deltas -- so this matters, not a formality.

Standalone, read-only with respect to Scripts/ and the trained artifacts:
reloads already-trained models (no retraining) and already-saved test
splits, computes predictions fresh, and runs:
  1. DeLong's test (Sun & Xu 2014 fast algorithm) for paired ROC-AUC
     comparisons on the same test set -- the statistically correct test
     here, since all models are scored on (a subset of) the same startups,
     not independent samples.
  2. Percentile bootstrap confidence intervals (5,000 resamples) on every
     individual model's ROC-AUC and on every delta, as a second, more
     general check that doesn't depend on DeLong's asymptotic assumptions.

DeLong implementation is validated against sklearn.metrics.roc_auc_score
before any p-value is trusted (see _validate_delong_auc()).

Run: python3 Experiments/significance_tests.py
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import joblib
import xgboost  # noqa: F401 -- must import before torch: importing torch
                # first causes a native OpenMP-library segfault on macOS
                # when an XGBoost model is later unpickled in-process
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Scripts")
sys.path.insert(0, SCRIPTS_DIR)
from pipeline_core import T_YEARS, transform_preprocessor
from gnn_model import HeteroGNN

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA      = os.path.join(ROOT, "Data", "processed")
MODEL_DIR = os.path.join(DATA, "models", f"T{T_YEARS}")
GRAPH_DIR = os.path.join(DATA, "graphs")
OUT_DIR   = os.path.join(ROOT, "Experiments", "output")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
N_BOOTSTRAP = 5000

BASELINE_FILES = {
    "LogReg": "logreg_baseline.pkl", "RandomForest": "rf_baseline.pkl",
    "HistGBM": "gbm_baseline.pkl", "XGBoost": "xgb_baseline.pkl",
    "LightGBM": "lgbm_baseline.pkl",
}
NETWORK_FILES = {
    "LogReg": "logreg_network.pkl", "RandomForest": "rf_network.pkl",
    "HistGBM": "gbm_network.pkl", "XGBoost": "xgb_network.pkl",
    "LightGBM": "lgbm_network.pkl",
}


# Fast DeLong (Sun & Xu 2014)
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted_transposed, m):
    """preds_sorted_transposed: (k models, n instances) array with the m
    positive-label columns first, negatives after. Returns (aucs, cov)."""
    n = preds_sorted_transposed.shape[1] - m
    pos = preds_sorted_transposed[:, :m]
    neg = preds_sorted_transposed[:, m:]
    k = preds_sorted_transposed.shape[0]

    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(preds_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, np.atleast_2d(delongcov)


def delong_test(y_true, prob_a, prob_b):
    """Paired DeLong test for two classifiers scored on the same instances.
    Returns (auc_a, auc_b, z_stat, p_value)."""
    y_true = np.asarray(y_true)
    order = np.argsort(-y_true, kind="mergesort")  # positives first
    y_sorted = y_true[order]
    m = int(y_sorted.sum())
    preds = np.vstack([np.asarray(prob_a)[order], np.asarray(prob_b)[order]])
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    var = max(var, 1e-12)
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)


def _validate_delong_auc(y_true, prob):
    """Sanity check: single-model DeLong AUC must match sklearn's
    roc_auc_score before any p-value from this module is trustworthy."""
    order = np.argsort(-np.asarray(y_true), kind="mergesort")
    y_sorted = np.asarray(y_true)[order]
    m = int(y_sorted.sum())
    preds = np.vstack([np.asarray(prob)[order], np.asarray(prob)[order]])
    aucs, _ = _fast_delong(preds, m)
    sklearn_auc = roc_auc_score(y_true, prob)
    diff = abs(aucs[0] - sklearn_auc)
    assert diff < 1e-9, f"DeLong AUC {aucs[0]} != sklearn AUC {sklearn_auc} (diff={diff})"
    return True


def bootstrap_auc_ci(y_true, prob, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true); prob = np.asarray(prob)
    n = len(y_true)
    boots = np.empty(n_boot)
    i = 0
    while i < n_boot:
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue  # need both classes present to define AUC
        boots[i] = roc_auc_score(y_true[idx], prob[idx])
        i += 1
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), boots


def bootstrap_delta_ci(y_true, prob_a, prob_b, n_boot=N_BOOTSTRAP, seed=SEED):
    """Paired bootstrap (same resampled indices for both models each draw)
    on AUC_a - AUC_b. Returns (delta, ci_low, ci_high, p_two_sided)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true); prob_a = np.asarray(prob_a); prob_b = np.asarray(prob_b)
    n = len(y_true)
    deltas = np.empty(n_boot)
    i = 0
    while i < n_boot:
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        deltas[i] = roc_auc_score(y_true[idx], prob_a[idx]) - roc_auc_score(y_true[idx], prob_b[idx])
        i += 1
    delta = roc_auc_score(y_true, prob_a) - roc_auc_score(y_true, prob_b)
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    # two-sided p-value: fraction of bootstrap deltas crossing zero, doubled
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return float(delta), float(ci_low), float(ci_high), float(min(p, 1.0))


# Load predictions for every model on the aligned test set
def load_all_predictions():
    # Aligned on company_name alone: verified zero duplicate company_name
    # values within this specific test split (33/3280 rows have a missing
    # company_permid, which breaks NaN-based tuple-key matching, so permid
    # is not used here -- company_name is the collision-free key for this
    # sample specifically, per the pipeline-wide identity-collision check).
    baseline_test = pd.read_parquet(os.path.join(MODEL_DIR, "split_test.parquet"))
    assert not baseline_test["company_name"].duplicated().any(), \
        "company_name is not unique in this test split -- alignment below is unsafe"
    baseline_test = baseline_test.sort_values("company_name").reset_index(drop=True)
    y = baseline_test["exit_within_T"].values

    preds = {}
    for name, fname in BASELINE_FILES.items():
        bundle = joblib.load(os.path.join(MODEL_DIR, fname))
        X = transform_preprocessor(bundle["preprocessor"], baseline_test)
        preds[f"{name}_Baseline"] = bundle["model"].predict_proba(X)[:, 1]

    # Network models were trained on firms_with_network_T7.parquet -- re-derive
    # that script's test split, then align to the same row order as baseline.
    import pipeline_core as pc
    firms_net = pd.read_parquet(os.path.join(DATA, "firms_with_network_T7.parquet"))
    firms_net_eng = pc.engineer_network_features(firms_net, "exit_within_T", "first_deal_date")
    _, _, net_test = pc.temporal_train_val_test_split(firms_net_eng, pc.TRAIN_END, pc.VAL_END)
    net_test = net_test.set_index("company_name").loc[baseline_test["company_name"]].reset_index()
    assert (net_test["exit_within_T"].values == y).all(), "network test labels misaligned with baseline"

    for name, fname in NETWORK_FILES.items():
        bundle = joblib.load(os.path.join(MODEL_DIR, fname))
        X = transform_preprocessor(bundle["preprocessor"], net_test)
        preds[f"{name}_Network"] = bundle["model"].predict_proba(X)[:, 1]

    # GNN: align test_graph.startup_ids (company_name, company_permid tuples)
    # to the same row order via company_name only, same rationale as above.
    test_graph = torch.load(os.path.join(GRAPH_DIR, "test_graph.pt"), weights_only=False)
    gnn_bundle = torch.load(os.path.join(MODEL_DIR, "gnn_model.pt"), weights_only=False)
    model = HeteroGNN(gnn_bundle["edge_types"], hidden_dim=gnn_bundle["params"]["hidden_dim"],
                      dropout=gnn_bundle["params"]["dropout"])
    with torch.no_grad():
        model(test_graph.x_dict, test_graph.edge_index_dict)
    model.load_state_dict(gnn_bundle["model_state"])
    model.eval()
    with torch.no_grad():
        gnn_prob_raw = torch.sigmoid(model(test_graph.x_dict, test_graph.edge_index_dict)).numpy()
    gnn_y_raw = test_graph["startup"].y.numpy()
    gnn_names = [sid[0] for sid in test_graph.startup_ids]
    assert len(set(gnn_names)) == len(gnn_names), "company_name not unique in GNN test graph"
    gnn_id_to_pos = {name: i for i, name in enumerate(gnn_names)}
    order = [gnn_id_to_pos[cn] for cn in baseline_test["company_name"]]
    preds["GNN"] = gnn_prob_raw[order]
    assert (gnn_y_raw[order] == y).all(), "GNN test labels misaligned with baseline"

    return y, preds


def main():
    print("\nSignificance tests: headline ROC-AUC comparisons")
    y, preds = load_all_predictions()
    print(f"Aligned test set: {len(y):,} startups, {int(y.sum()):,} positive\n")

    print("Validating DeLong AUC against sklearn.roc_auc_score for every model...")
    for name, p in preds.items():
        _validate_delong_auc(y, p)
    print("  OK -- all match to <1e-9\n")

    print(f"{'Model':<20s} {'ROC-AUC':>8s} {'95% Bootstrap CI':>20s}")
    auc_rows = []
    for name, p in preds.items():
        auc = roc_auc_score(y, p)
        lo, hi, _ = bootstrap_auc_ci(y, p)
        print(f"{name:<20s} {auc:>8.4f}   [{lo:.4f}, {hi:.4f}]")
        auc_rows.append({"model": name, "roc_auc": round(auc, 4),
                         "ci_low": round(lo, 4), "ci_high": round(hi, 4)})

    comparisons = [
        ("LogReg_Network", "LogReg_Baseline"), ("RandomForest_Network", "RandomForest_Baseline"),
        ("HistGBM_Network", "HistGBM_Baseline"), ("XGBoost_Network", "XGBoost_Baseline"),
        ("LightGBM_Network", "LightGBM_Baseline"),
        ("GNN", "RandomForest_Baseline"), ("GNN", "LogReg_Network"),
        ("LogReg_Network", "RandomForest_Baseline"),
    ]

    print("\nPaired comparisons (DeLong test + bootstrap)")
    print(f"{'A vs B':<38s} {'ΔAUC':>8s} {'DeLong p':>10s} {'Boot 95% CI':>20s} {'Boot p':>8s}  Sig?")
    comp_rows = []
    for a, b in comparisons:
        auc_a, auc_b, z, p_delong = delong_test(y, preds[a], preds[b])
        delta, ci_lo, ci_hi, p_boot = bootstrap_delta_ci(y, preds[a], preds[b])
        sig = "***" if p_delong < 0.01 else ("**" if p_delong < 0.05 else ("*" if p_delong < 0.10 else "ns"))
        label = f"{a} vs {b}"
        print(f"{label:<38s} {delta:>+8.4f} {p_delong:>10.4f}   [{ci_lo:+.4f}, {ci_hi:+.4f}]  {p_boot:>7.4f}  {sig}")
        comp_rows.append({
            "comparison": label, "auc_a": round(auc_a, 4), "auc_b": round(auc_b, 4),
            "delta": round(delta, 4), "delong_z": round(z, 4), "delong_p": round(p_delong, 4),
            "boot_ci_low": round(ci_lo, 4), "boot_ci_high": round(ci_hi, 4),
            "boot_p": round(p_boot, 4), "significant_at_05": p_delong < 0.05,
        })

    print("\n(ns = not significant at 10%; * p<0.10, ** p<0.05, *** p<0.01, DeLong test)")

    pd.DataFrame(auc_rows).to_csv(os.path.join(OUT_DIR, "significance_auc_estimates.csv"), index=False)
    pd.DataFrame(comp_rows).to_csv(os.path.join(OUT_DIR, "significance_comparisons.csv"), index=False)
    with open(os.path.join(OUT_DIR, "significance_summary.json"), "w") as f:
        json.dump({"n_test": len(y), "n_positive": int(y.sum()),
                   "auc_estimates": auc_rows, "comparisons": comp_rows}, f, indent=2)
    print(f"\nSaved -> {OUT_DIR}/significance_auc_estimates.csv, significance_comparisons.csv")


if __name__ == "__main__":
    main()
