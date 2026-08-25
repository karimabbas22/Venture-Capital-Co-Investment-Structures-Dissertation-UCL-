"""
07_run_perturbation_tests.py — test whether graph-based predictions are
robust to perturbations of the investor co-investment network.

Protocol: for each perturbation scenario (type x budget x seed, or hub
removal by K), perturb the investor-investor graph, recompute network
features, rerun ALL trained network models with the new features, and
compare predictions against each model's own unperturbed (clean) baseline
via stability_metrics.evaluate_stability(). Perturbed features are computed
ONCE per (type, budget, seed) combination and reused across all 5 models —
graph rebuilding dominates runtime, model scoring is comparatively free.

Config (budgets, seeds, perturbation types) lives in experiment_config.json
so scenarios can be adjusted without touching this script.
"""

import os
import sys
import json
import time

import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_core import (
    SEED, T_YEARS, BETWEENNESS_K,
    build_coinvestment_snapshot, compute_snapshot_metrics,
    temporal_train_val_test_split, transform_preprocessor,
    TRAIN_END, VAL_END, set_global_seed,
    engineer_network_features,
)
from perturbation_core import perturb_graph, hub_removal, load_experiment_config
from stability_metrics import evaluate_stability

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA        = os.path.join(ROOT, "Data", "processed")
MODEL_DIR   = os.path.join(DATA, "models", f"T{T_YEARS}")
OUT_DIR     = os.path.join(MODEL_DIR, "perturbation")
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_config.json")
os.makedirs(OUT_DIR, exist_ok=True)

METRIC_COLS = ["degree", "weighted_degree", "betweenness_centrality",
               "pagerank", "eigenvector_lcc", "clustering_coeff"]

MODEL_FILES = {
    "LogReg":       "logreg_network.pkl",
    "HistGBM":      "gbm_network.pkl",
    "XGBoost":      "xgb_network.pkl",
    "LightGBM":     "lgbm_network.pkl",
    "RandomForest": "rf_network.pkl",
}


def aggregate_network_to_startup(die_fp: pd.DataFrame, inv_metrics: pd.DataFrame,
                                 snapshot_col: str = "snapshot_date") -> pd.DataFrame:
    """Recompute startup-level network features from perturbed investor metrics.
    Only the columns the current reduced model actually uses are aggregated."""
    merged = die_fp.merge(
        inv_metrics[["investor_org_id", snapshot_col] + METRIC_COLS],
        on=["investor_org_id", snapshot_col], how="left", suffixes=("_old", ""),
    )
    for col in METRIC_COLS:
        if f"{col}_old" in merged.columns:
            merged[col] = merged[col].fillna(merged[f"{col}_old"])
            merged.drop(columns=[f"{col}_old"], inplace=True)

    merged["known_at_snapshot"] = merged["degree"].notna()

    return (
        merged.groupby("company_name")
        .agg(
            n_investors_first_round=("investor_org_id", "nunique"),
            n_known_investors=("known_at_snapshot", "sum"),
            mean_pagerank=("pagerank", "mean"),
            max_pagerank=("pagerank", "max"),
            mean_eigenvector_lcc=("eigenvector_lcc", "mean"),
        )
        .reset_index()
    )


def engineer_test_features(test: pd.DataFrame) -> pd.DataFrame:
    """Reduced-feature engineering, shared with script 05 via pipeline_core."""
    return engineer_network_features(test, "exit_within_T", "first_deal_date")


def main():
    set_global_seed()
    cfg = load_experiment_config(CONFIG_PATH)

    print(f"{'='*70}\n  PERTURBATION ANALYSIS (all network models)\n{'='*70}")
    print(f"Config: budgets={cfg['budgets']}  seeds={cfg['seeds']}  "
          f"types={cfg['perturbation_types']}  hub_k={cfg['hub_removal_k']}")

    models = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    if not models:
        print("[ERROR] No trained network models found — run script 05 first")
        return
    print(f"Loaded {len(models)} network models: {', '.join(models.keys())}")

    deals = pd.read_parquet(os.path.join(DATA, "deals.parquet"))
    deals = deals[deals["investor_name"] != "Undisclosed Firm"].copy()
    deals["investor_org_id"] = deals["investor_org_id"].fillna(
        deals["investor_name"]).astype(str)

    firms = pd.read_parquet(os.path.join(DATA, "firms_with_network_T7.parquet"))
    firms_eng = firms.dropna(subset=["exit_within_T", "first_deal_date"]).copy()
    _, _, test = temporal_train_val_test_split(firms_eng, TRAIN_END, VAL_END)
    print(f"Test startups: {len(test):,}")

    fp = firms[firms["company_name"].isin(test["company_name"])][
        ["company_name", "first_deal_date", "snapshot_date"]].copy()
    die_fp = deals.merge(fp, on="company_name", how="inner")
    die_fp = die_fp[die_fp["investment_date"] <= die_fp["first_deal_date"]]
    die_fp = die_fp.drop_duplicates(subset=["company_name", "investor_org_id"])

    needed_snaps = sorted(fp["snapshot_date"].unique())
    print(f"Test cohort uses {len(needed_snaps)} snapshots")

    test_eng = engineer_test_features(test)
    y_test = test_eng["exit_within_T"].values

    orig_probs = {}
    for name, bundle in models.items():
        X_orig = transform_preprocessor(bundle["preprocessor"], test_eng)
        orig_probs[name] = bundle["model"].predict_proba(X_orig)[:, 1]
        print(f"  {name:12s} clean test AUC={roc_auc_score(y_test, orig_probs[name]):.4f}")

    inv_bs = pd.read_parquet(os.path.join(DATA, "investor_features_by_snapshot.parquet"))
    last_snap = max(needed_snaps)
    hub_candidates = (
        inv_bs[inv_bs["snapshot_date"] == last_snap]
        .sort_values("degree", ascending=False)
    )

    all_rows = []

    for name in models:
        row = evaluate_stability(y_test, orig_probs[name], orig_probs[name],
                                 threshold=cfg["flip_threshold"],
                                 topk_values=tuple(cfg["topk_values"]))
        row.update({"model": name, "perturb_type": "clean", "budget": 0.0, "seed": -1})
        all_rows.append(row)

    def build_perturbed_test(perturb_fn):
        pert_metrics = []
        for snap in needed_snaps:
            G = build_coinvestment_snapshot(deals, pd.Timestamp(snap))
            G_pert = perturb_fn(G)
            m = compute_snapshot_metrics(G_pert, betweenness_k=BETWEENNESS_K, seed=SEED)
            m["snapshot_date"] = pd.Timestamp(snap)
            pert_metrics.append(m)
        pert_inv = pd.concat(pert_metrics, ignore_index=True)
        pert_startup = aggregate_network_to_startup(die_fp, pert_inv)

        test_pert = test_eng.drop(
            columns=[c for c in pert_startup.columns
                    if c != "company_name" and c in test_eng.columns],
            errors="ignore")
        test_pert = test_pert.merge(pert_startup, on="company_name", how="left")
        test_pert["mean_pagerank"] = test_pert["mean_pagerank"].fillna(0.0)
        test_pert["missing_evc"] = test_pert["mean_eigenvector_lcc"].isna().astype(int)
        return test_pert

    def score_all_models(test_pert):
        return {name: bundle["model"].predict_proba(
                    transform_preprocessor(bundle["preprocessor"], test_pert))[:, 1]
                for name, bundle in models.items()}

    # ── Edge-level, budget-indexed perturbations ────────────────────────────
    for ptype in cfg["perturbation_types"]:
        for budget in cfg["budgets"]:
            for seed in cfg["seeds"]:
                t0 = time.time()
                test_pert = build_perturbed_test(
                    lambda G, p=ptype, b=budget, s=seed: perturb_graph(G, p, b, s))
                probs_pert = score_all_models(test_pert)
                for name in models:
                    row = evaluate_stability(y_test, orig_probs[name], probs_pert[name],
                                             threshold=cfg["flip_threshold"],
                                             topk_values=tuple(cfg["topk_values"]))
                    row.update({"model": name, "perturb_type": ptype,
                               "budget": budget, "seed": seed})
                    all_rows.append(row)
                print(f"  {ptype:26s} budget={budget:<6} seed={seed}  "
                      f"({len(models)} models, {time.time()-t0:.1f}s)")

    # ── Hub removal (K-indexed, not budget-indexed) ─────────────────────────
    for K in cfg["hub_removal_k"]:
        t0 = time.time()
        hub_ids = hub_candidates.head(K)["investor_org_id"].tolist()
        test_pert = build_perturbed_test(lambda G, ids=hub_ids: hub_removal(G, ids))
        probs_pert = score_all_models(test_pert)
        for name in models:
            row = evaluate_stability(y_test, orig_probs[name], probs_pert[name],
                                     threshold=cfg["flip_threshold"],
                                     topk_values=tuple(cfg["topk_values"]))
            row.update({"model": name, "perturb_type": "hub_removal",
                       "budget": None, "seed": -1, "hub_k": K})
            all_rows.append(row)
        print(f"  hub_removal K={K:<4}  ({len(models)} models, {time.time()-t0:.1f}s)")

    # ── Save ─────────────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv(os.path.join(OUT_DIR, "perturbation_summary.csv"), index=False)
    with open(os.path.join(OUT_DIR, "perturbation_results.json"), "w") as f:
        json.dump({
            "clean_auc": {name: round(float(roc_auc_score(y_test, orig_probs[name])), 4)
                         for name in models},
            "n_test_startups": len(test),
            "config": cfg,
        }, f, indent=2)

    print(f"\n{'='*70}\n  PERTURBATION SUMMARY (mean over seeds, by model/type/budget)\n{'='*70}")
    agg_cols = ["delta_roc_auc", "flip_rate", "mean_abs_delta_p", "spearman_rho",
               "jaccard_top50", "delta_brier"]
    perturbed = summary_df[summary_df["perturb_type"] != "clean"]
    agg = perturbed.groupby(["model", "perturb_type", "budget"], dropna=False)[agg_cols].mean().round(5)
    print(agg.to_string())
    print(f"\nAll outputs → {OUT_DIR}")


if __name__ == "__main__":
    main()
