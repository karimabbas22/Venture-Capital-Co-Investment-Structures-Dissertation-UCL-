"""
Experiments/placebo_hub_removal.py -- placebo test for the hub-removal
robustness finding, addressing a supervisor-review point: the existing
hub-removal result (removing the top-K most-connected investors is far
more damaging than random edge noise) is consistent with "centrality
specifically matters," but is also consistent with a weaker, less
interesting explanation -- "removing ANY K nodes from a sparse graph is
this damaging, regardless of which K." Without a placebo, these two
explanations are observationally confounded in the existing results.

This script removes K RANDOM (non-hub) investors instead of the top-K by
degree, at the same K grid as experiment_config.json's hub_removal_k
(densified to every integer 1-25), repeated across the same 5 seeds used
elsewhere (random selection needs repetition; true hub removal is
deterministic and does not). Same models, same test set, same
stability_metrics.evaluate_stability() schema as scripts 07/11, so the
placebo numbers merge directly against the existing hub_removal rows in
robustness_summary_wide.csv for a same-scale comparison.

Standalone and read-only with respect to Scripts/ and trained artifacts --
reuses perturbation_core.hub_removal() (which already accepts an arbitrary
node-ID list, not just top-K-by-degree, so no new perturbation primitive
is needed) and graph_core.remove_investor_nodes() unmodified.

MUST run AFTER Scripts/11_aggregate_robustness_report.py -- this script's
comparison step reads Data/processed/models/T7/perturbation/
robustness_summary_wide.csv (written by script 11), and if 11 hasn't run
against the densified hub_removal_k grid yet, that file is still the old
K in {5,10,25} summary. Correct order: 07 -> 10 -> 11 -> this script ->
Experiments/plot_realism_and_placebo_curves.py.

Run: python3 Experiments/placebo_hub_removal.py
"""

import os
import sys
import json
import copy
import time

import numpy as np
import pandas as pd
import xgboost  # noqa: F401 -- import before torch, see significance_tests.py
import joblib
import torch

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Scripts")
sys.path.insert(0, SCRIPTS_DIR)
from pipeline_core import (
    SEED, T_YEARS, BETWEENNESS_K, TRAIN_END, VAL_END,
    build_coinvestment_snapshot, compute_snapshot_metrics,
    temporal_train_val_test_split, transform_preprocessor,
    engineer_network_features, set_global_seed,
)
from perturbation_core import (
    hub_removal, load_experiment_config, hetero_edges_to_nx, recompute_investor_features,
)
from stability_metrics import evaluate_stability
from gnn_model import HeteroGNN
from graph_core import remove_investor_nodes

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA        = os.path.join(ROOT, "Data", "processed")
GRAPH_DIR   = os.path.join(DATA, "graphs")
MODEL_DIR   = os.path.join(DATA, "models", f"T{T_YEARS}")
PERT_DIR    = os.path.join(MODEL_DIR, "perturbation")
OUT_DIR     = os.path.join(ROOT, "Experiments", "output")
CONFIG_PATH = os.path.join(SCRIPTS_DIR, "experiment_config.json")
os.makedirs(OUT_DIR, exist_ok=True)

METRIC_COLS = ["degree", "weighted_degree", "betweenness_centrality",
               "pagerank", "eigenvector_lcc", "clustering_coeff"]

MODEL_FILES = {
    "LogReg": "logreg_network.pkl", "HistGBM": "gbm_network.pkl",
    "XGBoost": "xgb_network.pkl", "LightGBM": "lgbm_network.pkl",
    "RandomForest": "rf_network.pkl",
}


def aggregate_network_to_startup(die_fp: pd.DataFrame, inv_metrics: pd.DataFrame,
                                 snapshot_col: str = "snapshot_date") -> pd.DataFrame:
    merged = die_fp.merge(
        inv_metrics[["investor_org_id", snapshot_col] + METRIC_COLS],
        on=["investor_org_id", snapshot_col], how="left", suffixes=("_old", ""))
    for col in METRIC_COLS:
        if f"{col}_old" in merged.columns:
            merged[col] = merged[col].fillna(merged[f"{col}_old"])
            merged.drop(columns=[f"{col}_old"], inplace=True)
    merged["known_at_snapshot"] = merged["degree"].notna()
    return (merged.groupby("company_name")
            .agg(n_investors_first_round=("investor_org_id", "nunique"),
                n_known_investors=("known_at_snapshot", "sum"),
                mean_pagerank=("pagerank", "mean"),
                max_pagerank=("pagerank", "max"),
                mean_eigenvector_lcc=("eigenvector_lcc", "mean"))
            .reset_index())


def run_tabular_placebo(cfg):
    print(f"\n{'='*70}\n  TABULAR PLACEBO: random-K investor removal vs true hub removal\n{'='*70}")

    models = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    print(f"Loaded {len(models)} network models")

    deals = pd.read_parquet(os.path.join(DATA, "deals.parquet"))
    deals = deals[deals["investor_name"] != "Undisclosed Firm"].copy()
    deals["investor_org_id"] = deals["investor_org_id"].fillna(deals["investor_name"]).astype(str)

    firms = pd.read_parquet(os.path.join(DATA, "firms_with_network_T7.parquet"))
    firms_eng = firms.dropna(subset=["exit_within_T", "first_deal_date"]).copy()
    _, _, test = temporal_train_val_test_split(firms_eng, TRAIN_END, VAL_END)

    fp = firms[firms["company_name"].isin(test["company_name"])][
        ["company_name", "first_deal_date", "snapshot_date"]].copy()
    die_fp = deals.merge(fp, on="company_name", how="inner")
    die_fp = die_fp[die_fp["investment_date"] <= die_fp["first_deal_date"]]
    die_fp = die_fp.drop_duplicates(subset=["company_name", "investor_org_id"])
    needed_snaps = sorted(fp["snapshot_date"].unique())

    test_eng = engineer_network_features(test, "exit_within_T", "first_deal_date")
    y_test = test_eng["exit_within_T"].values

    orig_probs = {}
    for name, bundle in models.items():
        X_orig = transform_preprocessor(bundle["preprocessor"], test_eng)
        orig_probs[name] = bundle["model"].predict_proba(X_orig)[:, 1]

    inv_bs = pd.read_parquet(os.path.join(DATA, "investor_features_by_snapshot.parquet"))
    last_snap = max(needed_snaps)
    snap_investors = inv_bs[inv_bs["snapshot_date"] == last_snap]["investor_org_id"].tolist()
    print(f"Investor pool at last test-cohort snapshot: {len(snap_investors):,}")

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
            columns=[c for c in pert_startup.columns if c != "company_name" and c in test_eng.columns],
            errors="ignore")
        test_pert = test_pert.merge(pert_startup, on="company_name", how="left")
        test_pert["mean_pagerank"] = test_pert["mean_pagerank"].fillna(0.0)
        test_pert["missing_evc"] = test_pert["mean_eigenvector_lcc"].isna().astype(int)
        return test_pert

    def score_all_models(test_pert):
        return {name: bundle["model"].predict_proba(
                    transform_preprocessor(bundle["preprocessor"], test_pert))[:, 1]
                for name, bundle in models.items()}

    all_rows = []
    for K in cfg["hub_removal_k"]:
        for seed in cfg["seeds"]:
            t0 = time.time()
            rng = np.random.default_rng(seed * 1000 + K)  # distinct stream per (K, seed)
            random_ids = rng.choice(snap_investors, size=min(K, len(snap_investors)), replace=False).tolist()
            test_pert = build_perturbed_test(lambda G, ids=random_ids: hub_removal(G, ids))
            probs_pert = score_all_models(test_pert)
            for name in models:
                row = evaluate_stability(y_test, orig_probs[name], probs_pert[name],
                                         threshold=cfg["flip_threshold"],
                                         topk_values=tuple(cfg["topk_values"]))
                row.update({"model": name, "perturb_type": "placebo_random_removal",
                           "hub_k": K, "seed": seed})
                all_rows.append(row)
            print(f"  random-removal K={K:<4} seed={seed}  ({len(models)} models, {time.time()-t0:.1f}s)")

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUT_DIR, "placebo_tabular_random_removal.csv"), index=False)
    return df


COINV_RELATION = ("investor", "co_invests_with", "investor")


def run_gnn_placebo(cfg):
    print(f"\n{'='*70}\n  GNN PLACEBO: random-K investor node removal vs true hub removal\n{'='*70}")

    test_path = os.path.join(GRAPH_DIR, "test_graph.pt")
    model_path = os.path.join(MODEL_DIR, "gnn_model.pt")
    prep_path = os.path.join(GRAPH_DIR, "preprocessors.pt")
    test_data = torch.load(test_path, weights_only=False)
    bundle = torch.load(model_path, weights_only=False)
    investor_preprocessor = torch.load(prep_path, weights_only=False)["investor_preprocessor"]

    model = HeteroGNN(bundle["edge_types"], hidden_dim=bundle["params"]["hidden_dim"],
                      dropout=bundle["params"]["dropout"])
    with torch.no_grad():
        model(test_data.x_dict, test_data.edge_index_dict)
    model.load_state_dict(bundle["model_state"])
    model.eval()

    def predict(data):
        with torch.no_grad():
            return torch.sigmoid(model(data.x_dict, data.edge_index_dict)).numpy()

    y_test = test_data["startup"].y.numpy()
    orig_probs = predict(test_data)
    n_investor = test_data["investor"].num_nodes
    print(f"Investor nodes in test graph: {n_investor:,}")

    all_rows = []
    for K in cfg["hub_removal_k"]:
        for seed in cfg["seeds"]:
            t0 = time.time()
            rng = np.random.default_rng(seed * 1000 + K)
            remove_idx = set(rng.choice(n_investor, size=min(K, n_investor), replace=False).tolist())
            pert_data = remove_investor_nodes(test_data, remove_idx)
            num_investor_kept = pert_data["investor"].num_nodes
            G_kept = hetero_edges_to_nx(pert_data[COINV_RELATION].edge_index, num_investor_kept)
            pert_data["investor"].x = recompute_investor_features(
                G_kept, num_investor_kept, investor_preprocessor)
            probs = predict(pert_data)
            row = evaluate_stability(y_test, orig_probs, probs,
                                     threshold=cfg["flip_threshold"],
                                     topk_values=tuple(cfg["topk_values"]))
            row.update({"model": "GNN", "perturb_type": "placebo_random_removal",
                       "hub_k": K, "seed": seed})
            all_rows.append(row)
            print(f"  random-removal K={K:<4} seed={seed}  ({time.time()-t0:.2f}s)")

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUT_DIR, "placebo_gnn_random_removal.csv"), index=False)
    return df


def compare_against_true_hub_removal(tabular_placebo, gnn_placebo):
    print(f"\n{'='*70}\n  COMPARISON: true hub removal vs placebo random removal\n{'='*70}")

    true_hub = pd.read_csv(os.path.join(PERT_DIR, "robustness_summary_wide.csv"))
    true_hub = true_hub[true_hub["perturb_type"] == "hub_removal"]

    placebo_all = pd.concat([tabular_placebo, gnn_placebo], ignore_index=True)
    placebo_agg = placebo_all.groupby(["model", "hub_k"]).agg(
        delta_roc_auc_mean=("delta_roc_auc", "mean"),
        delta_roc_auc_std=("delta_roc_auc", "std"),
        spearman_rho_mean=("spearman_rho", "mean"),
        flip_rate_mean=("flip_rate", "mean"),
    ).reset_index()

    print(f"\n{'Model':<14s}{'K':>4s}{'True hub ΔAUC':>16s}{'Placebo ΔAUC':>16s}{'True ρ':>9s}{'Placebo ρ':>11s}")
    rows = []
    for _, tr in true_hub.iterrows():
        pl = placebo_agg[(placebo_agg["model"] == tr["model"]) & (placebo_agg["hub_k"] == tr["hub_k"])]
        if pl.empty:
            continue
        pl = pl.iloc[0]
        print(f"{tr['model']:<14s}{int(tr['hub_k']):>4d}{tr['delta_roc_auc_mean']:>16.4f}"
              f"{pl['delta_roc_auc_mean']:>16.4f}{tr['spearman_rho_mean']:>9.4f}{pl['spearman_rho_mean']:>11.4f}")
        rows.append({
            "model": tr["model"], "hub_k": int(tr["hub_k"]),
            "true_hub_delta_roc_auc": round(float(tr["delta_roc_auc_mean"]), 4),
            "placebo_delta_roc_auc_mean": round(float(pl["delta_roc_auc_mean"]), 4),
            "placebo_delta_roc_auc_std": round(float(pl["delta_roc_auc_std"]), 4),
            "true_hub_spearman_rho": round(float(tr["spearman_rho_mean"]), 4),
            "placebo_spearman_rho_mean": round(float(pl["spearman_rho_mean"]), 4),
            "true_hub_flip_rate": round(float(tr["flip_rate_mean"]), 4),
            "placebo_flip_rate_mean": round(float(pl["flip_rate_mean"]), 4),
            "true_worse_than_placebo": abs(tr["delta_roc_auc_mean"]) > abs(pl["delta_roc_auc_mean"]),
        })

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "placebo_vs_true_hub_comparison.csv"), index=False)
    n_confirms = out["true_worse_than_placebo"].sum()
    print(f"\nTrue hub removal is more damaging (|ΔAUC|) than placebo random removal "
          f"in {n_confirms}/{len(out)} (model, K) cells.")
    print(f"Saved -> {OUT_DIR}/placebo_vs_true_hub_comparison.csv")
    return out


def main():
    set_global_seed()
    cfg = load_experiment_config(CONFIG_PATH)
    print(f"Config: hub_removal_k={cfg['hub_removal_k']}  seeds={cfg['seeds']}")

    tabular_placebo = run_tabular_placebo(cfg)
    gnn_placebo = run_gnn_placebo(cfg)
    compare_against_true_hub_removal(tabular_placebo, gnn_placebo)


if __name__ == "__main__":
    main()
