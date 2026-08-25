"""
Experiments/test_literature_features.py -- standalone, read-only test of
three literature-motivated candidate features against the frozen network
model feature set (script 05). Does NOT modify pipeline_core.py or any
numbered Scripts/ file -- everything here is additive and self-contained.
Not wired into the main pipeline or the final report (script 12).

Candidates tested (see Dissertation Literature Review.rtf):
  1. n_investors_first_round -- the startup's OWN bipartite degree (Esposito
     et al. 2022: startup centrality matters, not just investor centrality).
     Already computed by script 04, just never included in the reduced
     feature set -- no new computation needed here.
  2. mean_weighted_degree    -- tie-strength / repeat-co-investment proxy
     (Bellavitis et al. 2020: repeated co-investment between the same
     investors has nonlinear, sometimes negative, returns). Aggregated
     fresh here from investor_features_by_snapshot.parquet's existing
     weighted_degree column -- script 04 computes this per investor per
     snapshot but never aggregates it to startup level.
  3. mean_community_share    -- Louvain community size (normalized by graph
     size) of the startup's first-round investors (Bubna et al. 2020;
     Gu et al. 2022: VCs cluster into tightly-knit communities that shape
     access and performance). Computed fresh here via
     nx.algorithms.community.louvain_communities on the same per-snapshot
     graphs script 04 builds -- never persisted anywhere else.

Protocol: identical temporal split, identical 5 model classes, identical
hyperparameter grids, identical threshold-tuning as script 05 -- only the
feature SET varies per variant, so any AUC/F1 delta is attributable to the
new variable, not a protocol change. Val is the principled comparison;
test is reported for completeness only (this is an exploratory side
experiment, not a final model selection).

Run: python3 Experiments/test_literature_features.py
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Scripts")
sys.path.insert(0, SCRIPTS_DIR)
from pipeline_core import (
    SEED, TRAIN_END, VAL_END, CAT_FEATURE_COLS,
    BASELINE_NUM_COLS, NETWORK_BINARY_COLS,
    temporal_train_val_test_split, fit_preprocessor, transform_preprocessor,
    set_global_seed,
    engineer_baseline_features, tune_on_val, evaluate_with_tuned_threshold,
    build_coinvestment_snapshot,
)

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA     = os.path.join(ROOT, "Data", "processed")
OUT_DIR  = os.path.join(ROOT, "Experiments", "output")
os.makedirs(OUT_DIR, exist_ok=True)

LABEL_COL    = "exit_within_T"
DATE_COL     = "first_deal_date"
RANDOM_STATE = SEED


# ── Build the three candidate features (additive only) ───────────────────────
def build_candidate_features() -> pd.DataFrame:
    firms = pd.read_parquet(os.path.join(DATA, "firms_with_network_T7.parquet"))
    deals = pd.read_parquet(os.path.join(DATA, "deals.parquet"))
    deals = deals[deals["investor_name"] != "Undisclosed Firm"].copy()
    deals["investor_org_id"] = deals["investor_org_id"].fillna(deals["investor_name"]).astype(str)

    fp = firms[["company_name", "first_deal_date", "snapshot_date"]]
    die_fp = deals.merge(fp, on="company_name", how="inner")
    die_fp = die_fp[die_fp["investment_date"] <= die_fp["first_deal_date"]]
    die_fp = die_fp.dropna(subset=["investor_org_id"])
    die_fp = die_fp.drop_duplicates(subset=["company_name", "investor_org_id"])
    print(f"Startup-investor pairs at anchor: {len(die_fp):,}")

    # -- Candidate 2: mean_weighted_degree (tie strength) --------------------
    snap = pd.read_parquet(os.path.join(DATA, "investor_features_by_snapshot.parquet"))
    die_fp_wd = die_fp.merge(
        snap[["investor_org_id", "snapshot_date", "weighted_degree"]],
        on=["investor_org_id", "snapshot_date"], how="left")
    wd_agg = die_fp_wd.groupby("company_name")["weighted_degree"].mean().rename("mean_weighted_degree")

    # -- Candidate 3: mean_community_share (Louvain community size) ----------
    needed_snaps = sorted(firms["snapshot_date"].dropna().unique())
    print(f"\nComputing Louvain communities for {len(needed_snaps)} snapshots...")
    community_rows = []
    for i, snap_date in enumerate(needed_snaps):
        t0 = time.time()
        G = build_coinvestment_snapshot(deals, pd.Timestamp(snap_date))
        n_total = G.number_of_nodes()
        if n_total == 0:
            continue
        comms = nx.algorithms.community.louvain_communities(G, weight="weight", seed=SEED)
        for comm in comms:
            share = len(comm) / n_total
            for node in comm:
                community_rows.append({"investor_org_id": node,
                                       "snapshot_date": pd.Timestamp(snap_date),
                                       "community_share": share})
        print(f"  [{i+1:2d}/{len(needed_snaps)}] {pd.Timestamp(snap_date).date()}  "
              f"nodes={n_total:>6,}  communities={len(comms):>4,}  ({time.time()-t0:4.1f}s)")
    comm_df = pd.DataFrame(community_rows)

    die_fp_comm = die_fp.merge(comm_df, on=["investor_org_id", "snapshot_date"], how="left")
    comm_agg = die_fp_comm.groupby("company_name")["community_share"].mean().rename("mean_community_share")

    firms = firms.merge(wd_agg, on="company_name", how="left")
    firms = firms.merge(comm_agg, on="company_name", how="left")
    firms["mean_weighted_degree"] = firms["mean_weighted_degree"].fillna(0.0)
    firms["mean_community_share"] = firms["mean_community_share"].fillna(0.0)
    firms["n_investors_first_round"] = firms["n_investors_first_round"].fillna(0.0)

    return firms


def _finalize_features(df: pd.DataFrame, num_cols: list) -> pd.DataFrame:
    """Same finishing step as pipeline_core.engineer_network_features, but
    parametrized over whichever candidate num_cols this variant uses,
    instead of the frozen global NETWORK_NUM_COLS constant."""
    df = engineer_baseline_features(df, LABEL_COL, DATE_COL)
    df["missing_evc"] = df["mean_eigenvector_lcc"].isna().astype(int)
    for col in num_cols:
        df[col] = df[col].fillna(0.0)
    return df


# ── Train + evaluate one feature-set variant ──────────────────────────────────
def train_and_eval_variant(variant_name: str, df: pd.DataFrame, extra_network_cols: list) -> list:
    """extra_network_cols is the network-specific numeric columns for this
    variant (e.g. ["mean_pagerank"] for control) -- BASELINE_NUM_COLS is
    always included too, matching script 05's actual feature set exactly
    except for the network block under test."""
    num_cols = BASELINE_NUM_COLS + extra_network_cols
    bin_cols = NETWORK_BINARY_COLS
    cat_cols = CAT_FEATURE_COLS

    work = _finalize_features(df, extra_network_cols)
    train, val, test = temporal_train_val_test_split(work, TRAIN_END, VAL_END)
    y_tr, y_val, y_te = train[LABEL_COL], val[LABEL_COL], test[LABEL_COL]
    n_neg, n_pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    pre = fit_preprocessor(train, num_cols, bin_cols, cat_cols)
    X_tr, X_val, X_te = (transform_preprocessor(pre, s) for s in (train, val, test))

    results = []

    def make_lr(C):
        return LogisticRegression(C=C, class_weight="balanced", solver="lbfgs",
                                  max_iter=1000, random_state=RANDOM_STATE)
    p, _, _ = tune_on_val(make_lr, [{"C": c} for c in [0.01, 0.1, 1.0, 10.0]],
                          X_tr, y_tr, X_val, y_val)
    clf = make_lr(**p).fit(X_tr, y_tr)
    results.append(evaluate_with_tuned_threshold(f"LogReg_{variant_name}", clf,
                                                  X_val, y_val, X_te, y_te, "Test", None, verbose=False))

    def make_rf(max_depth, min_samples_leaf):
        return RandomForestClassifier(n_estimators=300, max_depth=max_depth,
                                      min_samples_leaf=min_samples_leaf,
                                      class_weight="balanced", n_jobs=-1,
                                      random_state=RANDOM_STATE)
    p, _, _ = tune_on_val(make_rf,
        [{"max_depth": d, "min_samples_leaf": m} for d in [10, 20, None] for m in [5, 10]],
        X_tr, y_tr, X_val, y_val)
    clf = make_rf(**p).fit(X_tr, y_tr)
    results.append(evaluate_with_tuned_threshold(f"RandomForest_{variant_name}", clf,
                                                  X_val, y_val, X_te, y_te, "Test", None, verbose=False))

    def make_gbm(learning_rate, max_iter, max_depth):
        return HistGradientBoostingClassifier(learning_rate=learning_rate, max_iter=max_iter,
            max_depth=max_depth, class_weight="balanced", random_state=RANDOM_STATE)
    gbm_grid = [{"learning_rate": lr, "max_iter": n, "max_depth": d}
                for lr in [0.05, 0.1] for n in [100, 200] for d in [3, 5]]
    p, _, _ = tune_on_val(make_gbm, gbm_grid, X_tr, y_tr, X_val, y_val)
    clf = make_gbm(**p).fit(X_tr, y_tr)
    results.append(evaluate_with_tuned_threshold(f"HistGBM_{variant_name}", clf,
                                                  X_val, y_val, X_te, y_te, "Test", None, verbose=False))

    if HAS_XGB:
        def make_xgb(n_estimators, max_depth, learning_rate, subsample=0.8):
            return XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate, subsample=subsample,
                colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
                eval_metric="logloss", tree_method="hist",
                random_state=RANDOM_STATE, n_jobs=-1)
        xgb_grid = [{"n_estimators": n, "max_depth": d, "learning_rate": lr, "subsample": ss}
                    for n in [200, 400] for d in [3, 5] for lr in [0.05, 0.1] for ss in [0.7, 1.0]]
        p, _, _ = tune_on_val(make_xgb, xgb_grid, X_tr, y_tr, X_val, y_val)
        clf = make_xgb(**p).fit(X_tr, y_tr)
        results.append(evaluate_with_tuned_threshold(f"XGBoost_{variant_name}", clf,
                                                      X_val, y_val, X_te, y_te, "Test", None, verbose=False))

    if HAS_LGBM:
        def make_lgbm(num_leaves, max_depth, learning_rate, n_estimators):
            return LGBMClassifier(num_leaves=num_leaves, max_depth=max_depth,
                learning_rate=learning_rate, n_estimators=n_estimators,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        lgbm_grid = [{"num_leaves": nl, "max_depth": -1, "learning_rate": lr, "n_estimators": n}
                     for nl in [15, 31] for lr in [0.05, 0.1] for n in [200, 400]]
        p, _, _ = tune_on_val(make_lgbm, lgbm_grid, X_tr, y_tr, X_val, y_val)
        clf = make_lgbm(**p).fit(X_tr, y_tr)
        results.append(evaluate_with_tuned_threshold(f"LightGBM_{variant_name}", clf,
                                                      X_val, y_val, X_te, y_te, "Test", None, verbose=False))

    for r in results:
        r["variant"] = variant_name
    return results


def main():
    set_global_seed()
    print(f"{'='*70}\n  LITERATURE-MOTIVATED FEATURE TEST (exploratory, isolated)\n{'='*70}")

    firms = build_candidate_features()

    variants = {
        "control":       ["mean_pagerank"],
        "own_degree":    ["mean_pagerank", "n_investors_first_round"],
        "tie_strength":  ["mean_pagerank", "mean_weighted_degree"],
        "community":     ["mean_pagerank", "mean_community_share"],
        "combined":      ["mean_pagerank", "n_investors_first_round",
                          "mean_weighted_degree", "mean_community_share"],
    }

    all_results = []
    for name, num_cols in variants.items():
        print(f"\n{'-'*70}\n  Variant: {name}  (network num cols = {num_cols})\n{'-'*70}")
        all_results.extend(train_and_eval_variant(name, firms, num_cols))

    res = pd.DataFrame(all_results)
    res.to_csv(os.path.join(OUT_DIR, "literature_feature_test_results.csv"), index=False)

    # ── Summary: best model per variant + delta vs control ───────────────────
    print(f"\n{'='*70}\n  SUMMARY (best model per variant, by val threshold F1 tie-broken on ROC-AUC)\n{'='*70}")
    control_best = res[res["variant"] == "control"].loc[
        res[res["variant"] == "control"]["roc_auc"].idxmax()]
    print(f"  {'Variant':<15s} {'Best model':<25s} {'ROC-AUC':>8s} {'ΔvsCtrl':>9s} {'F1':>8s} {'ΔvsCtrl':>9s}")
    for name in variants:
        sub = res[res["variant"] == name]
        best = sub.loc[sub["roc_auc"].idxmax()]
        d_roc = best["roc_auc"] - control_best["roc_auc"]
        d_f1  = best["f1"] - control_best["f1"]
        print(f"  {name:<15s} {best['model']:<25s} {best['roc_auc']:>8.4f} {d_roc:>+9.4f} "
              f"{best['f1']:>8.4f} {d_f1:>+9.4f}")

    print(f"\n  Per-model breakdown, ROC-AUC by variant:")
    pivot = res.pivot_table(index=res["model"].str.split("_").str[0], columns="variant", values="roc_auc")
    pivot = pivot[list(variants.keys())]
    print(pivot.round(4).to_string())

    print(f"\nFull results -> {os.path.join(OUT_DIR, 'literature_feature_test_results.csv')}")


if __name__ == "__main__":
    main()
