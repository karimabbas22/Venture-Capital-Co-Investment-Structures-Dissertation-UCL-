"""
Experiments/full_raw_features_test.py -- decomposes the GNN's clean-graph
AUC advantage into "richer inputs" vs "architecture". The GNN consumes 6
raw per-snapshot investor metrics (degree, weighted_degree, betweenness,
pagerank, eigenvector_lcc, clustering_coeff) while the tabular network
model consumes exactly one hand-aggregated summary (mean_pagerank), so the
GNN's higher point-estimate AUC (not statistically significant per
Experiments/significance_tests.py, but still the natural next question)
could be "message passing is a better architecture" or simply "the GNN
sees six numbers per investor and the tabular model sees one" -- these
were previously conflated.

This builds an intermediate tabular condition: mean+max of all 6 raw
investor metrics (12 columns total) instead of just mean_pagerank, fed to
the same 5 tabular classifiers. If this closes most of the gap to the
GNN's 0.6582, the "architecture matters" claim weakens; if it doesn't,
the claim strengthens.

Two of the six metrics' mean+max aggregates don't exist in
firms_with_network_T7.parquet (only mean_weighted_degree/max_weighted_degree
and max_clustering are missing -- mean_clustering already exists but
max_clustering was never aggregated) -- computed fresh here via the same
die_fp merge pattern already used in test_literature_features.py, reading
only from investor_features_by_snapshot.parquet (already computed by
script 04, read-only here).

Standalone, isolated, does not touch Scripts/ or the production tabular
models. Run: python3 Experiments/full_raw_features_test.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import xgboost  # noqa: F401 -- import before torch-adjacent libs, see significance_tests.py
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
    engineer_baseline_features, tune_on_val, evaluate_with_tuned_threshold,
    set_global_seed,
)

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA     = os.path.join(ROOT, "Data", "processed")
OUT_DIR  = os.path.join(ROOT, "Experiments", "output")
os.makedirs(OUT_DIR, exist_ok=True)

LABEL_COL    = "exit_within_T"
DATE_COL     = "first_deal_date"
RANDOM_STATE = SEED

GNN_CLEAN_AUC = 0.6582  # from Data/processed/models/T7/gnn_results.csv, for reference only


def build_full_raw_features() -> pd.DataFrame:
    """Add the two missing mean+max aggregates (weighted_degree, max_clustering)
    to firms_with_network_T7.parquet's existing 8 (mean/max degree, betweenness,
    pagerank, eigenvector_lcc, plus mean_clustering) -- same anchor-safe merge
    pattern 04/07/test_literature_features.py already use."""
    firms = pd.read_parquet(os.path.join(DATA, "firms_with_network_T7.parquet"))
    deals = pd.read_parquet(os.path.join(DATA, "deals.parquet"))
    deals = deals[deals["investor_name"] != "Undisclosed Firm"].copy()
    deals["investor_org_id"] = deals["investor_org_id"].fillna(deals["investor_name"]).astype(str)

    fp = firms[["company_name", "first_deal_date", "snapshot_date"]]
    die_fp = deals.merge(fp, on="company_name", how="inner")
    die_fp = die_fp[die_fp["investment_date"] <= die_fp["first_deal_date"]]
    die_fp = die_fp.dropna(subset=["investor_org_id"])
    die_fp = die_fp.drop_duplicates(subset=["company_name", "investor_org_id"])

    snap = pd.read_parquet(os.path.join(DATA, "investor_features_by_snapshot.parquet"))
    merged = die_fp.merge(
        snap[["investor_org_id", "snapshot_date", "weighted_degree", "clustering_coeff"]],
        on=["investor_org_id", "snapshot_date"], how="left")

    extra = merged.groupby("company_name").agg(
        mean_weighted_degree=("weighted_degree", "mean"),
        max_weighted_degree=("weighted_degree", "max"),
        max_clustering=("clustering_coeff", "max"),
    ).reset_index()

    firms = firms.merge(extra, on="company_name", how="left")
    for col in ["mean_weighted_degree", "max_weighted_degree", "max_clustering"]:
        firms[col] = firms[col].fillna(0.0)
    return firms


FULL_RAW_NET_COLS = [
    "mean_degree", "max_degree",
    "mean_weighted_degree", "max_weighted_degree",
    "mean_betweenness", "max_betweenness",
    "mean_pagerank", "max_pagerank",
    "mean_eigenvector_lcc", "max_eigenvector_lcc",
    "mean_clustering", "max_clustering",
]


def _finalize(df: pd.DataFrame, extra_net_cols: list) -> pd.DataFrame:
    df = engineer_baseline_features(df, LABEL_COL, DATE_COL)
    df["missing_evc"] = df["mean_eigenvector_lcc"].isna().astype(int)
    for col in extra_net_cols:
        df[col] = df[col].fillna(0.0)
    return df


def train_and_eval_variant(variant_name: str, df: pd.DataFrame, extra_network_cols: list) -> list:
    num_cols = BASELINE_NUM_COLS + extra_network_cols
    bin_cols = NETWORK_BINARY_COLS
    cat_cols = CAT_FEATURE_COLS

    work = _finalize(df, extra_network_cols)
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
    print("\nFull raw investor features test (decompose GNN's advantage)")

    firms = build_full_raw_features()

    variants = {
        "control": ["mean_pagerank"],
        "full_raw": FULL_RAW_NET_COLS,
    }

    all_results = []
    for name, cols in variants.items():
        print(f"\n--- Variant: {name} ({len(cols)} network num cols) ---")
        all_results.extend(train_and_eval_variant(name, firms, cols))

    res = pd.DataFrame(all_results)
    res.to_csv(os.path.join(OUT_DIR, "full_raw_features_results.csv"), index=False)

    print("\nSummary")
    print(f"{'Model':<14s}{'control':>10s}{'full_raw':>10s}{'delta':>10s}")
    pivot = res.pivot_table(index=res["model"].str.split("_").str[0], columns="variant", values="roc_auc")
    pivot = pivot[["control", "full_raw"]]
    pivot["delta"] = pivot["full_raw"] - pivot["control"]
    for model, row in pivot.iterrows():
        print(f"{model:<14s}{row['control']:>10.4f}{row['full_raw']:>10.4f}{row['delta']:>+10.4f}")

    best_control = pivot["control"].max()
    best_full_raw = pivot["full_raw"].max()
    print(f"\nBest control (mean_pagerank only): {best_control:.4f}")
    print(f"Best full_raw (12 raw metrics):     {best_full_raw:.4f}")
    print(f"GNN (6 raw metrics, message passing): {GNN_CLEAN_AUC:.4f}")
    print(f"\nGap closed by full_raw vs control:  {best_full_raw - best_control:+.4f}")
    print(f"Remaining gap, full_raw vs GNN:      {GNN_CLEAN_AUC - best_full_raw:+.4f}")
    print(f"(For reference, control vs GNN gap was: {GNN_CLEAN_AUC - best_control:+.4f})")

    print(f"\nSaved -> {os.path.join(OUT_DIR, 'full_raw_features_results.csv')}")


if __name__ == "__main__":
    main()
