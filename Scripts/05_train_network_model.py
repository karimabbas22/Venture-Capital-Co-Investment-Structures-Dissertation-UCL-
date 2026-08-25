"""
05_train_network_model.py — network-augmented exit prediction model.

Adds co-investment network features (from script 04) on top of the baseline
firm features (from script 03). Measures the lift that network information
provides over the baseline.

Outputs → Data/processed/models/T{T_YEARS}/
    network_results.csv / .json
    network_*.pkl
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import joblib

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

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_core import (
    SEED, T_YEARS, TRAIN_END, VAL_END,
    temporal_train_val_test_split, fit_preprocessor, transform_preprocessor,
    preprocessor_feature_names, set_global_seed,
    engineer_network_features, tune_on_val, evaluate_with_tuned_threshold,
    BASELINE_NUM_COLS as FIRM_NUM,
    NETWORK_NUM_COLS as NET_NUM,
    NETWORK_BINARY_COLS as BINARY,
    CAT_FEATURE_COLS as CAT_COLS,
)

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA     = os.path.join(ROOT, "Data", "processed")
HORIZON  = T_YEARS
IN_FILE  = os.path.join(DATA, "firms_with_network_T7.parquet")
OUT_DIR  = os.path.join(DATA, "models", f"T{HORIZON}")
os.makedirs(OUT_DIR, exist_ok=True)

LABEL_COL    = "exit_within_T"
DATE_COL     = "first_deal_date"
RANDOM_STATE = SEED

# Feature groups.
# Baseline block matches the reduced set in script 03 (same rows, same split,
# strictly comparable) via pipeline_core's shared constants. Network block is
# deliberately parsimonious: PageRank is retained as the single preferred
# centrality measure (degree, betweenness, and eigenvector centrality are
# collinear with it at r=0.72-0.95 and were dropped), plus one explicit
# missing-network-coverage indicator. A validation-only sweep confirmed
# mean_pagerank performs within tolerance of the syndicate-size-based
# alternative (n_known_investors) across all 5 model classes, so PageRank is
# preferred here as the more theoretically grounded choice rather than on
# marginal AUC alone.
ALL_NUM = FIRM_NUM + NET_NUM


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    return engineer_network_features(df, LABEL_COL, DATE_COL)


def main():
    set_global_seed()

    df = pd.read_parquet(IN_FILE)
    print(f"\nNETWORK MODEL (T={HORIZON}y)")
    print(f"Shape: {df.shape}")

    df = engineer_features(df)

    train, val, test = temporal_train_val_test_split(df, TRAIN_END, VAL_END)
    print(f"\nTemporal split:")
    for name, s in [("Train", train), ("Val", val), ("Test", test)]:
        pos = s[LABEL_COL].sum()
        print(f"  {name:<6} n={len(s):>5,} | exit=1: {pos:,} ({100*pos/len(s):.1f}%)")

    y_tr, y_val, y_te = train[LABEL_COL], val[LABEL_COL], test[LABEL_COL]
    n_neg, n_pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    pre = fit_preprocessor(train, ALL_NUM, BINARY, CAT_COLS)
    X_tr  = transform_preprocessor(pre, train)
    X_val = transform_preprocessor(pre, val)
    X_te  = transform_preprocessor(pre, test)
    feat_names = preprocessor_feature_names(pre)
    print(f"\nFeature matrix: {X_tr.shape[1]} cols (firm={len(FIRM_NUM)} + net={len(NET_NUM)} + binary + cat)")

    results = []
    saved = {}

    # Load baseline for comparison
    baseline_path = os.path.join(OUT_DIR, "baseline_results.json")
    baseline = []
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)

    # LogReg
    def make_lr(C):
        return LogisticRegression(C=C, class_weight="balanced",
                                  solver="lbfgs", max_iter=1000,
                                  random_state=RANDOM_STATE)
    p, _, _ = tune_on_val(make_lr, [{"C": c} for c in [0.01, 0.1, 1.0, 10.0]],
                          X_tr, y_tr, X_val, y_val)
    clf = make_lr(**p).fit(X_tr, y_tr)
    results.append(evaluate_with_tuned_threshold("LogReg_Network", clf, X_val, y_val, X_te, y_te, "Test", OUT_DIR, save_calibration_csv=True))
    saved["logreg_network"] = {"preprocessor": pre, "model": clf}

    # RandomForest
    def make_rf(max_depth, min_samples_leaf):
        return RandomForestClassifier(n_estimators=300, max_depth=max_depth,
                                      min_samples_leaf=min_samples_leaf,
                                      class_weight="balanced", n_jobs=-1,
                                      random_state=RANDOM_STATE)
    p, _, _ = tune_on_val(make_rf,
        [{"max_depth": d, "min_samples_leaf": m} for d in [10, 20, None] for m in [5, 10]],
        X_tr, y_tr, X_val, y_val)
    clf = make_rf(**p).fit(X_tr, y_tr)
    results.append(evaluate_with_tuned_threshold("RandomForest_Network", clf, X_val, y_val, X_te, y_te, "Test", OUT_DIR, save_calibration_csv=True))
    saved["rf_network"] = {"preprocessor": pre, "model": clf}

    # HistGBM
    def make_gbm(learning_rate, max_iter, max_depth):
        return HistGradientBoostingClassifier(learning_rate=learning_rate,
            max_iter=max_iter, max_depth=max_depth, class_weight="balanced",
            random_state=RANDOM_STATE)
    gbm_grid = [{"learning_rate": lr, "max_iter": n, "max_depth": d}
                for lr in [0.05, 0.1] for n in [100, 200] for d in [3, 5]]
    p, _, _ = tune_on_val(make_gbm, gbm_grid, X_tr, y_tr, X_val, y_val)
    clf = make_gbm(**p).fit(X_tr, y_tr)
    results.append(evaluate_with_tuned_threshold("HistGBM_Network", clf, X_val, y_val, X_te, y_te, "Test", OUT_DIR, save_calibration_csv=True))
    saved["gbm_network"] = {"preprocessor": pre, "model": clf}

    # XGBoost
    if HAS_XGB:
        def make_xgb(n_estimators, max_depth, learning_rate, subsample=0.8):
            return XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate, subsample=subsample,
                colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
                eval_metric="logloss", tree_method="hist",
                random_state=RANDOM_STATE, n_jobs=-1)
        xgb_grid = [{"n_estimators": n, "max_depth": d, "learning_rate": lr, "subsample": ss}
                    for n in [200, 400] for d in [3, 5]
                    for lr in [0.05, 0.1] for ss in [0.7, 1.0]]
        p, _, _ = tune_on_val(make_xgb, xgb_grid, X_tr, y_tr, X_val, y_val)
        clf = make_xgb(**p).fit(X_tr, y_tr)
        results.append(evaluate_with_tuned_threshold("XGBoost_Network", clf, X_val, y_val, X_te, y_te, "Test", OUT_DIR, save_calibration_csv=True))

        # Print feature importance for the best tree model
        fi = pd.Series(clf.feature_importances_, index=feat_names).sort_values(ascending=False)
        print(f"\n  [XGBoost_Network] Top 20 features:")
        print(fi.head(20).round(5).to_string())
        saved["xgb_network"] = {"preprocessor": pre, "model": clf}

    # LightGBM
    if HAS_LGBM:
        def make_lgbm(num_leaves, max_depth, learning_rate, n_estimators):
            return LGBMClassifier(num_leaves=num_leaves, max_depth=max_depth,
                learning_rate=learning_rate, n_estimators=n_estimators,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                class_weight="balanced", random_state=RANDOM_STATE,
                n_jobs=-1, verbose=-1)
        lgbm_grid = [{"num_leaves": nl, "max_depth": -1, "learning_rate": lr, "n_estimators": n}
                     for nl in [15, 31] for lr in [0.05, 0.1] for n in [200, 400]]
        p, _, _ = tune_on_val(make_lgbm, lgbm_grid, X_tr, y_tr, X_val, y_val)
        clf = make_lgbm(**p).fit(X_tr, y_tr)
        results.append(evaluate_with_tuned_threshold("LightGBM_Network", clf, X_val, y_val, X_te, y_te, "Test", OUT_DIR, save_calibration_csv=True))
        saved["lgbm_network"] = {"preprocessor": pre, "model": clf}

    # Save
    for name, obj in saved.items():
        joblib.dump(obj, os.path.join(OUT_DIR, f"{name}.pkl"))

    pd.DataFrame(results).to_csv(os.path.join(OUT_DIR, "network_results.csv"), index=False)
    with open(os.path.join(OUT_DIR, "network_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Network lift comparison
    print("\nNETWORK LIFT SUMMARY")
    net_by_name = {r["model"].replace("_Network", ""): r for r in results}
    for br in baseline:
        base_name = br["model"].replace("_Baseline", "")
        nr = net_by_name.get(base_name)
        if nr:
            d_roc = nr["roc_auc"] - br["roc_auc"]
            d_pr  = nr["pr_auc"] - br["pr_auc"]
            print(f"  {base_name:<15s}  Baseline={br['roc_auc']:.4f}  "
                  f"Network={nr['roc_auc']:.4f}  Δ={d_roc:+.4f} ROC-AUC  "
                  f"Δ={d_pr:+.4f} PR-AUC")

    best_net = max(results, key=lambda x: x["roc_auc"])
    best_base = max(baseline, key=lambda x: x["roc_auc"]) if baseline else None
    if best_base:
        d = best_net["roc_auc"] - best_base["roc_auc"]
        print(f"\n  Best network: {best_net['model']} ({best_net['roc_auc']:.4f})")
        print(f"  Best baseline: {best_base['model']} ({best_base['roc_auc']:.4f})")
        print(f"  Overall lift: {d:+.4f} ROC-AUC")

    print(f"\nAll outputs → {OUT_DIR}")


if __name__ == "__main__":
    main()
