"""
03_train_baseline_model.py — baseline model using ONLY non-network features.

This script establishes predictive power BEFORE adding the co-investment
graph (network model in script 05).

Features (all observable at the startup's first VC round) — the reduced,
literature-informed set that survived forward/backward selection (single
source of truth: pipeline_core.BASELINE_NUM_COLS/BASELINE_BINARY_COLS/
CAT_FEATURE_COLS, reused here and by 05/07/graph_core.py/14). Raw round
size, deal value, round number, syndicate size, round-stage/state
categoricals, and the size-missing flag were all found redundant or dead
during feature reduction and dropped.
─────────────────────────────────────────────────────────
  Numeric:
    - log_first_round_raised      (log1p of round equity at first round)
    - cohort_year                 (year of first deal, as numeric)
    - firm_age_at_first_round     (years from founding to first round)

  Binary: none

  Categorical:
    - sector                      (TRBC economic sector)

Models trained: LogReg, RandomForest, HistGBM, XGBoost, LightGBM.
All tuned on validation ROC-AUC. Test set scored once.

Outputs → Data/processed/models/T{T_YEARS}/
    baseline_results.csv / .json
    split_train/val/test.parquet
    baseline_*.pkl
    calibration_*.png / .csv
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
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
    BASELINE_NUM_COLS, BASELINE_BINARY_COLS, CAT_FEATURE_COLS,
    temporal_train_val_test_split, fit_preprocessor, transform_preprocessor,
    preprocessor_feature_names, set_global_seed,
    engineer_baseline_features, tune_on_val, evaluate_with_tuned_threshold,
)

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA     = os.path.join(ROOT, "Data", "processed")
HORIZON  = T_YEARS
IN_FILE  = os.path.join(DATA, "firms_panel.parquet")
OUT_DIR  = os.path.join(DATA, "models", f"T{HORIZON}")
os.makedirs(OUT_DIR, exist_ok=True)

LABEL_COL    = "exit_within_T"
DATE_COL     = "first_deal_date"
RANDOM_STATE = SEED

# ── Feature groups (NO network features — baseline only) ─────────────────────
# The reduced, parsimonious set that survived forward/backward selection
# (see pipeline_core.py BASELINE_NUM_COLS/CAT_FEATURE_COLS — single source
# of truth, reused by graph_core.py too).
FIRM_NUM = BASELINE_NUM_COLS
BINARY   = BASELINE_BINARY_COLS
CAT_COLS = CAT_FEATURE_COLS

FORBIDDEN_FEATURES = {
    "exit_date", "observation_window_end", "exit_type",
    "mean_degree", "max_degree", "mean_pagerank", "max_pagerank",
}


def load_and_inspect(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"{'='*60}\n  BASELINE DATASET  (T={HORIZON}y)\n{'='*60}")
    print(f"Shape      : {df.shape}")
    print(f"Date range : {df[DATE_COL].min().date()} → {df[DATE_COL].max().date()}")
    vc = df[LABEL_COL].value_counts()
    print(f"Label      : exit=1 {vc.get(1,0):,} ({100*vc.get(1,0)/len(df):.1f}%)  |  "
          f"exit=0 {vc.get(0,0):,} ({100*vc.get(0,0)/len(df):.1f}%)")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = engineer_baseline_features(df, LABEL_COL, DATE_COL)
    leaked = FORBIDDEN_FEATURES & set(FIRM_NUM + BINARY + CAT_COLS)
    assert not leaked, f"forbidden features in model: {leaked}"
    return df


def print_feature_importance(tag: str, model, feat_names: list, top_n: int = 20):
    if hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        return
    fi = pd.Series(importances, index=feat_names).sort_values(ascending=False)
    print(f"\n  [{tag}] Top {min(top_n, len(fi))} features:")
    print(fi.head(top_n).round(5).to_string())


def main():
    set_global_seed()

    df = load_and_inspect(IN_FILE)
    df = engineer_features(df)

    train, val, test = temporal_train_val_test_split(df, TRAIN_END, VAL_END)
    print(f"\nTemporal split (anchor date):")
    for name, s in [("Train", train), ("Val", val), ("Test", test)]:
        pos = s[LABEL_COL].sum()
        print(f"  {name:<6} n={len(s):>5,} | "
              f"{s[DATE_COL].min().date()} – {s[DATE_COL].max().date()} | "
              f"exit=1: {pos:,} ({100*pos/len(s):.1f}%)")

    y_tr, y_val, y_te = train[LABEL_COL], val[LABEL_COL], test[LABEL_COL]

    n_neg, n_pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"\nClass balance (train) — neg={n_neg}, pos={n_pos}, "
          f"scale_pos_weight={scale_pos_weight:.3f}")

    for name, s in [("train", train), ("val", val), ("test", test)]:
        s.to_parquet(os.path.join(OUT_DIR, f"split_{name}.parquet"), index=False)

    pre = fit_preprocessor(train, FIRM_NUM, BINARY, CAT_COLS)
    X_tr  = transform_preprocessor(pre, train)
    X_val = transform_preprocessor(pre, val)
    X_te  = transform_preprocessor(pre, test)
    feat_names = preprocessor_feature_names(pre)
    print(f"\nFeature matrix: {X_tr.shape[1]} cols")

    results = []
    saved = {}

    # ── Logistic Regression ───────────────────────────────────────────────────
    print(f"\n{'='*60}\n  LOGISTIC REGRESSION (Baseline)\n{'='*60}")

    def make_lr(C):
        return LogisticRegression(C=C, class_weight="balanced",
                                  solver="lbfgs", max_iter=1000,
                                  random_state=RANDOM_STATE)

    lr_params, lr_val_roc, _ = tune_on_val(
        make_lr, [{"C": c} for c in [0.01, 0.1, 1.0, 10.0]],
        X_tr, y_tr, X_val, y_val)
    print(f"  Best C={lr_params['C']}  val ROC-AUC={lr_val_roc:.4f}")

    lr_clf = make_lr(**lr_params).fit(X_tr, y_tr)
    results.append(evaluate_with_tuned_threshold("LogReg_Baseline", lr_clf, X_val, y_val, X_te, y_te, "Test", OUT_DIR, save_calibration_csv=True))
    print_feature_importance("LogReg_Baseline", lr_clf, feat_names)
    saved["logreg_baseline"] = {"preprocessor": pre, "model": lr_clf}

    # ── Random Forest ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}\n  RANDOM FOREST (Baseline)\n{'='*60}")

    def make_rf(max_depth, min_samples_leaf):
        return RandomForestClassifier(
            n_estimators=300, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)

    rf_params, rf_val_roc, _ = tune_on_val(
        make_rf,
        [{"max_depth": d, "min_samples_leaf": m} for d in [10, 20, None] for m in [5, 10]],
        X_tr, y_tr, X_val, y_val)
    print(f"  Best params={rf_params}  val ROC-AUC={rf_val_roc:.4f}")

    rf_clf = make_rf(**rf_params).fit(X_tr, y_tr)
    results.append(evaluate_with_tuned_threshold("RandomForest_Baseline", rf_clf, X_val, y_val, X_te, y_te, "Test", OUT_DIR, save_calibration_csv=True))
    print_feature_importance("RandomForest_Baseline", rf_clf, feat_names)
    saved["rf_baseline"] = {"preprocessor": pre, "model": rf_clf}

    # ── Hist Gradient Boosting ────────────────────────────────────────────────
    print(f"\n{'='*60}\n  HIST GRADIENT BOOSTING (Baseline)\n{'='*60}")

    def make_gbm(learning_rate, max_iter, max_depth):
        return HistGradientBoostingClassifier(
            learning_rate=learning_rate, max_iter=max_iter,
            max_depth=max_depth, class_weight="balanced",
            random_state=RANDOM_STATE)

    gbm_grid = [{"learning_rate": lr, "max_iter": n, "max_depth": d}
                for lr in [0.05, 0.1] for n in [100, 200] for d in [3, 5]]

    gbm_params, gbm_val_roc, _ = tune_on_val(
        make_gbm, gbm_grid, X_tr, y_tr, X_val, y_val)
    print(f"  Best params={gbm_params}  val ROC-AUC={gbm_val_roc:.4f}")

    gbm_clf = make_gbm(**gbm_params).fit(X_tr, y_tr)
    results.append(evaluate_with_tuned_threshold("HistGBM_Baseline", gbm_clf, X_val, y_val, X_te, y_te, "Test", OUT_DIR, save_calibration_csv=True))
    saved["gbm_baseline"] = {"preprocessor": pre, "model": gbm_clf}

    # ── XGBoost ───────────────────────────────────────────────────────────────
    if HAS_XGB:
        print(f"\n{'='*60}\n  XGBOOST (Baseline)\n{'='*60}")

        def make_xgb(n_estimators, max_depth, learning_rate, subsample=0.8):
            return XGBClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate, subsample=subsample,
                colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
                eval_metric="logloss", tree_method="hist",
                random_state=RANDOM_STATE, n_jobs=-1)

        xgb_grid = [{"n_estimators": n, "max_depth": d, "learning_rate": lr, "subsample": ss}
                    for n in [200, 400] for d in [3, 5]
                    for lr in [0.05, 0.1] for ss in [0.7, 1.0]]

        xgb_params, xgb_val_roc, _ = tune_on_val(
            make_xgb, xgb_grid, X_tr, y_tr, X_val, y_val)
        print(f"  Best params={xgb_params}  val ROC-AUC={xgb_val_roc:.4f}")

        xgb_clf = make_xgb(**xgb_params).fit(X_tr, y_tr)
        results.append(evaluate_with_tuned_threshold("XGBoost_Baseline", xgb_clf, X_val, y_val, X_te, y_te, "Test", OUT_DIR, save_calibration_csv=True))
        print_feature_importance("XGBoost_Baseline", xgb_clf, feat_names)
        saved["xgb_baseline"] = {"preprocessor": pre, "model": xgb_clf}

    # ── LightGBM ──────────────────────────────────────────────────────────────
    if HAS_LGBM:
        print(f"\n{'='*60}\n  LIGHTGBM (Baseline)\n{'='*60}")

        def make_lgbm(num_leaves, max_depth, learning_rate, n_estimators):
            return LGBMClassifier(
                num_leaves=num_leaves, max_depth=max_depth,
                learning_rate=learning_rate, n_estimators=n_estimators,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)

        lgbm_grid = [{"num_leaves": nl, "max_depth": -1, "learning_rate": lr, "n_estimators": n}
                     for nl in [15, 31] for lr in [0.05, 0.1] for n in [200, 400]]

        lgbm_params, lgbm_val_roc, _ = tune_on_val(
            make_lgbm, lgbm_grid, X_tr, y_tr, X_val, y_val)
        print(f"  Best params={lgbm_params}  val ROC-AUC={lgbm_val_roc:.4f}")

        lgbm_clf = make_lgbm(**lgbm_params).fit(X_tr, y_tr)
        results.append(evaluate_with_tuned_threshold("LightGBM_Baseline", lgbm_clf, X_val, y_val, X_te, y_te, "Test", OUT_DIR, save_calibration_csv=True))
        print_feature_importance("LightGBM_Baseline", lgbm_clf, feat_names)
        saved["lgbm_baseline"] = {"preprocessor": pre, "model": lgbm_clf}

    # ── Save artefacts ────────────────────────────────────────────────────────
    for name, obj in saved.items():
        joblib.dump(obj, os.path.join(OUT_DIR, f"{name}.pkl"))

    pd.DataFrame(results).to_csv(os.path.join(OUT_DIR, "baseline_results.csv"), index=False)
    with open(os.path.join(OUT_DIR, "baseline_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    best = max(results, key=lambda x: x["roc_auc"])
    print(f"\n{'='*60}\n  BASELINE SUMMARY  (T={HORIZON}y)\n{'='*60}")
    print(f"  Best model : {best['model']} with ROC-AUC={best['roc_auc']:.4f}")
    print(f"\n  All results:")
    for r in sorted(results, key=lambda x: -x["roc_auc"]):
        print(f"    {r['model']:<25s}  ROC-AUC={r['roc_auc']:.4f}  "
              f"PR-AUC={r['pr_auc']:.4f}  Brier={r['brier']:.4f}")
    print(f"\nAll outputs → {OUT_DIR}")


if __name__ == "__main__":
    main()
