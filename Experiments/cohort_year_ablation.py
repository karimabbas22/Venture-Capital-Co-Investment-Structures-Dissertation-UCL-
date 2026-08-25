"""
Experiments/cohort_year_ablation.py -- tests a specific methodological
concern raised in a supervisor-style review: `cohort_year` is a raw numeric
feature, but train is anchored <=2013 while test is anchored 2016-2017 --
every test row has a cohort_year value NO tree-based model (RandomForest,
HistGBM, XGBoost, LightGBM) ever saw during training. Tree splits cannot
extrapolate beyond the maximum value seen in training, so any cohort_year
split learned on train collapses to a constant decision for every test row
-- cohort_year may be dead weight (or worse) for tree models specifically,
even though it appears in their feature-importance tables during training.
LogReg is not subject to this failure mode (a linear coefficient
extrapolates by construction), so it's included as a contrast, not because
the concern applies to it.

This trains baseline AND network variants of all 5 model classes twice --
once with cohort_year included (matches the frozen production feature set
exactly), once with it dropped -- and compares test ROC-AUC. Standalone,
read-only with respect to Scripts/ and the trained production models
(nothing here overwrites baseline_results.json/network_results.json).

Run: python3 Experiments/cohort_year_ablation.py
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
    BASELINE_NUM_COLS, NETWORK_NUM_COLS, NETWORK_BINARY_COLS,
    temporal_train_val_test_split, fit_preprocessor, transform_preprocessor,
    set_global_seed, engineer_baseline_features, engineer_network_features,
    tune_on_val, evaluate_with_tuned_threshold,
)

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA     = os.path.join(ROOT, "Data", "processed")
OUT_DIR  = os.path.join(ROOT, "Experiments", "output")
os.makedirs(OUT_DIR, exist_ok=True)

LABEL_COL    = "exit_within_T"
DATE_COL     = "first_deal_date"
RANDOM_STATE = SEED


def train_and_eval(variant_name: str, feature_set: str, df: pd.DataFrame, num_cols: list) -> list:
    bin_cols = [] if feature_set == "baseline" else NETWORK_BINARY_COLS
    cat_cols = CAT_FEATURE_COLS

    if feature_set == "baseline":
        work = engineer_baseline_features(df, LABEL_COL, DATE_COL)
    else:
        work = engineer_network_features(df, LABEL_COL, DATE_COL)

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
        r["feature_set"] = feature_set
    return results


def main():
    set_global_seed()
    print(f"{'='*70}\n  COHORT_YEAR ABLATION (tree-model temporal-extrapolation check)\n{'='*70}")

    baseline_df = pd.read_parquet(os.path.join(DATA, "firms_panel.parquet"))
    network_df = pd.read_parquet(os.path.join(DATA, "firms_with_network_T7.parquet"))

    all_results = []

    for feature_set, df in [("baseline", baseline_df), ("network", network_df)]:
        num_with = BASELINE_NUM_COLS if feature_set == "baseline" else BASELINE_NUM_COLS + NETWORK_NUM_COLS
        num_without = [c for c in num_with if c != "cohort_year"]

        print(f"\n--- {feature_set}: WITH cohort_year ---")
        all_results.extend(train_and_eval("with_cohort_year", feature_set, df, num_with))
        print(f"--- {feature_set}: WITHOUT cohort_year ---")
        all_results.extend(train_and_eval("without_cohort_year", feature_set, df, num_without))

    res = pd.DataFrame(all_results)
    res.to_csv(os.path.join(OUT_DIR, "cohort_year_ablation_results.csv"), index=False)

    print(f"\n{'='*70}\n  SUMMARY: test ROC-AUC, with vs without cohort_year\n{'='*70}")
    res["model_name"] = res["model"].str.replace(r"_(with|without)_cohort_year", "", regex=True)
    pivot = res.pivot_table(index=["feature_set", "model_name"], columns="variant", values="roc_auc")
    pivot = pivot[["with_cohort_year", "without_cohort_year"]]
    pivot["delta"] = pivot["without_cohort_year"] - pivot["with_cohort_year"]
    print(pivot.round(4).to_string())

    tree_models = ["RandomForest", "HistGBM", "XGBoost", "LightGBM"]
    tree_rows = pivot[pivot.index.get_level_values("model_name").isin(tree_models)]
    print(f"\nTree models only -- mean delta (without - with cohort_year): {tree_rows['delta'].mean():+.4f}")
    lr_rows = pivot[pivot.index.get_level_values("model_name") == "LogReg"]
    print(f"LogReg only -- mean delta (without - with cohort_year): {lr_rows['delta'].mean():+.4f}")

    print(f"\nSaved -> {os.path.join(OUT_DIR, 'cohort_year_ablation_results.csv')}")


if __name__ == "__main__":
    main()
