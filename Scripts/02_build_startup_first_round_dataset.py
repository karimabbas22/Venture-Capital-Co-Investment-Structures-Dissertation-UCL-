"""
02_build_startup_first_round_dataset.py — startup panel + exit label construction.

Methodology
───────────
1. Anchor date = first investment date over the FULL deal history (2000-2024).
2. Tabular features are anchor-time only: first-round size, deal value,
   syndicate size, round stage, sector. Post-anchor funding rounds are
   outcomes, not predictors.
3. exit_within_T uses censoring-aware logic from pipeline_core:
   label 1 = successful exit in (anchor, anchor+T]; label 0 only when the
   full T-year window is observable.
4. ENTRY_END = 2017-12-31 so that every retained startup's 7-year window
   closes by DATA_END_DATE (2024-12-31).

Outputs
───────
  Data/processed/firms_panel.parquet      eligible classification sample
  Data/processed/firms_panel_all.parquet  full entry-window panel incl.
                                          censored/inconsistent rows (audit)

Note — survival-framing readiness: this script already computes exit_date,
exit_type, and the anchor date for every startup, which is the same
information a time-to-event / competing-risks framing would need
(time_to_event = exit_date - anchor_date; event_observed = exit_date
notna; event_type = exit_type). No new columns are required to build a
descriptive survival extension on top of this dataset — see
ROBUSTNESS_REFRAMING_PLAN.md, Section D.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_core import (
    T_YEARS, ENTRY_START, ENTRY_END, DATA_END_DATE, SUCCESS_EXIT_TYPES,
    build_anchor_dates, compute_tabular_features_for_startup,
    build_exit_labels, mark_censored_observations,
    filter_classification_sample, set_global_seed,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "Data", "processed")


def main():
    set_global_seed()

    # ── 1. Load ──────────────────────────────────────────────────────────────
    deals = pd.read_parquet(os.path.join(DATA, "deals.parquet"))
    exits = pd.read_parquet(os.path.join(DATA, "exits.parquet"))
    print(f"Loaded deals: {len(deals):,} rows "
          f"({deals['investment_date'].min().date()} – "
          f"{deals['investment_date'].max().date()})")
    print(f"Loaded exits: {len(exits):,} rows")

    # ── 2. Anchor dates from full history ────────────────────────────────────
    anchors = build_anchor_dates(deals)
    print(f"\nUnique startups (full history): {len(anchors):,}")

    # Exclude companies whose TRUE first investment predates the deal data.
    # These companies had their Round 1 before 2010 — their first-round
    # investors are unobservable.
    n_pre_window = int(anchors["pre_window_first_round"].sum())
    print(f"Companies with true first round before deal data window: {n_pre_window:,}")
    anchors = anchors[~anchors["pre_window_first_round"]].copy()
    print(f"After excluding pre-window first rounds: {len(anchors):,}")

    # ── 3. Entry-window eligibility on the TRUE anchor ───────────────────────
    in_window = (
        (anchors["first_deal_date"] >= ENTRY_START)
        & (anchors["first_deal_date"] <= ENTRY_END)
    )
    n_pre  = int((anchors["first_deal_date"] < ENTRY_START).sum())
    n_post = int((anchors["first_deal_date"] > ENTRY_END).sum())
    firms = anchors[in_window].copy()
    print(f"Entry window {ENTRY_START.date()} – {ENTRY_END.date()}: "
          f"{len(firms):,} startups retained "
          f"({n_pre:,} pre-window, {n_post:,} post-window excluded)")

    # ── 4. Anchor-time tabular features ──────────────────────────────────────
    firms = compute_tabular_features_for_startup(firms, deals)

    # ── 5. Exit labels + censoring ────────────────────────────────────────────
    firms = build_exit_labels(firms, exits,
                              horizon_years=T_YEARS,
                              data_end_date=DATA_END_DATE,
                              success_types=SUCCESS_EXIT_TYPES)
    firms = mark_censored_observations(firms, DATA_END_DATE)

    print(f"\nExit label sources:")
    print(firms["exit_source"].value_counts().to_string())
    print(f"\nexit_within_T (entry window, n={len(firms):,}):")
    print(firms["exit_within_T"].value_counts().to_string())
    print(f"right_censored=1 : {int(firms['right_censored'].sum()):,} "
          f"(should be 0 with ENTRY_END={ENTRY_END.date()})")
    print(f"exit_inconsistent: {int(firms['exit_inconsistent'].sum()):,}")

    # ── 6. Filter to the eligible classification sample ──────────────────────
    print()
    firms_panel = filter_classification_sample(firms)

    assert (firms_panel["observation_window_end"] <= DATA_END_DATE).all(), \
        "eligible sample contains unobservable label windows"
    assert (firms_panel["right_censored"] == 0).all()

    print(f"\nexit_within_T (eligible sample, n={len(firms_panel):,}):")
    print(firms_panel["exit_within_T"].value_counts().to_string())
    print(f"\nexit_type among labelled exits:")
    print(firms_panel.loc[firms_panel["exit_within_T"] == 1, "exit_type"]
          .value_counts().to_string())

    # ── 7. Save ───────────────────────────────────────────────────────────────
    firms.to_parquet(os.path.join(DATA, "firms_panel_all.parquet"), index=False)
    firms_panel.to_parquet(os.path.join(DATA, "firms_panel.parquet"), index=False)
    print(f"\nSaved firms_panel.parquet      ({firms_panel.shape[0]:,} rows, eligible only)")
    print(f"Saved firms_panel_all.parquet  ({firms.shape[0]:,} rows, incl. censored)")
    print(f"\nCohort sizes by anchor year:")
    print(firms_panel["first_deal_date"].dt.year.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
