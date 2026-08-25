"""
pipeline_core.py — shared configuration and leakage-safe building blocks for
the VC co-investment exit-prediction pipeline.

Data source: Refinitiv (Yearly Company Deal Level Data, Exit Data, Company Datas)

Every temporal rule of the methodology lives here so that all scripts agree:

  * Anchor date        : a startup's FIRST investment date, computed over the
                         FULL deal history, never over a date-filtered subset.
  * Label              : exit_within_T = 1 iff a successful exit occurs in
                         (anchor, anchor + T years] AND on or before
                         DATA_END_DATE. A 0 label is only assigned when the
                         full T-year window is observable (window end <=
                         DATA_END_DATE). Otherwise the row is right-censored
                         and excluded from binary classification.
  * Graph snapshots    : co-investment graphs are built from deals with
                         Investment Date <= snapshot date only. Each startup is
                         assigned the latest snapshot <= its anchor date.
  * Temporal split     : train/val/test are disjoint, chronologically ordered
                         blocks of startups by anchor date.
  * Preprocessing      : fit_preprocessor() must only ever be called on the
                         training split; transform_preprocessor() applies the
                         frozen transformer to val/test.
"""

from __future__ import annotations

import random
from itertools import combinations

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, fbeta_score, brier_score_loss, confusion_matrix,
)
from sklearn.calibration import calibration_curve

# ════════════════════════════════════════════════════════════════════════════
#  Refinitiv column name mappings
# ════════════════════════════════════════════════════════════════════════════
# These map Refinitiv's verbose column names to short internal names used
# throughout the pipeline.

# -- Yearly Company Deal Level Data --
DEAL_COL_MAP = {
    "Investee Company Name":                                   "company_name",
    "Investee Company Nation":                                 "company_nation",
    "Investee Company TRBC Economic Sector":                   "sector",
    "Investment Date":                                         "investment_date",
    "Round Number":                                            "round_number",
    "Unnamed: 5":                                              "round_stage",
    "Firm Investor Name":                                      "investor_name",
    "Firm Investor Organization Id":                           "investor_org_id",
    "Fund Investor Name":                                      "fund_name",
    "Investor Equity Total\n(USD, Millions)":                  "investor_equity_mn",
    "Disclosed Fund Equity Contribution\n(USD, Millions)":     "fund_equity_mn",
    "Round Equity Total\n(USD, Millions)":                     "round_equity_mn",
    "Disclosed Debt Contribution\n(USD, Millions)":            "debt_contribution_mn",
    "Deal Value\n(USD, Millions)":                             "deal_value_mn",
    "Deal Rank Value\n(USD, Millions)":                        "deal_rank_value_mn",
    "Venture Capital Deals\n('|')":                            "is_vc_deal",
    "Investee Company Status\n('|')":                          "company_status",
    "Investee Company State/Region\n('|')":                    "company_state",
    "Investee Primary NAICS 2022\n('|')":                      "naics_2022",
    "Firm Investors First Investment Date\n('|')":             "investor_first_date",
    "Investee Company Total Funding Received To Date\n(USD, '|')": "total_funding_usd",
    "Investee Company First Investment Received Date\n('|')":  "company_first_inv_date",
    "Investee Company Last Investment Received Date\n('|')":   "company_last_inv_date",
    "Investee Company Number Of Investments Received To Date\n('|')": "n_investments_to_date",
    "Investee Company Number Of Investor Firms To Date\n('|')":"n_investor_firms_to_date",
    "Investee Company Founded Date\n('|')":                    "company_founded_date",
    "Investee Company All Investor Firms\n('|')":              "all_investor_firms",
    "Investee Company Current Operating Stage\n('|')":         "operating_stage",
    "Investee Company Current Investor Firms\n('|')":          "current_investor_firms",
    "Investee Company Historical Investor Firms\n('|')":       "historical_investor_firms",
    "Fund Investors Portfolio Company Participation Rounds\n('|')": "fund_participation_rounds",
    "Investee Company IPO Date\n('|')":                        "ipo_date",
    "Investee Portfolio Status\n('|')":                        "portfolio_status",
    "Investee Company PermID\n('|')":                          "company_permid",
    "Investee Company Current Public Status\n('|')":           "public_status",
    "New Or Follow on Investment Indicator\n('|')":            "new_or_followon",
    "Investee Primary SIC\n('|')":                             "primary_sic",
    "Pure Venture Capital Deals\n('|')":                       "is_pure_vc_deal",
    "Investee Company Number Of Employees Most Recent Year End\n('|')": "n_employees",
}

# -- Exit Data --
EXIT_COL_MAP = {
    "SDC Deal Number":                        "exit_deal_number",
    "Company Name":                           "company_name",
    "Exit Type":                              "exit_type",
    "Date Announced / Filed":                 "exit_date_announced",
    "Date Completed / Issued":                "exit_date_completed_raw",
    "Exit Date Completed / Issued":           "exit_date",
    "Ticker":                                 "ticker",
    "Proceeds Amt + Overallot Sold All Markets\n(USD)": "proceeds_usd",
    "Acquiror Name":                          "acquiror_name",
    "Time to Exit (Years)":                   "time_to_exit_years",
    "No. of Deals":                           "n_exit_deals",
    "No. of IPOs":                            "n_ipos",
    "Exit Portfolio Company TRBC Economic Sector": "exit_sector",
    "Exit Portfolio Company TRBC Industry Group":  "exit_industry_group",
    "Exit Portfolio Company TRBC Industry":        "exit_industry",
    "Exit Portfolio Company Nation\n('|')":         "exit_nation",
}

# ════════════════════════════════════════════════════════════════════════════
#  Central configuration — single source of truth for all scripts
# ════════════════════════════════════════════════════════════════════════════

SEED = 42

# T=7yr fixed-horizon binary classification is a BENCHMARK operationalization
# of the underlying time-to-event outcome (exit timing/type), chosen for
# comparability with prior graph-ML exit-prediction literature — not the
# conceptually canonical framing. A survival/competing-risks framing would
# use the full entry-cohort population without the ENTRY_END restriction
# below; see ROBUSTNESS_REFRAMING_PLAN.md for that extension's scope.
T_YEARS = 7

# Entry window for the startup panel. ENTRY_END is chosen so that EVERY
# eligible startup's T-year outcome window closes on or before DATA_END_DATE.
# With data covering 2000-2024, ENTRY_END = 2017 ensures 2017 + 7 = 2024.
# This restricts the classification sample to the OBSERVABILITY-ELIGIBLE
# subsample (11,791 of 51,464 total companies, ~22.9%) — companies entering
# 2018-2024 (the majority of the universe) are excluded solely because their
# 7yr window hasn't closed yet, not for any data-quality reason. ENTRY_START
# is deliberately kept at 2010 (not pushed back to match the data's full
# 2000+ range) to preserve a single coherent vintage for the classification
# task; the co-investment graph's HISTORY still draws on the full 2000+
# range via build_coinvestment_snapshot (see SNAPSHOT_START below) -- see
# ROBUSTNESS_REFRAMING_PLAN.md, "2000-2024 Data Extension" section.
ENTRY_START   = pd.Timestamp("2010-01-01")
ENTRY_END     = pd.Timestamp("2017-12-31")
DATA_END_DATE = pd.Timestamp("2024-12-31")

# Chronological split boundaries on ANCHOR (first-deal) date.
#   train : anchor <= TRAIN_END                  (2010-2013 cohorts)
#   val   : TRAIN_END < anchor <= VAL_END        (2014-2015 cohorts)
#   test  : VAL_END   < anchor <= ENTRY_END      (2016-2017 cohorts)
TRAIN_END = pd.Timestamp("2013-12-31")
VAL_END   = pd.Timestamp("2015-12-31")

# Exit types from the exit file that count as successful (label = 1).
SUCCESS_EXIT_TYPES = ("IPO", "Merger", "Secondary Sales")

# Company Status values from the deal data that indicate successful exit.
SUCCESS_STATUS = {"Acquisition", "Pending Acquisition", "Went Public",
                  "Merger", "LBO", "In Registration"}

# Company Status values indicating failure/shutdown (mirrors SUCCESS_STATUS).
# Used only by the survival/competing-risks panel (survival_core.py) — the
# fixed-horizon T=7 classification task above only distinguishes exit vs.
# no-exit, not failure vs. ongoing.
FAILURE_STATUS = {"Defunct", "Bankruptcy - Chapter 11", "Bankruptcy - Chapter 7"}

# Entry floor for the survival panel only. NOT the same as ENTRY_START above,
# which stays fixed at 2010-01-01 for the T=7 classification benchmark's
# observability-eligible subsample. The survival panel uses the full
# historical range the 2000-2024 data now supports, since duration models
# handle variable follow-up length via right-censoring natively.
SURV_ENTRY_START = pd.Timestamp("2000-01-01")

# Snapshot grid for as-of-date graph construction.
SNAPSHOT_FREQ  = "QE"
SNAPSHOT_START = pd.Timestamp("1999-12-31")
SNAPSHOT_END   = ENTRY_END

BETWEENNESS_K = 200
PAGERANK_TOP_DECILE = 0.90

# Temporal extension periods for network comparison (script 06)
PERIOD_1 = (pd.Timestamp("2010-01-01"), pd.Timestamp("2014-12-31"))
PERIOD_2 = (pd.Timestamp("2015-01-01"), pd.Timestamp("2020-12-31"))


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ════════════════════════════════════════════════════════════════════════════
#  Data loading helpers
# ════════════════════════════════════════════════════════════════════════════

def load_and_rename_deals(path: str) -> pd.DataFrame:
    """Load the Yearly Company Deal Level Data and apply column renaming."""
    df = pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path)
    rename = {k: v for k, v in DEAL_COL_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)
    df["investment_date"] = pd.to_datetime(df["investment_date"], errors="coerce")
    for col in ["company_first_inv_date", "company_last_inv_date",
                "company_founded_date", "ipo_date", "investor_first_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_and_rename_exits(path: str) -> pd.DataFrame:
    """Load the Exit Data and apply column renaming."""
    df = pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path)
    rename = {k: v for k, v in EXIT_COL_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)
    df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
    return df


# ════════════════════════════════════════════════════════════════════════════
#  Anchor dates and anchor-time tabular features
# ════════════════════════════════════════════════════════════════════════════

def build_anchor_dates(df_deals: pd.DataFrame) -> pd.DataFrame:
    """
    One row per startup with its anchor (first investment) date.

    Uses `company_first_inv_date` (Refinitiv's "Investee Company First
    Investment Received Date") as the TRUE anchor. This field covers the
    company's full history including pre-2010 rounds that are outside the
    deal data window. Companies whose true first investment predates
    ENTRY_START are excluded — their Round 1 investors are unobservable.

    Falls back to min(investment_date) from the deal data only if the
    Refinitiv field is missing.
    """
    d = df_deals.dropna(subset=["investment_date"]).copy()

    if "company_first_inv_date" in d.columns:
        d["company_first_inv_date"] = pd.to_datetime(
            d["company_first_inv_date"], errors="coerce")

    anchors = (
        d.sort_values("investment_date", kind="mergesort")
        .groupby("company_name", sort=False)
        .agg(
            first_deal_date_data = ("investment_date", "min"),
            first_deal_date_ref  = ("company_first_inv_date", "first"),
            sector               = ("sector", "first"),
            company_state        = ("company_state", "first"),
            company_permid       = ("company_permid", "first"),
            company_status       = ("company_status", "first"),
            ipo_date             = ("ipo_date", "first"),
            company_founded_date = ("company_founded_date", "first"),
        )
        .reset_index()
    )

    # Use the Refinitiv first-investment date as the true anchor.
    # If a company's true first investment predates the deal data window,
    # this correctly identifies it for exclusion.
    anchors["first_deal_date"] = anchors["first_deal_date_ref"].fillna(
        anchors["first_deal_date_data"])
    anchors["first_deal_date"] = pd.to_datetime(anchors["first_deal_date"])
    anchors["pre_window_first_round"] = (
        anchors["first_deal_date"] < anchors["first_deal_date_data"]
    )

    return anchors


def compute_tabular_features_for_startup(anchors: pd.DataFrame,
                                          df_deals: pd.DataFrame) -> pd.DataFrame:
    """
    Anchor-time tabular features, vectorised over all startups.
    Only deal records with investment_date <= the startup's anchor date are used.
    """
    d = df_deals.dropna(subset=["investment_date"]).copy()
    d = d.merge(anchors[["company_name", "first_deal_date"]],
                on="company_name", how="inner")
    d = d[d["investment_date"] <= d["first_deal_date"]]

    d = d.sort_values(["investment_date"], kind="mergesort")

    feats = (
        d.groupby("company_name", sort=False)
        .agg(
            n_first_round_deals   = ("investor_name", "count"),
            first_round_raised_mn = ("round_equity_mn", "first"),
            first_round_deal_value_mn = ("deal_value_mn", "first"),
            first_round_size_missing = ("round_equity_mn",
                                        lambda s: float(s.isna().all())),
            first_round_stage     = ("round_stage", "first"),
            first_round_number    = ("round_number", "first"),
            syndicate_size        = ("investor_name", "nunique"),
        )
        .reset_index()
    )
    feats["first_round_raised_mn"] = feats["first_round_raised_mn"].fillna(0.0)
    feats["first_round_deal_value_mn"] = feats["first_round_deal_value_mn"].fillna(0.0)

    result = anchors.merge(feats, on="company_name", how="left")

    # Firm age at first round (years)
    if "company_founded_date" in result.columns:
        result["company_founded_date"] = pd.to_datetime(
            result["company_founded_date"], errors="coerce")
        result["firm_age_at_first_round"] = (
            (result["first_deal_date"] - result["company_founded_date"]).dt.days
            / 365.25
        )
    return result


# ════════════════════════════════════════════════════════════════════════════
#  Exit labels with explicit right-censoring
# ════════════════════════════════════════════════════════════════════════════

def build_exit_labels(df_firms: pd.DataFrame,
                      df_exits: pd.DataFrame,
                      horizon_years: int = T_YEARS,
                      data_end_date: pd.Timestamp = DATA_END_DATE,
                      success_types: tuple = SUCCESS_EXIT_TYPES) -> pd.DataFrame:
    """
    Build exit labels from TWO sources:
      1. Exit file (exact dates for 6,072 overlapping companies)
      2. Company Status from deal data (Acquisition/Went Public/Merger/LBO)
         + IPO Date for IPO timing

    For companies with an exit file match: use the exact exit date.
    For companies with Company Status indicating exit but no file match:
      - IPO: use IPO Date as exit date
      - Acquisition/Merger/LBO: flag as status_exit (exit occurred, date unknown)
    """
    # Source 1: exit file
    succ = df_exits[df_exits["exit_type"].isin(success_types)].copy()
    earliest = (
        succ.sort_values("exit_date", kind="mergesort")
        .groupby("company_name", sort=False)
        .agg(exit_date_file=("exit_date", "min"),
             exit_type_file=("exit_type", "first"))
        .reset_index()
    )

    out = df_firms.merge(earliest, on="company_name", how="left")
    out["observation_window_end"] = (
        out["first_deal_date"] + pd.DateOffset(years=horizon_years)
    )

    # Source 2: Company Status + IPO Date from deal data
    out["status_exit"] = out["company_status"].isin(SUCCESS_STATUS).astype(int)
    out["ipo_date"] = pd.to_datetime(out["ipo_date"], errors="coerce")

    # Determine best exit date: prefer file date, then IPO date, then None
    out["exit_date"] = out["exit_date_file"]

    # For companies with status exit but no file date, use IPO date if available
    needs_date = out["exit_date"].isna() & (out["status_exit"] == 1)
    has_ipo = needs_date & out["ipo_date"].notna()
    out.loc[has_ipo, "exit_date"] = out.loc[has_ipo, "ipo_date"]

    # Determine exit type
    out["exit_type"] = out["exit_type_file"]
    out.loc[has_ipo & out["exit_type"].isna(), "exit_type"] = "IPO"
    no_type = out["exit_type"].isna() & (out["status_exit"] == 1)
    out.loc[no_type, "exit_type"] = out.loc[no_type, "company_status"]

    # Label source tracking
    out["exit_source"] = "none"
    out.loc[out["exit_date_file"].notna(), "exit_source"] = "exit_file"
    out.loc[has_ipo & out["exit_date_file"].isna(), "exit_source"] = "ipo_date"
    # Status-only exits (date unknown)
    status_only = (out["status_exit"] == 1) & (out["exit_date"].isna())
    out.loc[status_only, "exit_source"] = "status_only"

    out["exit_inconsistent"] = (
        out["exit_date"].notna() & (out["exit_date"] <= out["first_deal_date"])
    ).astype(int)

    # exit_within_T for dated exits
    out["exit_within_T"] = (
        out["exit_date"].notna()
        & (out["exit_date"] > out["first_deal_date"])
        & (out["exit_date"] <= out["observation_window_end"])
        & (out["exit_date"] <= data_end_date)
    ).astype(int)

    # For status-only exits (date unknown): count as exit if the company
    # status is a success status. These exits definitely happened but we
    # can't verify timing. Conservative: include them.
    out.loc[status_only & (out["exit_inconsistent"] == 0), "exit_within_T"] = 1

    return out


def mark_censored_observations(df: pd.DataFrame,
                               data_end_date: pd.Timestamp = DATA_END_DATE
                               ) -> pd.DataFrame:
    df = df.copy()
    df["right_censored"] = (
        (df["observation_window_end"] > data_end_date)
        & (df["exit_within_T"] == 0)
    ).astype(int)
    return df


def filter_classification_sample(df: pd.DataFrame, verbose: bool = True
                                 ) -> pd.DataFrame:
    n0 = len(df)
    bad_exit = df["exit_inconsistent"] == 1
    censored = (df["right_censored"] == 1) & (df["exit_within_T"] == 0)
    keep = ~bad_exit & ~censored
    out = df[keep].copy()
    if verbose:
        print(f"  filter_classification_sample: {n0:,} -> {len(out):,} "
              f"(dropped {int(bad_exit.sum())} inconsistent-exit, "
              f"{int(censored.sum())} right-censored)")
    return out


# ════════════════════════════════════════════════════════════════════════════
#  Temporal split
# ════════════════════════════════════════════════════════════════════════════

def temporal_train_val_test_split(df: pd.DataFrame,
                                  train_end: pd.Timestamp = TRAIN_END,
                                  val_end: pd.Timestamp = VAL_END,
                                  test_end: pd.Timestamp | None = None,
                                  date_col: str = "first_deal_date"):
    if test_end is None:
        test_end = df[date_col].max()
    assert train_end < val_end <= test_end, "split boundaries out of order"

    train = df[df[date_col] <= train_end].copy()
    val   = df[(df[date_col] > train_end) & (df[date_col] <= val_end)].copy()
    test  = df[(df[date_col] > val_end) & (df[date_col] <= test_end)].copy()

    assert_temporal_split(train, val, test, date_col)
    return train, val, test


def assert_temporal_split(train, val, test, date_col: str = "first_deal_date"):
    for name, s in [("train", train), ("val", val), ("test", test)]:
        assert len(s) > 0, f"{name} split is empty"
    assert train[date_col].max() < val[date_col].min(), "train overlaps val in time"
    assert val[date_col].max() < test[date_col].min(), "val overlaps test in time"


# ════════════════════════════════════════════════════════════════════════════
#  As-of-date co-investment graph snapshots
# ════════════════════════════════════════════════════════════════════════════

def build_snapshot_dates(start: pd.Timestamp = SNAPSHOT_START,
                         end: pd.Timestamp = SNAPSHOT_END,
                         freq: str = SNAPSHOT_FREQ) -> pd.DatetimeIndex:
    dates = pd.date_range(start, end, freq=freq)
    assert len(dates) > 0, "empty snapshot grid"
    return dates


def assign_snapshot(anchor_dates: pd.Series,
                    snapshot_dates: pd.DatetimeIndex) -> pd.Series:
    snaps = snapshot_dates.sort_values()
    idx = np.searchsorted(snaps.values, anchor_dates.values, side="right") - 1
    assert (idx >= 0).all(), "anchor date precedes the first snapshot"
    assigned = pd.Series(snaps.values[idx], index=anchor_dates.index,
                         name="snapshot_date")
    assert (assigned.values <= anchor_dates.values).all(), \
        "snapshot assignment leaked past an anchor date"
    return assigned


def build_coinvestment_snapshot(deals_df: pd.DataFrame,
                                as_of_date: pd.Timestamp,
                                node_col: str = "investor_org_id") -> nx.Graph:
    """
    Weighted undirected co-investment graph from deals with
    investment_date <= as_of_date ONLY.

    Nodes are investor org IDs (Refinitiv Firm Investor Organization Id).
    Edge weight = number of co-invested deals.
    """
    sub = deals_df[deals_df["investment_date"] <= as_of_date].copy()
    if len(sub):
        assert sub["investment_date"].max() <= as_of_date

    G = nx.Graph(as_of_date=str(as_of_date.date()))

    # Add nodes with investor name attribute
    node_attrs = (
        sub[[node_col, "investor_name"]]
        .drop_duplicates(subset=node_col)
        .set_index(node_col)
        .to_dict("index")
    )
    for nid, attrs in node_attrs.items():
        G.add_node(nid, **attrs)

    # Group deals by (company_name, investment_date, round_number) to
    # identify co-investment events. Investors in the same round = co-investors.
    deal_groups = (
        sub.groupby(["company_name", "investment_date"])[node_col]
        .apply(lambda s: set(s.dropna()))
    )
    for investors in deal_groups:
        if len(investors) < 2:
            continue
        for u, v in combinations(sorted(investors), 2):
            if G.has_edge(u, v):
                G[u][v]["weight"] += 1
            else:
                G.add_edge(u, v, weight=1)
    return G


def compute_snapshot_metrics(G: nx.Graph,
                             betweenness_k: int = BETWEENNESS_K,
                             seed: int = SEED) -> pd.DataFrame:
    n = G.number_of_nodes()
    if n == 0:
        return pd.DataFrame()

    degree          = dict(G.degree())
    weighted_degree = dict(G.degree(weight="weight"))
    k = min(betweenness_k, n)
    betweenness = nx.betweenness_centrality(G, k=k, normalized=True, seed=seed)
    clustering  = nx.clustering(G)
    pagerank    = nx.pagerank(G, alpha=0.85)

    lcc_nodes = max(nx.connected_components(G), key=len) if n else set()
    try:
        ev_lcc = nx.eigenvector_centrality_numpy(G.subgraph(lcc_nodes), weight=None) \
            if len(lcc_nodes) > 2 else {nid: 0.0 for nid in lcc_nodes}
    except Exception:
        ev_lcc = {nid: 0.0 for nid in lcc_nodes}

    rows = []
    for nid in G.nodes():
        attrs = G.nodes[nid]
        in_lcc = nid in lcc_nodes
        rows.append({
            "investor_org_id":        nid,
            "investor_name":          attrs.get("investor_name"),
            "degree":                 degree[nid],
            "weighted_degree":        weighted_degree[nid],
            "betweenness_centrality": betweenness[nid],
            "pagerank":               pagerank[nid],
            "eigenvector_lcc":        ev_lcc[nid] if in_lcc else float("nan"),
            "in_lcc":                 in_lcc,
            "clustering_coeff":       clustering[nid],
        })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
#  Leakage-safe preprocessing (fit on train ONLY)
# ════════════════════════════════════════════════════════════════════════════

OHE_MIN_FREQ = 50


def fit_preprocessor(X_train: pd.DataFrame,
                     numeric_cols: list[str],
                     binary_cols: list[str],
                     categorical_cols: list[str]) -> ColumnTransformer:
    transformers = [("num", StandardScaler(), numeric_cols)]
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(
            min_frequency=OHE_MIN_FREQ,
            handle_unknown="infrequent_if_exist",
            sparse_output=False,
        ), categorical_cols))
    if binary_cols:
        transformers.append(("bin", "passthrough", binary_cols))
    pre = ColumnTransformer(transformers, remainder="drop")
    pre.fit(X_train)
    return pre


def transform_preprocessor(preprocessor: ColumnTransformer,
                           X: pd.DataFrame) -> np.ndarray:
    return preprocessor.transform(X)


def preprocessor_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "num" or name == "bin":
            names.extend(list(cols))
        elif name == "cat":
            names.extend(trans.get_feature_names_out(cols).tolist())
    return names


# ════════════════════════════════════════════════════════════════════════════
#  Shared feature engineering (used by modelling scripts)
# ════════════════════════════════════════════════════════════════════════════

# Reduced, literature-informed feature set (validated via forward/backward
# selection + PageRank-preference literature review). Raw round size, deal
# value, round number, syndicate size, round-stage/state categoricals, and
# the size-missing flag were all found redundant or dead; the centrality
# block was collapsed to PageRank alone. This is the SAME set every
# modelling script (03, 05, 07, graph_core.py) uses -- previously each
# hardcoded its own local copy since these constants lagged behind that
# reduction; now they are the single source of truth again.
BASELINE_NUM_COLS = ["log_first_round_raised", "cohort_year", "firm_age_at_first_round"]
NETWORK_NUM_COLS = ["mean_pagerank"]
BASELINE_BINARY_COLS: list[str] = []
NETWORK_BINARY_COLS = ["missing_evc"]
CAT_FEATURE_COLS = ["sector"]


def engineer_common_covariates(df: pd.DataFrame,
                               date_col: str = "first_deal_date") -> pd.DataFrame:
    """The reduced baseline covariates only -- no leakage assertions, no
    label-based row filtering. Safe for ANY panel sharing this schema,
    including ones without exit_within_T/right_censored semantics (e.g.
    the survival panel in survival_core.py, which intentionally retains
    censored rows). engineer_baseline_features() below layers the
    classification-pipeline-specific guards on top of this."""
    df = df.copy()
    df["log_first_round_raised"] = np.log1p(
        df["first_round_raised_mn"].clip(lower=0).fillna(0))
    df["cohort_year"] = df[date_col].dt.year
    df["firm_age_at_first_round"] = df["firm_age_at_first_round"].fillna(0).clip(lower=0)
    for c in CAT_FEATURE_COLS:
        df[c] = df[c].fillna("Unknown").astype(str)
    return df


def engineer_baseline_features(df: pd.DataFrame,
                               label_col: str = "exit_within_T",
                               date_col: str = "first_deal_date") -> pd.DataFrame:
    df = df.dropna(subset=[label_col, date_col]).copy()
    assert (df["right_censored"] == 0).all(), "censored rows reached modelling"
    assert (df["observation_window_end"] <= DATA_END_DATE).all()
    return engineer_common_covariates(df, date_col)


def engineer_network_features(df: pd.DataFrame,
                              label_col: str = "exit_within_T",
                              date_col: str = "first_deal_date") -> pd.DataFrame:
    df = engineer_baseline_features(df, label_col, date_col)
    # NOTE: named after eigenvector centrality (only defined within the
    # graph's largest connected component -- see compute_snapshot_metrics),
    # but kept as a feature independent of dropping raw eigenvector
    # centrality from NETWORK_NUM_COLS: PageRank alone can't distinguish
    # "peripheral but connected" from "not in the connected core at all",
    # so this flag carries connectivity information PageRank doesn't.
    df["missing_evc"] = df["mean_eigenvector_lcc"].isna().astype(int)
    for col in NETWORK_NUM_COLS:
        df[col] = df[col].fillna(0.0)
    return df


# ════════════════════════════════════════════════════════════════════════════
#  Shared model-selection and evaluation helpers (used by 03, 05, 10)
# ════════════════════════════════════════════════════════════════════════════

def tune_on_val(make_fn, param_grid: list[dict], X_tr, y_tr, X_val, y_val):
    """Fit each config in param_grid on train, score on val, keep the best
    by ROC-AUC (PR-AUC as tiebreaker). Returns (best_params, best_roc, best_pr)."""
    best_roc, best_pr, best_params = -1.0, -1.0, {}
    for params in param_grid:
        clf = make_fn(**params)
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_val)[:, 1]
        roc = roc_auc_score(y_val, prob)
        pr = average_precision_score(y_val, prob)
        if (roc > best_roc) or (roc == best_roc and pr > best_pr):
            best_roc, best_pr, best_params = roc, pr, params
    return best_params, best_roc, best_pr


def tune_threshold(y_val, prob_val, metric: str = "f1") -> tuple[float, float]:
    """Scan decision thresholds on validation predictions and return the one
    that maximizes F1 (or F2, via metric="f2") -- replaces the fixed 0.5
    cutoff used by evaluate_classifier's default, which is arbitrary given
    the ~24% positive rate in this task. Tuned on val, applied to test, so
    this is a legitimate model-selection step, not test-set leakage."""
    y_val = np.asarray(y_val)
    prob_val = np.asarray(prob_val)
    thresholds = np.linspace(0.01, 0.99, 197)
    best_t, best_score = 0.5, -1.0
    for t in thresholds:
        pred = (prob_val >= t).astype(int)
        score = (fbeta_score(y_val, pred, beta=2, zero_division=0) if metric == "f2"
                 else f1_score(y_val, pred, zero_division=0))
        if score > best_score:
            best_score, best_t = score, t
    return round(float(best_t), 4), round(float(best_score), 4)


def evaluate_classifier(tag: str, y, prob,
                        split: str = "Test", out_dir: str | None = None,
                        verbose: bool = True,
                        save_calibration_csv: bool = False,
                        threshold: float = 0.5) -> dict:
    """Canonical classification metrics (ROC-AUC, PR-AUC, F1, F2, Brier)
    from true labels and predicted probabilities. Works identically for
    sklearn/XGBoost/LightGBM (predict_proba output) and the GNN (sigmoid
    output) -- callers extract probabilities themselves and pass them in.

    threshold controls the F1/F2/confusion-matrix cutoff only -- ROC-AUC,
    PR-AUC, and Brier are threshold-independent. Pass a value from
    tune_threshold() (fit on validation) instead of the 0.5 default to
    avoid an arbitrary cutoff on an imbalanced (~24% positive) task.

    If out_dir is given, saves a calibration plot (and optionally a CSV of
    the underlying curve) -- matplotlib is imported lazily inside this
    function so pipeline_core.py never forces a backend choice; callers
    must set matplotlib.use("Agg") (or similar) before this is first
    invoked, same convention as every script already follows."""
    prob = np.asarray(prob)
    pred = (prob >= threshold).astype(int)

    roc   = roc_auc_score(y, prob)
    prauc = average_precision_score(y, prob)
    f1    = f1_score(y, pred, zero_division=0)
    f2    = fbeta_score(y, pred, beta=2, zero_division=0)
    brier = brier_score_loss(y, prob)

    if verbose:
        cm = confusion_matrix(y, pred)
        print(f"\n  [{tag}] {split}:  ROC-AUC={roc:.4f}  PR-AUC={prauc:.4f}  "
              f"F1={f1:.4f}  Brier={brier:.4f}")
        print(f"    CM: {cm.tolist()}")

    if out_dir is not None:
        import os
        import matplotlib.pyplot as plt
        frac_pos, mean_pred = calibration_curve(y, prob, n_bins=10, strategy="uniform")
        if save_calibration_csv:
            pd.DataFrame({"mean_predicted": mean_pred, "fraction_positive": frac_pos}).to_csv(
                os.path.join(out_dir, f"calibration_{tag.lower()}_{split.lower()}.csv"),
                index=False)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(mean_pred, frac_pos, "s-", label=tag)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction positive")
        ax.set_title(f"Calibration — {tag} ({split})")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"calibration_{tag.lower()}.png"), dpi=120)
        plt.close(fig)

    return {"model": tag, "split": split,
            "roc_auc": round(float(roc), 4), "pr_auc": round(float(prauc), 4),
            "f1": round(float(f1), 4), "f2": round(float(f2), 4),
            "brier": round(float(brier), 4), "threshold": round(float(threshold), 4)}


def evaluate_with_tuned_threshold(tag: str, model, X_val, y_val, X_te, y_te,
                                  split: str = "Test", out_dir: str | None = None,
                                  save_calibration_csv: bool = False,
                                  verbose: bool = True) -> dict:
    """sklearn-style convenience wrapper: tune the decision threshold on
    validation predictions, then score the fitted model on test at that
    threshold instead of the fixed 0.5 default. Used by 03/05 in place of
    calling predict_proba + evaluate_classifier separately."""
    val_prob = model.predict_proba(X_val)[:, 1]
    threshold, val_f1 = tune_threshold(y_val, val_prob)
    test_prob = model.predict_proba(X_te)[:, 1]
    result = evaluate_classifier(tag, y_te, test_prob, split=split, out_dir=out_dir,
                                 verbose=verbose, save_calibration_csv=save_calibration_csv,
                                 threshold=threshold)
    result["val_threshold_f1"] = val_f1
    return result
