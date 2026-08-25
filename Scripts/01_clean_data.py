"""
01_clean_data.py — load the raw Refinitiv XLSX files, sanity-check them, and
save clean parquets.

Reads three source files, all already US-only:
  1. Yearly Company Deal Level Data.xlsx, one row per (startup, round, investor)
  2. Exit Data 2010-2024 US Refinitiv.xlsx, one row per exit event
  3. Company Datas.xlsx, a company-level summary used for supplementary fields

Writes:
  Data/processed/deals.parquet       cleaned deal-level data (renamed columns)
  Data/processed/exits.parquet       cleaned exit data (renamed columns)
  Data/processed/companies.parquet   company-level summary (optional enrichment)
"""

import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_core import (
    DEAL_COL_MAP,
    load_and_rename_deals, load_and_rename_exits,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "Data")
OUT  = os.path.join(ROOT, "Data", "processed")
os.makedirs(OUT, exist_ok=True)

# Convert XLSX to CSV (convenience copies)
_DOC_RE = re.compile(r"read\s*me|readme|notes?|instructions?|changelog", re.IGNORECASE)

xlsx_files = sorted(
    f for f in os.listdir(DATA)
    if f.endswith(".xlsx") and not f.startswith("~$")
)

for filename in xlsx_files:
    xlsx_path  = os.path.join(DATA, filename)
    base_name  = os.path.splitext(filename)[0]

    xl          = pd.ExcelFile(xlsx_path)
    data_sheets = [s for s in xl.sheet_names if not _DOC_RE.search(s)]

    if not data_sheets:
        print(f"Skipped  (no data sheets): {filename}")
        continue

    if len(data_sheets) == 1:
        df       = pd.read_excel(xlsx_path, sheet_name=data_sheets[0])
        csv_path = os.path.join(DATA, base_name + ".csv")
        df.to_csv(csv_path, index=False)
        print(f"Converted: {filename} [{data_sheets[0]}] -> {base_name}.csv")
    else:
        for sheet in data_sheets:
            safe    = re.sub(r'[/\\]', '-', sheet).strip()
            csv_path = os.path.join(DATA, f"{base_name}_{safe}.csv")
            df       = pd.read_excel(xlsx_path, sheet_name=sheet)
            df.to_csv(csv_path, index=False)
            print(f"Converted: {filename} [{sheet}] -> {os.path.basename(csv_path)}")

# Load and rename deal data
print("\nLoading Yearly Company Deal Level Data")
deals = load_and_rename_deals(os.path.join(DATA, "Yearly Company Deal Level Data.xlsx"))
print(f"Shape: {deals.shape}")
print(f"Date range: {deals['investment_date'].min().date()} – {deals['investment_date'].max().date()}")
print(f"Unique companies: {deals['company_name'].nunique():,}")
print(f"Unique investors (by name): {deals['investor_name'].nunique():,}")
print(f"Unique investors (by org ID): {deals['investor_org_id'].nunique():,}")
print(f"Missing investment_date: {deals['investment_date'].isna().sum()}")

print(f"\nRound stage distribution:")
print(deals["round_stage"].value_counts(dropna=False).to_string())

print(f"\nSector distribution (top 10):")
print(deals["sector"].value_counts().head(10).to_string())

# Load and rename exit data
print("\nLoading Exit Data")
exits = load_and_rename_exits(os.path.join(DATA, "Exit Data 2010-2024 US Refinitiv.xlsx"))
print(f"Shape: {exits.shape}")
print(f"Date range: {exits['exit_date'].min().date()} – {exits['exit_date'].max().date()}")
print(f"Unique companies: {exits['company_name'].nunique():,}")
print(f"Missing exit_date: {exits['exit_date'].isna().sum()}")

print(f"\nExit type distribution:")
print(exits["exit_type"].value_counts().to_string())

# Company-level overlap
deal_cos = set(deals["company_name"].unique())
exit_cos = set(exits["company_name"].unique())
overlap  = deal_cos & exit_cos
print("\nCompany overlap (deals and exits)")
print(f"Deal companies:  {len(deal_cos):,}")
print(f"Exit companies:  {len(exit_cos):,}")
print(f"Overlap:         {len(overlap):,}")

# Load company-level data (supplementary). Whether this file adds value
# beyond the deal data is still an open question; for now it's loaded and
# saved for reference, since it may provide founded date or employee count
# as additional baseline features.
print("\nLoading Company Datas (supplementary)")
companies = pd.read_excel(os.path.join(DATA, "Company Datas.xlsx"))
# Apply the same column renaming where columns match
rename = {k: v for k, v in DEAL_COL_MAP.items() if k in companies.columns}
companies = companies.rename(columns=rename)
print(f"Shape: {companies.shape}")
print(f"Unique companies: {companies['company_name'].nunique():,}")

# Coerce mixed-type columns to string before parquet save
for df_name, df_ in [("deals", deals), ("exits", exits), ("companies", companies)]:
    for col in df_.columns:
        if df_[col].dtype == object:
            df_[col] = df_[col].astype(str).replace("nan", pd.NA)

deals.to_parquet(os.path.join(OUT, "deals.parquet"), index=False)
exits.to_parquet(os.path.join(OUT, "exits.parquet"), index=False)
companies.to_parquet(os.path.join(OUT, "companies.parquet"), index=False)

print(f"\nSaved to {OUT}:")
print(f"  deals.parquet      ({deals.shape[0]:,} rows)")
print(f"  exits.parquet      ({exits.shape[0]:,} rows)")
print(f"  companies.parquet  ({companies.shape[0]:,} rows)")
