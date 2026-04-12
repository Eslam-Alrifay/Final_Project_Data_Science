"""
Stock Market Data Cleaning Pipeline
=====================================
Loads, cleans, validates, and visualizes stock data for multiple companies.
"""

import warnings
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  –  Edit only this section
# ══════════════════════════════════════════════════════════════════════════════
INPUT        = Path(r"F:\Data Scince\Final Project\all_companies.csv")
OUTPUT       = Path(r"F:\Data Scince\Final Project\all_companies_cleaned.csv")
PLOTS_DIR    = Path(r"F:\Data Scince\Final Project\plots")

PRICE_COLS   = ["Price", "Open", "High", "Low"]
DATE_FORMAT  = "%m/%d/%Y"
MULTIPLIERS  = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}

# ── Visual theme ──────────────────────────────────────────────────────────────
PALETTE       = "Set2"
ACCENT_COLOR  = "#2196F3"   # blue   – general bars
GOOD_COLOR    = "#4CAF50"   # green  – "after" state
BAD_COLOR     = "#F44336"   # red    – "before" / suspicious
FIG_DPI       = 130

sns.set_theme(style="whitegrid", context="talk", palette=PALETTE)
plt.rcParams.update({
    "figure.dpi"        : FIG_DPI,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.titlepad"     : 14,
})

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def section(title: str) -> None:
    """Print a styled section header."""
    bar = "═" * 62
    print(f"\n{bar}\n  ▸  {title}\n{bar}")


def convert_volume(val) -> float:
    """Convert volume strings like '1.2M' or '500K' to float."""
    if pd.isna(val):
        return np.nan
    val = str(val).strip().replace(",", "")
    if val in ("", "-", "nan", "None"):
        return np.nan
    suffix = val[-1].upper()
    if suffix in MULTIPLIERS:
        return float(val[:-1]) * MULTIPLIERS[suffix]
    return pd.to_numeric(val, errors="coerce")


def apply_type_conversions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all type conversions to a copy. Used for before/after snapshots."""
    df = df.copy()
    df.columns   = df.columns.str.strip()
    df["Date"]   = pd.to_datetime(df["Date"], format=DATE_FORMAT, errors="coerce")
    df["Vol."]   = df["Vol."].apply(convert_volume)
    df["Change %"] = (
        df["Change %"].astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
    )
    for col in PRICE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def detect_suspicious(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where OHLC price relationships are logically violated."""
    mask = (
        (df["High"]  < df["Low"])   |
        (df["Open"]  > df["High"])  |
        (df["Open"]  < df["Low"])   |
        (df["Price"] > df["High"])  |
        (df["Price"] < df["Low"])
    )
    return df[mask].copy()


def quality_snapshot(df: pd.DataFrame) -> dict:
    """Return a dict of key data-quality metrics."""
    return {
        "Missing Values"       : df.isnull().sum().sum(),
        "Duplicate Rows"       : df.duplicated().sum(),
        "Dup. Company-Date"    : df.duplicated(subset=["Company", "Date"]).sum(),
        "Suspicious OHLC Rows" : len(detect_suspicious(df)),
    }


def save_fig(filename: str) -> None:
    plt.savefig(PLOTS_DIR / filename, dpi=FIG_DPI, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load Dataset
# ══════════════════════════════════════════════════════════════════════════════
section("1. Load Dataset")
stocks = pd.read_csv(INPUT)
print(f"  Shape  :  {stocks.shape[0]:,} rows  ×  {stocks.shape[1]} columns")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Initial Inspection
# ══════════════════════════════════════════════════════════════════════════════
section("2. Initial Inspection")
print(f"\n── First 5 rows ──\n{stocks.head()}")
print(f"\n── Columns       : {list(stocks.columns)}")
print(f"\n── Data types & non-null counts ──")
stocks.info()
print(f"\n── Null counts per column ──\n{stocks.isnull().sum()}")
print(f"\n── Duplicate rows    : {stocks.duplicated().sum():,}")
print(f"── Unique companies  : {stocks['Company'].nunique():,}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Data Quality Snapshot  —  Before Cleaning
# ══════════════════════════════════════════════════════════════════════════════
section("3. Data Quality Snapshot  —  Before Cleaning")
raw    = apply_type_conversions(stocks)   # one-time conversion for comparison
BEFORE = quality_snapshot(raw)

for metric, value in BEFORE.items():
    print(f"  {metric:<26}: {value:,}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Clean Column Names
# ══════════════════════════════════════════════════════════════════════════════
section("4. Clean Column Names  (strip whitespace)")
stocks.columns = stocks.columns.str.strip()
print(f"  Columns → {list(stocks.columns)}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Convert Date Column
# ══════════════════════════════════════════════════════════════════════════════
section("5. Convert 'Date'  →  datetime")
stocks["Date"] = pd.to_datetime(stocks["Date"], format=DATE_FORMAT, errors="coerce")
print(f"  Invalid dates (NaT): {stocks['Date'].isna().sum():,}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Convert Numeric Price Columns
# ══════════════════════════════════════════════════════════════════════════════
section(f"6. Convert Numeric Columns  →  {PRICE_COLS}")
for col in PRICE_COLS:
    stocks[col] = pd.to_numeric(stocks[col], errors="coerce")
print("  Done.")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Clean 'Change %'
# ══════════════════════════════════════════════════════════════════════════════
section("7. Clean 'Change %'  (strip %, cast to float)")
stocks["Change %"] = (
    stocks["Change %"].astype(str)
    .str.replace("%", "", regex=False)
    .str.strip()
    .pipe(pd.to_numeric, errors="coerce")
)
print(f"  Invalid after conversion: {stocks['Change %'].isna().sum():,}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Clean 'Vol.'
# ══════════════════════════════════════════════════════════════════════════════
section("8. Clean 'Vol.'  (K / M / B  →  actual numbers)")
stocks["Vol."] = stocks["Vol."].apply(convert_volume)
print(f"  Missing volumes: {stocks['Vol.'].isna().sum():,}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. Remove Duplicate Rows
# ══════════════════════════════════════════════════════════════════════════════
section("9. Remove Duplicate Rows")
n_before   = len(stocks)
stocks     = stocks.drop_duplicates()
n_removed  = n_before - len(stocks)
dup_pairs  = stocks.duplicated(subset=["Company", "Date"]).sum()

print(f"  Removed  : {n_removed:,}  |  Remaining : {len(stocks):,}")
print(f"  Remaining (Company, Date) duplicates: {dup_pairs:,}")


# ══════════════════════════════════════════════════════════════════════════════
# 10. Sort Data
# ══════════════════════════════════════════════════════════════════════════════
section("10. Sort Data  (preserve Company order → then by Date)")
company_order      = stocks["Company"].drop_duplicates()
stocks["Company"]  = pd.Categorical(stocks["Company"], categories=company_order, ordered=True)
stocks             = stocks.sort_values(["Company", "Date"]).reset_index(drop=True)
stocks["Company"]  = stocks["Company"].astype(str)
print(stocks.head(10))


# ══════════════════════════════════════════════════════════════════════════════
# 11. Impute Missing Volume
# ══════════════════════════════════════════════════════════════════════════════
section("11. Impute Missing Volume  (linear interpolation → ffill → bfill)")
missing_vol_before = stocks["Vol."].isna().sum()

stocks["Vol."] = (
    stocks.groupby("Company")["Vol."]
    .transform(lambda s: s.interpolate(method="linear").ffill().bfill())
)
stocks["Vol."] = stocks["Vol."].round().astype("Int64")

print(f"  Missing before : {missing_vol_before:,}  |  After : {stocks['Vol.'].isna().sum():,}")


# ══════════════════════════════════════════════════════════════════════════════
# 12. Detect Suspicious OHLC Rows  —  Before Fix
# ══════════════════════════════════════════════════════════════════════════════
section("12. Detect Suspicious OHLC Rows  —  Before Fix")
invalid_before_fix = detect_suspicious(stocks)
print(f"  Suspicious rows: {len(invalid_before_fix):,}")
if not invalid_before_fix.empty:
    print(invalid_before_fix.head(10))


# ══════════════════════════════════════════════════════════════════════════════
# 13. Fix Suspicious Price Rows
# ══════════════════════════════════════════════════════════════════════════════
section("13. Fix Suspicious Price Rows")

# Step 1: swap inverted High / Low  (use .values to avoid index-alignment bugs)
swap_mask  = stocks["High"] < stocks["Low"]
swap_count = swap_mask.sum()
stocks.loc[swap_mask, ["High", "Low"]] = (
    stocks.loc[swap_mask, ["Low", "High"]].values
)

# Step 2: expand range to contain Open and Price
stocks["High"] = stocks[["High", "Open", "Price"]].max(axis=1)
stocks["Low"]  = stocks[["Low",  "Open", "Price"]].min(axis=1)

print(f"  Swapped High/Low pairs  : {swap_count:,}")
print("  Range expanded to include Open and Price.")


# ══════════════════════════════════════════════════════════════════════════════
# 14. Validate After Fix
# ══════════════════════════════════════════════════════════════════════════════
section("14. Validate  —  After Fix")
invalid_after_fix = detect_suspicious(stocks)
print(f"  Suspicious rows remaining: {len(invalid_after_fix):,}")
if not invalid_after_fix.empty:
    print(invalid_after_fix.head(10))


# ══════════════════════════════════════════════════════════════════════════════
# 15. Final Missing Values
# ══════════════════════════════════════════════════════════════════════════════
section("15. Final Missing Values")
print(stocks.isnull().sum().to_string())


# ══════════════════════════════════════════════════════════════════════════════
# 16. Visualizations
# ══════════════════════════════════════════════════════════════════════════════
section("16. Visualizations")

AFTER = quality_snapshot(stocks)
n_companies = stocks["Company"].nunique()


# ── A) Missing Values per Column ─────────────────────────────────────────────
missing_df = stocks.isnull().sum().reset_index()
missing_df.columns = ["Column", "Missing"]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(missing_df["Column"], missing_df["Missing"],
              color=ACCENT_COLOR, edgecolor="white", linewidth=0.8)
ax.bar_label(bars, padding=3, fontsize=10)
ax.set_title("Missing Values per Column — After Cleaning", fontsize=14, weight="bold")
ax.set_xlabel("Column")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
save_fig("A_missing_values.png")


# ── B) Rows per Company ───────────────────────────────────────────────────────
rows_df = (
    stocks["Company"].value_counts()
    .sort_values(ascending=True)
    .reset_index()
)
rows_df.columns = ["Company", "Rows"]

fig_h = max(6, n_companies * 0.38)
fig, ax = plt.subplots(figsize=(12, fig_h))
colors = sns.color_palette(PALETTE, len(rows_df))
bars   = ax.barh(rows_df["Company"], rows_df["Rows"],
                 color=colors, edgecolor="white")
ax.bar_label(bars, padding=4, fontsize=9)
ax.set_title("Number of Rows per Company", fontsize=14, weight="bold")
ax.set_xlabel("Rows")
plt.tight_layout()
save_fig("B_rows_per_company.png")


# ── C) Distribution of Change % ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(stocks["Change %"].dropna(), bins=40, kde=True,
             color=ACCENT_COLOR, edgecolor="white", linewidth=0.4, ax=ax)
ax.axvline(0, color=BAD_COLOR, linestyle="--", linewidth=1.4, label="Zero")
ax.set_title("Distribution of Daily Change %", fontsize=14, weight="bold")
ax.set_xlabel("Change %")
ax.set_ylabel("Frequency")
ax.legend(fontsize=11)
plt.tight_layout()
save_fig("C_change_pct_distribution.png")


# ── D) Suspicious Rows per Company (Before Fix) ───────────────────────────────
if not invalid_before_fix.empty:
    susp_df = (
        invalid_before_fix["Company"].value_counts()
        .sort_values(ascending=True)
        .reset_index()
    )
    susp_df.columns = ["Company", "Count"]

    fig, ax = plt.subplots(figsize=(12, max(5, len(susp_df) * 0.38)))
    bars = ax.barh(susp_df["Company"], susp_df["Count"],
                   color=BAD_COLOR, edgecolor="white", alpha=0.85)
    ax.bar_label(bars, padding=4, fontsize=9)
    ax.set_title("Suspicious OHLC Rows per Company — Before Fix",
                 fontsize=14, weight="bold")
    ax.set_xlabel("Count")
    plt.tight_layout()
    save_fig("D_suspicious_rows.png")
else:
    print("  No suspicious rows found before fixing.")


# ── E) Before vs After Data Quality ─────────────────────────────────────────
metrics      = list(BEFORE.keys())
before_vals  = list(BEFORE.values())
after_vals   = list(AFTER.values())
x            = np.arange(len(metrics))
width        = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
b1 = ax.bar(x - width / 2, before_vals, width,
            label="Before", color=BAD_COLOR,  alpha=0.85, edgecolor="white")
b2 = ax.bar(x + width / 2, after_vals,  width,
            label="After",  color=GOOD_COLOR, alpha=0.85, edgecolor="white")

ax.bar_label(b1, padding=3, fontsize=10)
ax.bar_label(b2, padding=3, fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=12, ha="right")
ax.set_title("Data Quality: Before vs After Preprocessing",
             fontsize=14, weight="bold")
ax.set_ylabel("Count")
ax.legend(fontsize=11)
plt.tight_layout()
save_fig("E_quality_before_after.png")


# ══════════════════════════════════════════════════════════════════════════════
# 17. Save Cleaned Dataset
# ══════════════════════════════════════════════════════════════════════════════
section("17. Save Cleaned Dataset")
# stocks.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
print(f"  Saved  →  {OUTPUT}")
print(f"  Shape  :  {stocks.shape[0]:,} rows   {stocks.shape[1]} columns")
print(f"\n── Final Preview ──\n{stocks.head(10)}")