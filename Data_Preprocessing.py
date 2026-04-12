import warnings
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
INPUT = Path(r"F:\Data_Scince\Final_Project\all_companies.csv")
OUTPUT = Path(r"F:\Data Scince\Final_Project\all_companies_cleaned.csv")
PLOTS_DIR = Path(r"F:\Data Scince\Final_Project\plots")

PRICE_COLS = ["Price", "Open", "High", "Low"]
DATE_FORMAT = "%m/%d/%Y"
MULTIPLIERS = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}

PALETTE = "Set2"
ACCENT_COLOR = "#2196F3"
GOOD_COLOR = "#4CAF50"
BAD_COLOR = "#F44336"
FIG_DPI = 130


# ══════════════════════════════════════════════════════════════════════════════
# STYLE
# ══════════════════════════════════════════════════════════════════════════════
sns.set_theme(style="whitegrid", context="talk", palette=PALETTE)
plt.rcParams.update({
    "figure.dpi": FIG_DPI,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 14,
})

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1) LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("1) LOAD DATA")
print("═" * 70)

stocks = pd.read_csv(INPUT)
print(f"Shape: {stocks.shape[0]:,} rows  {stocks.shape[1]} columns")


# ══════════════════════════════════════════════════════════════════════════════
# 2) INITIAL INSPECTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("2) INITIAL INSPECTION")
print("═" * 70)

print("\nFirst 5 rows:")
print(stocks.head())

print("\nColumns:")
print(list(stocks.columns))

print("\nData types:")
stocks.info()

print("\nNull counts per column:")
print(stocks.isnull().sum())

print(f"\nDuplicate rows: {stocks.duplicated().sum():,}")
print(f"Unique companies: {stocks['Company'].nunique():,}")


# ══════════════════════════════════════════════════════════════════════════════
# 3) BEFORE CLEANING SNAPSHOT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("3) BEFORE CLEANING SNAPSHOT")
print("═" * 70)

raw = stocks.copy()
raw.columns = raw.columns.str.strip()

raw["Date"] = pd.to_datetime(raw["Date"], format=DATE_FORMAT, errors="coerce")

for col in PRICE_COLS:
    raw[col] = pd.to_numeric(raw[col], errors="coerce")

raw["Change %"] = (
    raw["Change %"]
    .astype(str)
    .str.replace("%", "", regex=False)
    .str.strip()
)
raw["Change %"] = pd.to_numeric(raw["Change %"], errors="coerce")

raw_vol = raw["Vol."].astype(str).str.strip().str.replace(",", "", regex=False)
raw_vol = raw_vol.replace({"": np.nan, "-": np.nan, "nan": np.nan, "None": np.nan})

raw["Vol."] = pd.to_numeric(raw_vol, errors="coerce")

for suffix, multiplier in MULTIPLIERS.items():
    mask = raw_vol.str.endswith(suffix, na=False)
    raw.loc[mask, "Vol."] = pd.to_numeric(raw_vol[mask].str[:-1], errors="coerce") * multiplier

raw_suspicious_mask = (
    (raw["High"] < raw["Low"]) |
    (raw["Open"] > raw["High"]) |
    (raw["Open"] < raw["Low"]) |
    (raw["Price"] > raw["High"]) |
    (raw["Price"] < raw["Low"])
)

BEFORE = {
    "Missing Values": int(raw.isnull().sum().sum()),
    "Duplicate Rows": int(raw.duplicated().sum()),
    "Dup. Company-Date": int(raw.duplicated(subset=["Company", "Date"]).sum()),
    "Suspicious OHLC Rows": int(raw_suspicious_mask.sum()),
}

for metric, value in BEFORE.items():
    print(f"{metric:<24}: {value:,}")


# ══════════════════════════════════════════════════════════════════════════════
# 4) CLEAN COLUMNS AND TYPES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("4) CLEAN COLUMNS AND TYPES")
print("═" * 70)

stocks.columns = stocks.columns.str.strip()

stocks["Date"] = pd.to_datetime(stocks["Date"], format=DATE_FORMAT, errors="coerce")

for col in PRICE_COLS:
    stocks[col] = pd.to_numeric(stocks[col], errors="coerce")

stocks["Change %"] = (
    stocks["Change %"]
    .astype(str)
    .str.replace("%", "", regex=False)
    .str.strip()
)
stocks["Change %"] = pd.to_numeric(stocks["Change %"], errors="coerce")

vol = stocks["Vol."].astype(str).str.strip().str.replace(",", "", regex=False)
vol = vol.replace({"": np.nan, "-": np.nan, "nan": np.nan, "None": np.nan})

stocks["Vol."] = pd.to_numeric(vol, errors="coerce")

for suffix, multiplier in MULTIPLIERS.items():
    mask = vol.str.endswith(suffix, na=False)
    stocks.loc[mask, "Vol."] = pd.to_numeric(vol[mask].str[:-1], errors="coerce") * multiplier

print(f"Invalid dates (NaT): {stocks['Date'].isna().sum():,}")
print(f"Invalid Change % values: {stocks['Change %'].isna().sum():,}")
print(f"Missing volumes: {stocks['Vol.'].isna().sum():,}")


# ══════════════════════════════════════════════════════════════════════════════
# 5) REMOVE DUPLICATES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("5) REMOVE DUPLICATES")
print("═" * 70)

rows_before = len(stocks)
stocks = stocks.drop_duplicates().copy()
rows_removed = rows_before - len(stocks)
dup_company_date = stocks.duplicated(subset=["Company", "Date"]).sum()

print(f"Removed duplicate rows: {rows_removed:,}")
print(f"Remaining rows: {len(stocks):,}")
print(f"Remaining duplicate Company-Date rows: {dup_company_date:,}")


# ══════════════════════════════════════════════════════════════════════════════
# 6) SORT DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("6) SORT DATA")
print("═" * 70)

company_order = stocks["Company"].drop_duplicates()
stocks["Company"] = pd.Categorical(stocks["Company"], categories=company_order, ordered=True)

stocks = stocks.sort_values(["Company", "Date"]).reset_index(drop=True)
stocks["Company"] = stocks["Company"].astype(str)

print(stocks.head(10))


# ══════════════════════════════════════════════════════════════════════════════
# 7) FILL MISSING VOLUME
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("7) FILL MISSING VOLUME")
print("═" * 70)

missing_vol_before = stocks["Vol."].isna().sum()

stocks["Vol."] = (
    stocks.groupby("Company")["Vol."]
    .transform(lambda s: s.interpolate(method="linear").ffill().bfill())
)

stocks["Vol."] = stocks["Vol."].round().astype("Int64")

missing_vol_after = stocks["Vol."].isna().sum()

print(f"Missing volume before: {missing_vol_before:,}")
print(f"Missing volume after : {missing_vol_after:,}")


# ══════════════════════════════════════════════════════════════════════════════
# 8) DETECT SUSPICIOUS OHLC BEFORE FIX
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("8) DETECT SUSPICIOUS OHLC BEFORE FIX")
print("═" * 70)

suspicious_before_mask = (
    (stocks["High"] < stocks["Low"]) |
    (stocks["Open"] > stocks["High"]) |
    (stocks["Open"] < stocks["Low"]) |
    (stocks["Price"] > stocks["High"]) |
    (stocks["Price"] < stocks["Low"])
)

invalid_before_fix = stocks.loc[suspicious_before_mask].copy()

print(f"Suspicious rows before fix: {len(invalid_before_fix):,}")
if not invalid_before_fix.empty:
    print(invalid_before_fix.head(10))


# ══════════════════════════════════════════════════════════════════════════════
# 9) FIX SUSPICIOUS OHLC
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("9) FIX SUSPICIOUS OHLC")
print("═" * 70)

swap_mask = stocks["High"] < stocks["Low"]
swap_count = int(swap_mask.sum())

stocks.loc[swap_mask, ["High", "Low"]] = stocks.loc[swap_mask, ["Low", "High"]].values

stocks["High"] = stocks[["High", "Open", "Price"]].max(axis=1)
stocks["Low"] = stocks[["Low", "Open", "Price"]].min(axis=1)

print(f"Swapped High/Low pairs: {swap_count:,}")
print("Adjusted High and Low to include Open and Price")


# ══════════════════════════════════════════════════════════════════════════════
# 10) VALIDATE AFTER FIX
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("10) VALIDATE AFTER FIX")
print("═" * 70)

suspicious_after_mask = (
    (stocks["High"] < stocks["Low"]) |
    (stocks["Open"] > stocks["High"]) |
    (stocks["Open"] < stocks["Low"]) |
    (stocks["Price"] > stocks["High"]) |
    (stocks["Price"] < stocks["Low"])
)

invalid_after_fix = stocks.loc[suspicious_after_mask].copy()

print(f"Suspicious rows after fix: {len(invalid_after_fix):,}")
if not invalid_after_fix.empty:
    print(invalid_after_fix.head(10))


# ══════════════════════════════════════════════════════════════════════════════
# 11) FINAL MISSING VALUES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("11) FINAL MISSING VALUES")
print("═" * 70)

print(stocks.isnull().sum().to_string())


# ══════════════════════════════════════════════════════════════════════════════
# 12) AFTER CLEANING SNAPSHOT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("12) AFTER CLEANING SNAPSHOT")
print("═" * 70)

AFTER = {
    "Missing Values": int(stocks.isnull().sum().sum()),
    "Duplicate Rows": int(stocks.duplicated().sum()),
    "Dup. Company-Date": int(stocks.duplicated(subset=["Company", "Date"]).sum()),
    "Suspicious OHLC Rows": int(suspicious_after_mask.sum()),
}

for metric, value in AFTER.items():
    print(f"{metric:<24}: {value:,}")


# ══════════════════════════════════════════════════════════════════════════════
# 13) VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("13) VISUALIZATIONS")
print("═" * 70)

# A) Missing values per column
missing_df = stocks.isnull().sum().reset_index()
missing_df.columns = ["Column", "Missing"]

plt.figure(figsize=(10, 5))
bars = plt.bar(
    missing_df["Column"],
    missing_df["Missing"],
    color=ACCENT_COLOR,
    edgecolor="white",
    linewidth=0.8
)
plt.bar_label(bars, padding=3, fontsize=10)
plt.title("Missing Values per Column — After Cleaning", fontsize=14, weight="bold")
plt.xlabel("Column")
plt.ylabel("Count")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "A_missing_values.png", dpi=FIG_DPI, bbox_inches="tight")
plt.show()


# B) Rows per company
rows_df = stocks["Company"].value_counts().sort_values(ascending=True).reset_index()
rows_df.columns = ["Company", "Rows"]

fig_h = max(6, stocks["Company"].nunique() * 0.38)

plt.figure(figsize=(12, fig_h))
colors = sns.color_palette(PALETTE, len(rows_df))
bars = plt.barh(rows_df["Company"], rows_df["Rows"], color=colors, edgecolor="white")
plt.bar_label(bars, padding=4, fontsize=9)
plt.title("Number of Rows per Company", fontsize=14, weight="bold")
plt.xlabel("Rows")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "B_rows_per_company.png", dpi=FIG_DPI, bbox_inches="tight")
plt.show()


# C) Change % distribution
plt.figure(figsize=(10, 5))
sns.histplot(
    stocks["Change %"].dropna(),
    bins=40,
    kde=True,
    color=ACCENT_COLOR,
    edgecolor="white",
    linewidth=0.4
)
plt.axvline(0, color=BAD_COLOR, linestyle="--", linewidth=1.4, label="Zero")
plt.title("Distribution of Daily Change %", fontsize=14, weight="bold")
plt.xlabel("Change %")
plt.ylabel("Frequency")
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "C_change_pct_distribution.png", dpi=FIG_DPI, bbox_inches="tight")
plt.show()


# D) Suspicious rows per company before fix
if not invalid_before_fix.empty:
    susp_df = invalid_before_fix["Company"].value_counts().sort_values(ascending=True).reset_index()
    susp_df.columns = ["Company", "Count"]

    plt.figure(figsize=(12, max(5, len(susp_df) * 0.38)))
    bars = plt.barh(
        susp_df["Company"],
        susp_df["Count"],
        color=BAD_COLOR,
        edgecolor="white",
        alpha=0.85
    )
    plt.bar_label(bars, padding=4, fontsize=9)
    plt.title("Suspicious OHLC Rows per Company — Before Fix", fontsize=14, weight="bold")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "D_suspicious_rows.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.show()
else:
    print("No suspicious rows found before fixing.")


# E) Before vs After quality
metrics = list(BEFORE.keys())
before_vals = list(BEFORE.values())
after_vals = list(AFTER.values())
x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(12, 6))
bars1 = plt.bar(
    x - width / 2,
    before_vals,
    width,
    label="Before",
    color=BAD_COLOR,
    alpha=0.85,
    edgecolor="white"
)

bars2 = plt.bar(
    x + width / 2,
    after_vals,
    width,
    label="After",
    color=GOOD_COLOR,
    alpha=0.85,
    edgecolor="white"
)

plt.bar_label(bars1, padding=3, fontsize=10)
plt.bar_label(bars2, padding=3, fontsize=10)
plt.xticks(x, metrics, rotation=12, ha="right")
plt.title("Data Quality: Before vs After Preprocessing", fontsize=14, weight="bold")
plt.ylabel("Count")
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "E_quality_before_after.png", dpi=FIG_DPI, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 14) SAVE CLEANED DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("14) SAVE CLEANED DATA")
print("═" * 70)

# stocks.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print(f"Saved → {OUTPUT}")
print(f"Shape : {stocks.shape[0]:,} rows × {stocks.shape[1]} columns")
print("\nFinal Preview:")
print(stocks.head(10))