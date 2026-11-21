"""Data-quality analysis: missing values, duplicates, outliers, schema and target validation."""
from __future__ import annotations

import pandas as pd

import config as C


def missing_value_analysis(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"missing": df.isna().sum(), "missing_pct": (df.isna().mean() * 100).round(2)})
    return out.reset_index(names="column")


def duplicate_analysis(df: pd.DataFrame) -> dict:
    return {
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "repeat_customer_garment_pairs": int(df.duplicated(["Customer_ID", "Garment_ID"]).sum()),
    }


def outlier_analysis(df: pd.DataFrame, cols=("Price", "Trial_Count")) -> pd.DataFrame:
    rows = []
    for c in cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((df[c] < lo) | (df[c] > hi)).sum())
        rows.append({"column": c, "q1": q1, "q3": q3, "lower_bound": lo, "upper_bound": hi,
                     "n_outliers": n, "outlier_pct": round(100 * n / len(df), 2)})
    return pd.DataFrame(rows)


def schema_problems(df: pd.DataFrame) -> list[str]:
    problems = []
    checks = {"Size": C.SIZES, "Sleeve_Type": C.SLEEVES, "Color": C.COLORS,
              "Collar_Type": C.COLLARS, "Fit": C.FITS, "Pattern": C.PATTERNS}
    for col, allowed in checks.items():
        bad = df.loc[~df[col].isin(allowed), col].unique().tolist()
        if bad:
            problems.append(f"{col}: unexpected values {bad}")
    if ((df["Price"] < C.PRICE_MIN) | (df["Price"] > C.PRICE_MAX)).any():
        problems.append("Price outside expected range")
    if (df["Trial_Count"] < 0).any():
        problems.append("Negative Trial_Count")
    return problems


def validate_target(df: pd.DataFrame) -> dict:
    return {
        "missing_labels": int(df[C.TARGET].isna().sum()),
        "non_binary_labels": int((~df[C.TARGET].isin([0, 1])).sum()),
        "conversion_rate": round(float(df[C.TARGET].mean()), 4),
        "class_counts": {int(k): int(v) for k, v in df[C.TARGET].value_counts().items()},
    }


def preprocessing_report(df: pd.DataFrame) -> dict:
    return {
        "missing": missing_value_analysis(df).to_dict("records"),
        "duplicates": duplicate_analysis(df),
        "outliers": outlier_analysis(df).to_dict("records"),
        "schema_problems": schema_problems(df),
        "target": validate_target(df),
        "note": "Outliers are reported, not removed: prices and trial counts are legitimate retail values.",
    }


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)
