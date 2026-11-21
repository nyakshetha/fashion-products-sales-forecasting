"""Feature aggregation (research groupings) and reusable sklearn preprocessing pipelines."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, StandardScaler

import config as C

CLUSTER_CATS = ["Color_Group", "Pattern_Group", "Size_Group", "Sleeve_Type", "Collar_Type", "Fit"]
CLUSTER_NUM = ["Price_Band_Idx"]


def _band_label(idx: int) -> str:
    lo = int(idx) * C.PRICE_BAND_WIDTH
    return f"{lo}-{lo + C.PRICE_BAND_WIDTH - 1}"


def add_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw attributes: colour -> warm/cool/neutral, pattern -> 4 groups, size -> S/M/L/XL,
    price -> INR 500 bands."""
    out = df.copy()
    out["Color_Group"] = out["Color"].map(C.COLOR_TO_GROUP)
    out["Pattern_Group"] = out["Pattern"].map(C.PATTERN_TO_GROUP)
    out["Size_Group"] = out["Size"].astype(int).map(C.SIZE_TO_GROUP)
    out["Price_Band_Idx"] = (out["Price"].astype(float) // C.PRICE_BAND_WIDTH).astype(int)
    out["Price_Band"] = out["Price_Band_Idx"].map(_band_label)
    return out


def make_cluster_encoder() -> Pipeline:
    """Raw garment attributes -> aggregated groups -> one-hot (0/1) + min-max price band.
    Every encoded column lies in [0, 1], so the numeric price band cannot dominate K-Means."""
    ct = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CLUSTER_CATS),
        ("num", MinMaxScaler(), CLUSTER_NUM),
    ])
    return Pipeline([("groups", FunctionTransformer(add_groups, validate=False)), ("encode", ct)])


def _size_to_str(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Size"] = out["Size"].astype(int).astype(str)
    return out


def purchase_feature_names(scenario: str) -> list[str]:
    """Scenario A: attributes only (cold start). Scenario B: attributes + prior Trial_Count."""
    num = ["Price", "Trend_Index"] + (["Trial_Count"] if scenario == "B" else [])
    return C.CAT_FEATURES + num


def make_purchase_preprocessor(scenario: str) -> Pipeline:
    num = [("price", StandardScaler(), ["Price"]), ("trend", StandardScaler(), ["Trend_Index"])]
    if scenario == "B":
        num.append(("trials", Pipeline([("log", FunctionTransformer(np.log1p, validate=False,
                                                                    feature_names_out="one-to-one")),
                                        ("scale", StandardScaler())]), ["Trial_Count"]))
    ct = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                             C.CAT_FEATURES)] + num)
    return Pipeline([("prep", FunctionTransformer(_size_to_str, validate=False)), ("ct", ct)])


def build_customer_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Customer preference profile = share of tried garments in each aggregated group."""
    g = add_groups(df)
    parts = [g.groupby("Customer_ID").size().rename("n_trials")]
    for col in ["Color_Group", "Pattern_Group", "Size_Group", "Fit", "Collar_Type", "Sleeve_Type"]:
        parts.append(pd.crosstab(g["Customer_ID"], g[col], normalize="index").add_prefix(f"{col}_"))
    parts.append(g.groupby("Customer_ID")["Price_Band_Idx"].mean().rename("Mean_Price_Band_Idx"))
    parts.append(g.groupby("Customer_ID")["Price"].mean().round(0).rename("Mean_Price"))
    parts.append(g.groupby("Customer_ID")[C.TARGET].mean().rename("Purchase_Rate"))
    return pd.concat(parts, axis=1).reset_index()
