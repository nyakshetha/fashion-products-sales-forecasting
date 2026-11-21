"""Retail-facing product scoring used by the dashboard.

Turns model output into things a store manager acts on: purchase-likelihood tier (High / Medium / Low),
demand level (fitting-room traffic), search-trend direction (Google Trends) and a recommended action
(Stock up / Promote / Monitor / Review price-fit / Limit stock).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from src import trends as T
from src.feature_engineering import purchase_feature_names
from src.inventory_insights import likelihood_tier

TIERS = ["High", "Medium", "Low"]
ACTIONS = ["Stock up", "Promote", "Monitor", "Review price / fit", "Limit stock"]
MIN_TRIALS_FOR_HISTORY = 5           # fewer trials -> score from attributes only (cold-start model)
TREND_THRESHOLD_PCT = 8.0


def tier_label(prob: float, base_rate: float) -> str:
    t = likelihood_tier(prob, base_rate)
    return "Medium" if t == "Moderate" else t


# ------------------------------------------------------------------ search trends
def trend_changes(table: pd.DataFrame) -> dict[str, float]:
    """% change of latest-2-week interest vs the previous 4 weeks, per keyword series."""
    out = {}
    for key in C.TREND_KEYWORDS:
        s = table[key].astype(float).to_numpy()
        recent = float(s[-2:].mean()) if len(s) >= 2 else (float(s.mean()) if len(s) > 0 else 50.0)
        prior = float(s[-6:-2].mean()) if len(s) >= 6 else (float(s[:-2].mean()) if len(s) > 2 else recent)
        if prior <= 1e-3:
            change = 0.0 if recent <= 1e-3 else min(100.0, (recent - prior) * 10.0)
        else:
            change = float((recent - prior) / prior * 100)
        out[key] = float(np.clip(change, -100.0, 300.0))
    return out


def direction(change_pct: float) -> str:
    return "Rising" if change_pct >= TREND_THRESHOLD_PCT else "Falling" if change_pct <= -TREND_THRESHOLD_PCT else "Stable"


def product_trend(pattern: str, fit: str, changes: dict[str, float]) -> tuple[str, float]:
    ch = float(np.mean([changes[f"pattern:{C.PATTERN_TO_GROUP[pattern]}"], changes[f"fit:{fit}"]]))
    return direction(ch), ch


# ------------------------------------------------------------------ actions
def action_for(tier: str, demand: str, trend: str) -> tuple[str, str]:
    if tier == "High":
        if demand == "High" or (demand == "Medium" and trend == "Rising"):
            why = "Strong purchase likelihood and healthy fitting-room traffic"
            return "Stock up", why + (", with search interest rising." if trend == "Rising" else ".")
        return "Promote", "Converts well when tried, but few shoppers are trying it - give it more visibility."
    if tier == "Low":
        if demand in ("High", "Medium"):
            return "Review price / fit", "Shoppers try it but are unlikely to buy - check price, sizing and fit."
        return "Limit stock", "Low purchase likelihood and low fitting-room interest - keep stock lean."
    if demand == "Low" and trend == "Rising":
        return "Promote", "Average likelihood, but search interest is rising - a push could lift trials."
    return "Monitor", "Average purchase likelihood - watch trial-to-purchase conversion."


# ------------------------------------------------------------------ scorecard
def build_scorecard(df: pd.DataFrame, predictor, table: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per product (Garment_ID) with likelihood, tier, demand, trend, segment and action."""
    table = T.load_trends() if table is None else table
    changes = trend_changes(table)
    base_rate = float(predictor.meta["base_rate"])

    d = df.copy()
    d["Trials_Total"] = d.groupby("Garment_ID")["Purchased"].transform("size")
    latest = {(p, f): T.trend_index(C.PATTERN_TO_GROUP[p], f, table)
              for p in C.PATTERNS for f in C.FITS}                       # current search interest per product type
    d["Trend_Index"] = [latest[(p, f)] for p, f in zip(d["Pattern"], d["Fit"])]
    d_b = d.assign(Trial_Count=d["Trials_Total"])
    d["pA"] = predictor.pipelines["A"].predict_proba(d[purchase_feature_names("A")])[:, 1]
    d["pB"] = predictor.pipelines["B"].predict_proba(d_b[purchase_feature_names("B")])[:, 1]
    d["Cluster"] = predictor.tree.predict(predictor.encoder.transform(d[C.ATTR_COLS]))

    sc = d.groupby("Garment_ID").agg(
        Color=("Color", "first"), Pattern=("Pattern", "first"), Fit=("Fit", "first"),
        Collar=("Collar_Type", "first"), Sleeve=("Sleeve_Type", "first"), Price=("Price", "first"),
        Trials=("Purchased", "size"), Purchases=("Purchased", "sum"),
        pA=("pA", "mean"), pB=("pB", "mean"),
        Cluster=("Cluster", lambda s: int(s.mode().iloc[0]))).reset_index()
    sc = sc.rename(columns={"Garment_ID": "Product ID"})
    sc["Conversion_Pct"] = (sc["Purchases"] / sc["Trials"] * 100).round(1)
    use_hist = sc["Trials"] >= MIN_TRIALS_FOR_HISTORY
    prob = np.where(use_hist, sc["pB"], sc["pA"])
    sc["Likelihood_Pct"] = (prob * 100).round(1)
    sc["Estimate_Basis"] = np.where(use_hist, "Trial history", "Product attributes")
    sc["Likelihood"] = [tier_label(p, base_rate) for p in prob]

    rank = sc["Trials"].rank(pct=True, method="average")
    sc["Demand"] = np.where(rank >= 0.75, "High", np.where(rank <= 0.25, "Low", "Medium"))

    w = predictor.weights.set_index("Cluster")["Weight_Pct"]
    sc["Segment"] = "Segment " + (sc["Cluster"] + 1).astype(str)
    sc["Segment_Share_Pct"] = sc["Cluster"].map(w).round(1)

    tr = [product_trend(p, f, changes) for p, f in zip(sc["Pattern"], sc["Fit"])]
    sc["Search_Trend"] = [t[0] for t in tr]
    sc["Search_Change_Pct"] = [round(t[1], 1) for t in tr]
    acts = [action_for(l, dm, st) for l, dm, st in zip(sc["Likelihood"], sc["Demand"], sc["Search_Trend"])]
    sc["Action"] = [a[0] for a in acts]
    sc["Reason"] = [a[1] for a in acts]
    sc["Product"] = (sc["Color"] + " " + sc["Pattern"].str.lower() + " shirt, " + sc["Fit"].str.lower())
    cols = ["Product ID", "Product", "Color", "Pattern", "Fit", "Collar", "Sleeve", "Price", "Trials",
            "Purchases", "Conversion_Pct", "Likelihood_Pct", "Likelihood", "Estimate_Basis", "Demand",
            "Segment", "Segment_Share_Pct", "Search_Trend", "Search_Change_Pct", "Action", "Reason"]
    return sc[cols].sort_values("Likelihood_Pct", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ activity / popularity
def weekly_activity(trials: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(trials["Trial_Timestamp"])
    week = ts.dt.normalize() - pd.to_timedelta(ts.dt.dayofweek, unit="D")
    w = trials.assign(Week=week).groupby("Week").agg(Trials=("Purchased", "size"),
                                                      Purchases=("Purchased", "sum")).reset_index()
    w["Conversion_Pct"] = (w["Purchases"] / w["Trials"] * 100).round(1)
    return w


def period_comparison(trials: pd.DataFrame, days: int = 7) -> dict:
    """Last `days` of activity vs the `days` before it (relative to the newest record)."""
    ts = pd.to_datetime(trials["Trial_Timestamp"])
    end = ts.max()
    cur = trials[ts > end - pd.Timedelta(days=days)]
    prev = trials[(ts <= end - pd.Timedelta(days=days)) & (ts > end - pd.Timedelta(days=2 * days))]
    def conv(x):
        return float(x["Purchased"].mean() * 100) if len(x) else float("nan")
    return {"trials": len(cur), "trials_prev": len(prev), "conv": conv(cur), "conv_prev": conv(prev),
            "purchases": int(cur["Purchased"].sum()), "purchases_prev": int(prev["Purchased"].sum())}


def attribute_summary(trials: pd.DataFrame, col: str, order=None) -> pd.DataFrame:
    g = trials.groupby(col).agg(Trials=("Purchased", "size"), Purchases=("Purchased", "sum")).reset_index()
    g["Share_Pct"] = (g["Trials"] / g["Trials"].sum() * 100).round(1)
    g["Conversion_Pct"] = (g["Purchases"] / g["Trials"] * 100).round(1)
    if order is not None:
        g[col] = pd.Categorical(g[col], categories=[o for o in order if o in set(g[col])], ordered=True)
        g = g.sort_values(col)
    else:
        g = g.sort_values("Trials", ascending=False)
    return g.reset_index(drop=True)
