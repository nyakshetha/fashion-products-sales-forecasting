"""Cluster weights and cluster-based relative demand (research steps 5 and 7).

Cluster Weight (%) = records in cluster / total records x 100.
This is the share of OBSERVED FITTING-ROOM TRIALS, i.e. preference - it is not a purchase probability.
Relative Demand Index = weight / (100 / K): 1.0 means an average-sized preference segment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_cluster_weights(labels: np.ndarray, k: int) -> pd.DataFrame:
    counts = np.bincount(labels, minlength=k)
    total = counts.sum()
    out = pd.DataFrame({"Cluster": np.arange(k), "Records": counts})
    out["Weight_Pct"] = (out["Records"] / total * 100).round(2)
    out["Relative_Demand_Index"] = (out["Weight_Pct"] / (100.0 / k)).round(3)
    return out


def demand_potential(weight_pct: float, k: int, high: float, low: float) -> tuple[float, str]:
    rel = weight_pct / (100.0 / k)
    return rel, ("High" if rel >= high else "Low" if rel <= low else "Average")
