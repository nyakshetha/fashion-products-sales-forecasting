"""Decision-support text for retailers. These are suggestions, NOT guaranteed outcomes or stock quantities."""
from __future__ import annotations

import config as C
from src.cluster_weighting import demand_potential

ASSUMPTIONS = [
    "Purchase likelihood is the model's estimate that ONE fitting-room trial of this garment ends in a purchase.",
    "Cluster weight is the share of observed fitting-room trials in the garment's preference cluster - "
    "it measures preference, not purchase probability.",
    "No stock quantity is suggested: that needs expected store traffic, current stock, replenishment lead time "
    "and cost constraints, which this model does not have.",
    "Models are trained on SYNTHETIC data; treat all outputs as a demonstration, not a validated forecast.",
]


def likelihood_tier(prob: float, base_rate: float, margin: float = C.TIER_MARGIN) -> str:
    if prob >= base_rate + margin:
        return "High"
    if prob <= base_rate - margin:
        return "Low"
    return "Moderate"


def recommend(prob: float, base_rate: float, cluster_weight_pct: float, n_clusters: int,
              scenario: str, trial_count: int | None = None) -> dict:
    tier = likelihood_tier(prob, base_rate)
    rel, d_tier = demand_potential(cluster_weight_pct, n_clusters, C.DEMAND_HIGH, C.DEMAND_LOW)
    headline = {
        "High": "Stronger purchase conversion potential - consider prioritising stock allocation.",
        "Moderate": "Monitor customer response and trial-to-purchase conversion before committing more stock.",
        "Low": "Review product attributes, pricing or initial stock allocation before scaling up.",
    }[tier]
    actions = [headline]
    if d_tier == "High" and tier != "Low":
        actions.append("The garment sits in a large preference segment: expect many fitting-room trials; "
                       "make sure core sizes are available.")
    elif d_tier == "Low" and tier == "High":
        actions.append("Likely to convert well but the segment is small: a niche item - keep stock lean and "
                       "replenish on evidence.")
    elif d_tier == "High" and tier == "Low":
        actions.append("Many shoppers will try it but few are predicted to buy: check price and fit before "
                       "allocating deep stock.")
    elif d_tier == "Low" and tier == "Low":
        actions.append("Small segment and low predicted conversion: consider a limited initial allocation.")
    if scenario == "B" and trial_count is not None and trial_count > 0:
        actions.append(f"{trial_count} earlier trial(s) were used as an input; refresh the estimate as new "
                       "trials and purchases come in.")
    if scenario == "A":
        actions.append("Cold-start estimate (no trial history): re-score after the first fitting-room trials.")
    return {"likelihood_tier": tier, "demand_tier": d_tier, "relative_demand_index": round(rel, 3),
            "headline": headline, "actions": actions, "assumptions": ASSUMPTIONS}
