"""K-Means customer-preference clustering (research step 4): elbow + silhouette for K = 2..10, fit K = 6."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import config as C
from src.feature_engineering import add_groups


def evaluate_k_range(X: np.ndarray, ks=C.K_RANGE, seed: int = C.SEED) -> pd.DataFrame:
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        rows.append({"k": k, "inertia": float(km.inertia_),
                     "silhouette": float(silhouette_score(X, km.labels_))})
    return pd.DataFrame(rows)


def fit_kmeans(X: np.ndarray, k: int = C.N_CLUSTERS, seed: int = C.SEED):
    """Fit K-Means and renumber clusters 0..k-1 from largest to smallest (deterministic ordering)."""
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)
    order = np.argsort(-np.bincount(km.labels_, minlength=k), kind="stable")
    remap = {int(old): new for new, old in enumerate(order)}
    km.cluster_centers_ = km.cluster_centers_[order]
    labels = np.array([remap[int(l)] for l in km.labels_])
    return km, labels


def cluster_profiles(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Describe each cluster by its dominant aggregated attributes."""
    g = add_groups(df).assign(Cluster=labels)
    rows = []
    for c, sub in g.groupby("Cluster"):
        def top(col):
            vc = sub[col].value_counts(normalize=True)
            return vc.index[0], float(vc.iloc[0])
        color, color_s = top("Color_Group")
        pat, pat_s = top("Pattern_Group")
        size, size_s = top("Size_Group")
        fit, _ = top("Fit")
        collar, _ = top("Collar_Type")
        sleeve, _ = top("Sleeve_Type")
        mean_price = float(sub["Price"].mean())
        # label by the attribute values that make this cluster DIFFERENT from the whole data (highest lift)
        cands = []
        for col in ["Color_Group", "Pattern_Group", "Size_Group", "Fit", "Collar_Type", "Sleeve_Type"]:
            share_in = sub[col].value_counts(normalize=True)
            share_all = g[col].value_counts(normalize=True)
            for val, s_in in share_in.items():
                if s_in >= 0.25:
                    cands.append((s_in / share_all[val], col, str(val)))
        picked, used = [], set()
        for lift, col, val in sorted(cands, reverse=True):
            if col not in used and len(picked) < 3:
                picked.append(val.title() if col != "Size_Group" else f"Size {val}")
                used.add(col)
        label = f"C{c + 1}: {' / '.join(picked)} / ~INR {int(round(mean_price, -1)):,}"
        rows.append({"Cluster": int(c), "Label": label, "Dominant_Color_Group": color,
                     "Color_Share": round(color_s, 3), "Dominant_Pattern_Group": pat,
                     "Pattern_Share": round(pat_s, 3), "Dominant_Size_Group": size,
                     "Size_Share": round(size_s, 3), "Dominant_Fit": fit, "Dominant_Collar": collar,
                     "Dominant_Sleeve": sleeve, "Mean_Price": round(mean_price, 0),
                     "Purchase_Rate": round(float(sub[C.TARGET].mean()), 3) if C.TARGET in sub else np.nan})
    return pd.DataFrame(rows)
