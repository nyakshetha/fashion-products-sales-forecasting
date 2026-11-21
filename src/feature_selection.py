"""Feature-importance investigation with Decision Tree, Random Forest and Lasso.

These three methods are used for feature SCREENING only - they are not the final prediction models.
Target for screening: Purchased (the only supervised label in the synthetic data). Importances of the
one-hot columns are summed back to the original attribute and normalised to sum to 1 per method.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import config as C
from src.feature_engineering import make_purchase_preprocessor, purchase_feature_names

ATTRS = C.CAT_FEATURES + ["Price", "Trend_Index", "Trial_Count"]


def _aggregate(names, values) -> pd.Series:
    agg = {a: 0.0 for a in ATTRS}
    for n, v in zip(names, values):
        n = n.split("__", 1)[1]
        for a in C.CAT_FEATURES:
            if n.startswith(a + "_"):
                agg[a] += abs(v)
                break
        else:
            key = {"Price": "Price", "Trend_Index": "Trend_Index", "Trial_Count": "Trial_Count"}[n]
            agg[key] += abs(v)
    s = pd.Series(agg)
    return s / s.sum() if s.sum() > 0 else s


def run_feature_selection(df: pd.DataFrame, seed: int = C.SEED, n_trees: int = 300) -> pd.DataFrame:
    prep = make_purchase_preprocessor("B")
    feats = purchase_feature_names("B")
    X = prep.fit_transform(df[feats])
    names = prep.named_steps["ct"].get_feature_names_out()
    y = df[C.TARGET].to_numpy()

    dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=30, random_state=seed).fit(X, y)
    rf = RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=10, random_state=seed, n_jobs=-1).fit(X, y)
    Xs = StandardScaler().fit_transform(X)
    lasso = LassoCV(cv=5, random_state=seed, max_iter=5000).fit(Xs, y)   # linear-probability screening

    out = pd.DataFrame({
        "Decision Tree": _aggregate(names, dt.feature_importances_),
        "Random Forest": _aggregate(names, rf.feature_importances_),
        "Lasso (|coef|)": _aggregate(names, lasso.coef_),
    })
    out["Mean importance"] = out.mean(axis=1)
    out = out.sort_values("Mean importance", ascending=False)
    out.index.name = "Feature"
    return out
