"""Inference: purchase likelihood + cluster assignment + inventory interpretation for a garment."""
from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

import config as C
from src import trends as T
from src.feature_engineering import purchase_feature_names
from src.inventory_insights import recommend


class PurchasePredictor:
    def __init__(self, models_dir=C.MODELS_DIR):
        models_dir = Path(models_dir)
        bundle = joblib.load(models_dir / "purchase_model.pkl")
        self.pipelines, self.meta = bundle["pipelines"], bundle["meta"]
        clus = joblib.load(models_dir / "clustering_model.pkl")
        self.weights, self.profiles = clus["weights"], clus["profiles"]
        self.tree = joblib.load(models_dir / "decision_tree_model.pkl")
        self.encoder = joblib.load(models_dir / "preprocessing_pipeline.pkl")
        if self.meta.get("sklearn_version") != sklearn.__version__:
            warnings.warn("Models were saved with scikit-learn "
                          f"{self.meta.get('sklearn_version')} but {sklearn.__version__} is installed; "
                          "re-run `python main.py` if loading or predictions misbehave.")

    @staticmethod
    def models_available(models_dir=C.MODELS_DIR) -> bool:
        return all((models_dir / f).exists() for f in
                   ["purchase_model.pkl", "clustering_model.pkl", "decision_tree_model.pkl",
                    "preprocessing_pipeline.pkl"])

    # ---------------------------------------------------------------------------------------
    @staticmethod
    def validate(size, sleeve, color, collar, fit, pattern, price, trial_count) -> list[str]:
        """Raise ValueError for impossible inputs; return soft warnings (e.g. outside training range)."""
        for name, val, allowed in [("Size", size, C.SIZES), ("Sleeve_Type", sleeve, C.SLEEVES),
                                   ("Color", color, C.COLORS), ("Collar_Type", collar, C.COLLARS),
                                   ("Fit", fit, C.FITS), ("Pattern", pattern, C.PATTERNS)]:
            if val not in allowed:
                raise ValueError(f"{name}={val!r} is not one of {allowed}")
        try:
            price = float(price)
        except (TypeError, ValueError):
            raise ValueError("Price must be a number")
        if not np.isfinite(price) or price <= 0:
            raise ValueError("Price must be a positive number")
        if trial_count is not None and (int(trial_count) != trial_count or trial_count < 0):
            raise ValueError("Trial_Count must be a non-negative whole number (or None for a new garment)")
        warns = []
        if not C.PRICE_MIN <= price <= C.PRICE_MAX:
            warns.append(f"Price INR {price:,.0f} is outside the training range "
                         f"(INR {C.PRICE_MIN:,}-{C.PRICE_MAX:,}); the estimate is an extrapolation.")
        if trial_count is not None and trial_count > 60:
            warns.append("Trial_Count is far above typical training values; treat the estimate with caution.")
        return warns

    def _frame(self, scenario, size, sleeve, color, collar, fit, pattern, price, trial_count, trend):
        row = {"Size": int(size), "Sleeve_Type": sleeve, "Color": color, "Collar_Type": collar, "Fit": fit,
               "Pattern": pattern, "Price": float(price), "Trend_Index": float(trend)}
        if scenario == "B":
            row["Trial_Count"] = int(trial_count)
        return pd.DataFrame([row])[purchase_feature_names(scenario)]

    def _explain(self, scenario, pipe, base_row: pd.DataFrame, p0: float) -> list[dict]:
        """Model-agnostic what-if: change one input to each alternative, compare with the actual estimate."""
        alts = {"Size": C.SIZES, "Sleeve_Type": C.SLEEVES, "Color": C.COLORS, "Collar_Type": C.COLLARS,
                "Fit": C.FITS, "Pattern": C.PATTERNS,
                "Price": list(np.linspace(C.PRICE_MIN, C.PRICE_MAX, 12).round(0))}
        if scenario == "B":
            alts["Trial_Count"] = [0, 1, 2, 4, 8, 15]
        out = []
        for feat, values in alts.items():
            frames = []
            for v in values:
                r = base_row.copy()
                r[feat] = v
                frames.append(r)
            probs = pipe.predict_proba(pd.concat(frames, ignore_index=True))[:, 1]
            out.append({"Feature": feat, "Value": base_row[feat].iloc[0],
                        "Effect_pp": round(float((p0 - probs.mean()) * 100), 1)})
        return sorted(out, key=lambda d: abs(d["Effect_pp"]), reverse=True)

    # ---------------------------------------------------------------------------------------
    def predict(self, size, sleeve, color, collar, fit, pattern, price,
                trial_count=None, trend_index=None, explain: bool = True) -> dict:
        """trial_count=None -> Scenario A (new garment). trial_count=int -> Scenario B."""
        warns = self.validate(size, sleeve, color, collar, fit, pattern, price, trial_count)
        scenario = "A" if trial_count is None else "B"
        trend_src = "user-supplied"
        if trend_index is None:
            trend_index, trend_src = T.default_trend_index(pattern, fit)
        pipe = self.pipelines[scenario]
        X = self._frame(scenario, size, sleeve, color, collar, fit, pattern, price, trial_count, trend_index)
        prob = float(pipe.predict_proba(X)[0, 1])

        attrs = pd.DataFrame([{"Size": int(size), "Sleeve_Type": sleeve, "Color": color, "Collar_Type": collar,
                               "Fit": fit, "Pattern": pattern, "Price": float(price)}])[C.ATTR_COLS]
        cid = int(self.tree.predict(self.encoder.transform(attrs))[0])
        wrow = self.weights.loc[self.weights["Cluster"] == cid].iloc[0]
        k = int(self.meta["n_clusters"])
        rec = recommend(prob, self.meta["base_rate"], float(wrow["Weight_Pct"]), k, scenario, trial_count)
        return {
            "scenario": scenario,
            "scenario_name": "New garment (cold start)" if scenario == "A" else "Garment with fitting-room history",
            "model": self.meta["primary"][scenario]["model"],
            "probability": prob, "likelihood_pct": round(prob * 100, 1),
            "predicted_class": int(prob >= self.meta["threshold"]),
            "predicted_class_label": "Likely to be purchased" if prob >= self.meta["threshold"]
            else "Less likely to be purchased",
            "base_rate": self.meta["base_rate"],
            "cluster": cid, "cluster_label": str(wrow["Label"]), "cluster_weight_pct": float(wrow["Weight_Pct"]),
            "cluster_records": int(wrow["Records"]),
            "trend_index": float(trend_index), "trend_source": trend_src,
            "recommendation": rec, "warnings": warns,
            "explanation": self._explain(scenario, pipe, X, prob) if explain else [],
        }
