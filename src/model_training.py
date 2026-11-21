"""Training: (1) research pipeline - K-Means, cluster weights, Decision Tree; (2) purchase-likelihood models.

Purchase-likelihood models are the supervised EXTENSION on the synthetic `Purchased` label; they are not
part of the original cluster-weight methodology. Two scenarios are trained:
  A - new garment (attributes + trend index, no trial history)   B - attributes + prior Trial_Count.
"""
from __future__ import annotations

import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text

import config as C
from src import clustering, evaluation
from src.cluster_weighting import compute_cluster_weights
from src.feature_engineering import make_cluster_encoder, make_purchase_preprocessor, purchase_feature_names

def _probe_xgboost():
    """Return XGBClassifier only if it imports AND works inside an sklearn pipeline (clone/fit/predict)."""
    try:
        from sklearn.base import clone
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBClassifier as _X
        X = np.random.default_rng(0).random((40, 3))
        y = np.array([0, 1] * 20)
        Pipeline([("s", StandardScaler()), ("c", clone(_X(n_estimators=5)))]).fit(X, y).predict_proba(X)
        return _X
    except Exception as exc:
        print(f"WARNING: XGBoost unavailable ({type(exc).__name__}: {str(exc)[:120]}). "
              "A stand-in gradient-boosting model will be used - run `pip install -r requirements.txt`.",
              file=sys.stderr)
        return None


XGBClassifier = _probe_xgboost()
HAS_XGB = XGBClassifier is not None

GB_NAME = "XGBoost" if HAS_XGB else "HistGradientBoosting (XGBoost stand-in)"


def candidate_models(fast: bool = False, seed: int = C.SEED) -> dict:
    n = 60 if fast else 250
    if HAS_XGB:
        gb = XGBClassifier(n_estimators=n, max_depth=3, learning_rate=0.05, subsample=0.8,
                           colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
                           eval_metric="logloss", random_state=seed, n_jobs=2)
    else:
        gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=n,
                                            l2_regularization=2.0, random_state=seed)
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=30, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=60 if fast else 300, min_samples_leaf=10,
                                                random_state=seed, n_jobs=-1),
        GB_NAME: gb,
    }


def build_pipeline(scenario: str, estimator) -> Pipeline:
    return Pipeline([("pre", make_purchase_preprocessor(scenario)), ("clf", estimator)])


# --------------------------------------------------------------------------------------------
# Research pipeline
# --------------------------------------------------------------------------------------------
def train_cluster_stack(df: pd.DataFrame, k: int = C.N_CLUSTERS, seed: int = C.SEED) -> dict:
    enc = make_cluster_encoder()
    X = enc.fit_transform(df[C.ATTR_COLS])
    k_eval = clustering.evaluate_k_range(X, seed=seed)
    km, labels = clustering.fit_kmeans(X, k, seed)
    weights = compute_cluster_weights(labels, k)
    profiles = clustering.cluster_profiles(df, labels)
    weights = weights.merge(profiles[["Cluster", "Label", "Purchase_Rate"]], on="Cluster")

    # Decision Tree: classify garments into the learned preference clusters
    Xtr, Xte, ytr, yte = train_test_split(X, labels, test_size=0.2, stratify=labels, random_state=seed)
    probe = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=seed).fit(Xtr, ytr)
    pred = probe.predict(Xte)
    dt_metrics = {"holdout_accuracy": float(accuracy_score(yte, pred)),
                  "holdout_macro_f1": float(f1_score(yte, pred, average="macro")),
                  "holdout_rows": int(len(yte))}
    tree = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=seed).fit(X, labels)
    names = enc.named_steps["encode"].get_feature_names_out()
    dt_metrics["depth"] = int(tree.get_depth())
    dt_metrics["leaves"] = int(tree.get_n_leaves())
    rules = export_text(tree, feature_names=[n.split("__", 1)[1] for n in names], max_depth=4)
    return {"encoder": enc, "kmeans": km, "labels": labels, "weights": weights, "profiles": profiles,
            "k_eval": k_eval, "tree": tree, "tree_metrics": dt_metrics, "tree_rules": rules,
            "inertia": float(km.inertia_),
            "silhouette": float(k_eval.loc[k_eval["k"] == k, "silhouette"].iloc[0]) if k in set(k_eval["k"]) else None,
            "X": X}


# --------------------------------------------------------------------------------------------
# Purchase-likelihood extension
# --------------------------------------------------------------------------------------------
def train_purchase_models(df: pd.DataFrame, fast: bool = False, seed: int = C.SEED) -> dict:
    y = df[C.TARGET].to_numpy()
    rows, holdout = [], {}
    primary, pipelines = {}, {}
    for scenario in ("A", "B"):
        feats = purchase_feature_names(scenario)
        holdout[scenario] = {}
        for split_name, group_col in (("customer_heldout", "Customer_ID"), ("garment_heldout", "Garment_ID")):
            tr, te = evaluation.grouped_split(df, group_col, seed=seed)
            for name, est in candidate_models(fast, seed).items():
                from sklearn.base import clone
                pipe = build_pipeline(scenario, clone(est)).fit(df.iloc[tr][feats], y[tr])
                prob = pipe.predict_proba(df.iloc[te][feats])[:, 1]
                m = evaluation.classification_metrics(y[te], prob)
                rows.append({"scenario": scenario, "split": split_name, "model": name, **m})
                if split_name == "customer_heldout":
                    holdout[scenario][name] = (y[te], prob)
        met = pd.DataFrame([r for r in rows if r["scenario"] == scenario and r["split"] == "customer_heldout"])
        best = met.sort_values("roc_auc", ascending=False).iloc[0]
        chosen = GB_NAME if GB_NAME in set(met["model"]) else best["model"]
        gap = float(best["roc_auc"] - met.loc[met["model"] == chosen, "roc_auc"].iloc[0])
        primary[scenario] = {
            "model": chosen, "best_by_auc": best["model"], "auc_gap_to_best": round(gap, 4),
            "note": (f"{chosen} is the project's designated primary model. "
                     + ("It also has the highest customer-held-out ROC-AUC." if gap <= 1e-9 else
                        f"On this synthetic data '{best['model']}' scores higher by {gap:.3f} ROC-AUC; "
                        "see the comparison table."))}
        final = build_pipeline(scenario, clone(candidate_models(fast, seed)[chosen])).fit(df[feats], y)
        pipelines[scenario] = final
    return {"metrics": pd.DataFrame(rows), "holdout": holdout, "primary": primary, "pipelines": pipelines,
            "base_rate": float(y.mean())}


# --------------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------------
def save_artifacts(cluster: dict, purchase: dict, models_dir=C.MODELS_DIR) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    meta = {"base_rate": purchase["base_rate"], "primary": purchase["primary"],
            "threshold": C.CLASS_THRESHOLD, "tier_margin": C.TIER_MARGIN,
            "uses_xgboost": HAS_XGB, "sklearn_version": sklearn.__version__,
            "python_version": sys.version.split()[0], "n_clusters": int(len(cluster["weights"]))}
    joblib.dump({"pipelines": purchase["pipelines"], "meta": meta}, models_dir / "purchase_model.pkl")
    joblib.dump({"kmeans": cluster["kmeans"], "weights": cluster["weights"], "profiles": cluster["profiles"]},
                models_dir / "clustering_model.pkl")
    joblib.dump(cluster["tree"], models_dir / "decision_tree_model.pkl")
    joblib.dump(cluster["encoder"], models_dir / "preprocessing_pipeline.pkl")
