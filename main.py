"""Train everything and write models, metrics, plots and reports.

    python main.py                 # full run
    python main.py --fast          # smaller models (quick check)
    python main.py --live-trends   # also try to download real Google Trends data with PyTrends
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

import config as C
from src import evaluation, feature_selection, model_training, preprocessing, trends, visualization
from src.data_loader import load_or_generate
from src.feature_engineering import build_customer_profiles


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--live-trends", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    for d in (C.PROCESSED_DIR, C.EXTERNAL_DIR, C.MODELS_DIR, C.PLOTS_DIR, C.METRICS_DIR, C.REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading dataset ...")
    df = load_or_generate()
    print(f"      {len(df):,} trials, {df['Customer_ID'].nunique():,} customers, "
          f"{df['Garment_ID'].nunique()} garments, conversion {df[C.TARGET].mean():.1%}")

    print("[2/7] Data quality checks ...")
    report = preprocessing.preprocessing_report(df)
    (C.METRICS_DIR / "data_quality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["schema_problems"]:
        sys.exit(f"Schema problems: {report['schema_problems']}")
    build_customer_profiles(df).to_csv(C.PROCESSED_DIR / "customer_profiles.csv", index=False, encoding="utf-8")

    print("[3/7] Google Trends signal ...")
    tt = trends.load_trends(live=args.live_trends)
    print(f"      trend source for inference defaults: {tt['source'].iloc[0]}")

    print("[4/7] Feature selection (Decision Tree / Random Forest / Lasso) ...")
    imp = feature_selection.run_feature_selection(df)
    imp.to_csv(C.METRICS_DIR / "feature_importance.csv", encoding="utf-8")

    print("[5/7] Research pipeline: K-Means (K=2..10, final K=6), cluster weights, Decision Tree ...")
    stack = model_training.train_cluster_stack(df)
    stack["k_eval"].to_csv(C.METRICS_DIR / "kmeans_k_evaluation.csv", index=False)
    stack["weights"].to_csv(C.METRICS_DIR / "cluster_weights.csv", index=False, encoding="utf-8")
    stack["profiles"].to_csv(C.METRICS_DIR / "cluster_profiles.csv", index=False, encoding="utf-8")
    (C.REPORTS_DIR / "decision_tree_rules.txt").write_text(stack["tree_rules"], encoding="utf-8")
    best_k = int(stack["k_eval"].sort_values("silhouette", ascending=False).iloc[0]["k"])
    cluster_metrics = {"k": C.N_CLUSTERS, "inertia": stack["inertia"], "silhouette": stack["silhouette"],
                       "k_with_highest_silhouette": best_k, "decision_tree": stack["tree_metrics"]}
    (C.METRICS_DIR / "cluster_metrics.json").write_text(json.dumps(cluster_metrics, indent=2), encoding="utf-8")
    print(f"      silhouette(K=6)={stack['silhouette']:.3f}; K with highest silhouette = {best_k}; "
          f"DT holdout accuracy={stack['tree_metrics']['holdout_accuracy']:.3f}")

    print(f"[6/7] Purchase-likelihood models (primary: {model_training.GB_NAME}) ...")
    purchase = model_training.train_purchase_models(df, fast=args.fast)
    purchase["metrics"].to_csv(C.METRICS_DIR / "purchase_model_metrics.csv", index=False)
    summary = {"base_rate": purchase["base_rate"], "primary": purchase["primary"],
               "uses_xgboost": model_training.HAS_XGB, "trend_source": str(tt["source"].iloc[0])}
    reports = {}
    for scen in ("A", "B"):
        name = purchase["primary"][scen]["model"]
        yt, yp = purchase["holdout"][scen][name]
        reports[scen] = evaluation.report_text(yt, yp)
        (C.REPORTS_DIR / f"classification_report_{scen}.txt").write_text(
            f"Scenario {scen} - {name} - customer-held-out test set\n\n{reports[scen]}", encoding="utf-8")
    (C.METRICS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    model_training.save_artifacts(stack, purchase)

    if not args.no_plots:
        print("[7/7] Plots ...")
        visualization.eda_plots(df)
        visualization.feature_importance_plot(imp)
        visualization.cluster_plots(stack)
        visualization.model_plots(purchase)
    write_results_report(df, imp, stack, purchase, cluster_metrics, reports)
    print("Done. Models -> models/, metrics -> outputs/metrics/, plots -> outputs/plots/, "
          "report -> outputs/reports/results_report.md")


def write_results_report(df, imp, stack, purchase, cm, reports) -> None:
    m = purchase["metrics"]
    cols = ["scenario", "split", "model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]
    L = ["# Results report (auto-generated by main.py)", "",
         "All numbers below come from the SYNTHETIC dataset and were produced by this run. They show that the "
         "pipeline works; they do not measure real-world predictive performance.", "",
         f"XGBoost installed for this run: **{model_training.HAS_XGB}**"
         + ("" if model_training.HAS_XGB else
            " (HistGradientBoosting was used as a stand-in under the name shown below - install `xgboost` and "
            "re-run `python main.py` to train the real XGBoost model)."), "",
         "## 1. Purchase-likelihood prediction (supervised extension)", "",
         f"Base purchase rate: {purchase['base_rate']:.1%}. Accuracy of always predicting 'not purchased' "
         f"would be about {1 - purchase['base_rate']:.1%}; compare models with ROC-AUC, PR-AUC and Brier score "
         "(lower is better; `brier_baseline` of the base rate is in the CSV).", ""]
    for scen in ("A", "B"):
        p = purchase["primary"][scen]
        L += [f"### Scenario {scen} ({'new garment, attributes only' if scen == 'A' else 'attributes + prior Trial_Count'})",
              f"Primary model: **{p['model']}**. {p['note']}", ""]
        L.append(md_table(m[m["scenario"] == scen][cols].drop(columns=["scenario"])))
        L += ["", "Classification report of the primary model (customer-held-out test set, threshold 0.5):", "",
              "```", reports[scen], "```", ""]
    L += ["## 2. Feature importance (screening)", "", md_table(imp.reset_index()), "",
          "## 3. Clustering quality (research methodology)", "",
          f"K = {cm['k']}: inertia {cm['inertia']:.1f}, silhouette {cm['silhouette']:.3f}. "
          f"The K in 2..10 with the highest silhouette is {cm['k_with_highest_silhouette']}; K = 6 is used "
          "because the research brief reports six clusters.", "", md_table(stack["k_eval"]), "",
          "## 4. Cluster weights and relative demand (research methodology)", "",
          "Cluster weight = share of fitting-room records, i.e. observed preference - NOT a purchase probability.", "",
          md_table(stack["weights"][["Cluster", "Label", "Records", "Weight_Pct", "Relative_Demand_Index", "Purchase_Rate"]]), "",
          "## 5. Decision Tree classification of garments into clusters", "",
          f"Held-out accuracy {cm['decision_tree']['holdout_accuracy']:.3f}, macro-F1 "
          f"{cm['decision_tree']['holdout_macro_f1']:.3f}, depth {cm['decision_tree']['depth']}, "
          f"{cm['decision_tree']['leaves']} leaves. Clusters are defined by the same aggregated attributes the tree "
          "sees, so very high accuracy is expected: the tree is an interpretable, deployable replica of the "
          "clustering, not independent evidence that the clusters are 'real'."]
    (C.REPORTS_DIR / "results_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
