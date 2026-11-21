"""All static plots (matplotlib, saved to outputs/plots/)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, roc_curve

import config as C
from src import evaluation
from src.feature_engineering import add_groups

PALETTE = ["#2E5E8C", "#D9822B", "#4C9F70", "#B5485D", "#7A6BAA", "#8C8C8C"]
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 11, "axes.labelsize": 9, "font.size": 9})


def _save(fig, name: str):
    C.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / name, bbox_inches="tight")
    plt.close(fig)


def _counts(ax, s: pd.Series, title: str, order=None, rot=0):
    vc = s.value_counts()
    vc = vc.reindex(order) if order is not None else vc.sort_index()
    ax.bar(vc.index.astype(str), vc.values, color=PALETTE[0])
    ax.set_title(title)
    ax.set_ylabel("Trials")
    ax.tick_params(axis="x", rotation=rot)


def _rate(ax, df: pd.DataFrame, col: str, order=None, rot=0, title=None):
    g = df.groupby(col)[C.TARGET].agg(["mean", "size"])
    g = g.reindex(order) if order is not None else g.sort_index()
    ax.bar(g.index.astype(str), g["mean"] * 100, color=PALETTE[1])
    ax.axhline(df[C.TARGET].mean() * 100, color="k", ls="--", lw=0.8, label="overall")
    ax.set_title(title or f"Purchase rate by {col}")
    ax.set_ylabel("Purchased (%)")
    ax.tick_params(axis="x", rotation=rot)
    for i, (m, n) in enumerate(zip(g["mean"], g["size"])):
        ax.text(i, m * 100 + 0.5, f"n={n}", ha="center", fontsize=6.5)


def eda_plots(df: pd.DataFrame) -> None:
    g = add_groups(df)
    # 1 overview
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    axes[0].hist(df.groupby("Customer_ID").size(), bins=np.arange(0.5, 9.5), color=PALETTE[0], rwidth=0.85)
    axes[0].set(title="Garments tried per customer", xlabel="Trials in one visit", ylabel="Customers")
    axes[1].hist(df.groupby("Garment_ID").size(), bins=25, color=PALETTE[2])
    axes[1].set(title="Fitting-room trials per garment", xlabel="Trials", ylabel="Garments")
    axes[2].hist(df["Trial_Timestamp"].dt.dayofweek, bins=np.arange(-0.5, 7.5), color=PALETTE[3], rwidth=0.85)
    axes[2].set(title="Trials by weekday (0 = Mon)", xlabel="Weekday", ylabel="Trials")
    _save(fig, "01_dataset_overview.png")
    # 2-8 attribute distributions
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    _counts(axes[0, 0], df["Size"], "Size", order=C.SIZES)
    _counts(axes[0, 1], df["Sleeve_Type"], "Sleeve type", order=C.SLEEVES)
    _counts(axes[0, 2], df["Color"], "Colour", rot=60)
    _counts(axes[1, 0], df["Collar_Type"], "Collar type", order=C.COLLARS)
    _counts(axes[1, 1], df["Fit"], "Fit", order=C.FITS)
    _counts(axes[1, 2], df["Pattern"], "Pattern", rot=45)
    _save(fig, "02_attribute_distributions.png")
    # price + trial count
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(df["Price"], bins=np.arange(500, 4500, 250), color=PALETTE[0], rwidth=0.9)
    ax.set(title="Price distribution (INR)", xlabel="Price (INR)", ylabel="Trials")
    _save(fig, "03_price_distribution.png")
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(df["Trial_Count"], bins=40, color=PALETTE[2])
    ax.set(title="Trial_Count (earlier trials of the same garment)", xlabel="Trial_Count", ylabel="Trials")
    _save(fig, "04_trial_count_distribution.png")
    # purchase distribution
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    vc = df[C.TARGET].value_counts().sort_index()
    ax.bar(["Not purchased", "Purchased"], vc.values, color=[PALETTE[5], PALETTE[2]])
    for i, v in enumerate(vc.values):
        ax.text(i, v, f"{v:,} ({v / len(df):.1%})", ha="center", va="bottom")
    ax.set(title="Purchase outcome of fitting-room trials", ylabel="Trials")
    _save(fig, "05_purchase_distribution.png")
    # conversion by attribute grid
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    _rate(axes[0, 0], df, "Size", C.SIZES)
    _rate(axes[0, 1], df, "Sleeve_Type", C.SLEEVES)
    _rate(axes[0, 2], df, "Collar_Type", C.COLLARS)
    _rate(axes[1, 0], df, "Fit", C.FITS)
    _rate(axes[1, 1], df, "Pattern", rot=45)
    _rate(axes[1, 2], g, "Color_Group", C.COLOR_GROUP_ORDER, title="Purchase rate by colour group")
    _save(fig, "06_conversion_by_attribute.png")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    order = sorted(g["Price_Band"].unique(), key=lambda s: int(s.split("-")[0]))
    _rate(ax, g, "Price_Band", order, rot=30, title="Purchase rate by price band (INR 500 bands)")
    _save(fig, "07_conversion_by_price_band.png")
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    _rate(ax, df, "Color", rot=40)
    _save(fig, "08_conversion_by_color.png")
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    _rate(ax, df, "Fit", C.FITS)
    _save(fig, "09_conversion_by_fit.png")
    # garment-level: trials vs conversion
    gg = df.groupby("Garment_ID").agg(trials=(C.TARGET, "size"), conv=(C.TARGET, "mean"))
    gg = gg[gg["trials"] >= 5]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(gg["trials"], gg["conv"] * 100, s=14, alpha=0.6, color=PALETTE[0])
    ax.set(title="Garments: trial volume vs conversion (>= 5 trials)", xlabel="Total trials",
           ylabel="Purchased (%)")
    _save(fig, "10_trials_vs_conversion.png")
    # correlation
    num = df[["Price", "Trial_Count", "Trend_Index", "Size", C.TARGET]].copy()
    num["Fit_UltraSlim"] = (df["Fit"] == "Ultra Slim").astype(int)
    num["Sleeve_Full"] = (df["Sleeve_Type"] == "Full").astype(int)
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation (numeric / binary-coded inputs)")
    _save(fig, "11_correlation.png")


def feature_importance_plot(imp: pd.DataFrame) -> None:
    cols = ["Decision Tree", "Random Forest", "Lasso (|coef|)"]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(imp))
    for i, c in enumerate(cols):
        ax.bar(x + (i - 1) * 0.27, imp[c] * 100, 0.27, label=c, color=PALETTE[i])
    ax.set_xticks(x, imp.index, rotation=30, ha="right")
    ax.set(ylabel="Relative importance (%)", title="Feature importance by screening method")
    ax.legend()
    _save(fig, "12_feature_importance.png")


def cluster_plots(stack: dict) -> None:
    ke, k = stack["k_eval"], C.N_CLUSTERS
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(ke["k"], ke["inertia"], "o-", color=PALETTE[0])
    axes[0].axvline(k, color="grey", ls="--", lw=0.8)
    axes[0].set(title="Elbow method", xlabel="K", ylabel="Inertia")
    axes[1].plot(ke["k"], ke["silhouette"], "o-", color=PALETTE[1])
    axes[1].axvline(k, color="grey", ls="--", lw=0.8)
    axes[1].set(title="Silhouette score", xlabel="K", ylabel="Silhouette")
    _save(fig, "13_cluster_selection.png")

    X, labels = stack["X"], stack["labels"]
    pts = PCA(n_components=2, random_state=C.SEED).fit_transform(X)
    rng = np.random.default_rng(C.SEED)
    jit = pts + rng.normal(0, 0.03, pts.shape)          # records share identical attribute combos
    fig, ax = plt.subplots(figsize=(6.4, 5))
    for c in range(k):
        m = labels == c
        ax.scatter(jit[m, 0], jit[m, 1], s=6, alpha=0.5, color=PALETTE[c % len(PALETTE)], label=f"C{c + 1}")
    ax.set(title="Preference clusters (PCA projection, slight jitter)", xlabel="PC1", ylabel="PC2")
    ax.legend(markerscale=2, ncol=3)
    _save(fig, "14_cluster_pca.png")

    w = stack["weights"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar([f"C{c + 1}" for c in w["Cluster"]], w["Records"], color=PALETTE[0])
    ax.set(title="Cluster distribution (records)", ylabel="Fitting-room records")
    _save(fig, "15_cluster_distribution.png")

    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(w))
    ax.bar(x - 0.2, w["Weight_Pct"], 0.4, label="Cluster weight % (trial share)", color=PALETTE[0])
    ax.bar(x + 0.2, w["Purchase_Rate"] * 100, 0.4, label="Purchase rate % (different quantity)", color=PALETTE[1])
    ax.set_xticks(x, [f"C{c + 1}" for c in w["Cluster"]])
    ax.set(title="Cluster weight vs purchase rate - not the same thing", ylabel="%")
    ax.legend(fontsize=7)
    _save(fig, "16_cluster_weights.png")


def model_plots(purchase: dict, scenario: str = "A") -> None:
    metrics = purchase["metrics"]
    for scen in ("A", "B"):
        sub = metrics[(metrics["scenario"] == scen)]
        fig, ax = plt.subplots(figsize=(8, 3.8))
        models = list(dict.fromkeys(sub["model"]))
        x = np.arange(len(models))
        for i, split in enumerate(["customer_heldout", "garment_heldout"]):
            vals = [sub[(sub["model"] == m) & (sub["split"] == split)]["roc_auc"].iloc[0] for m in models]
            ax.bar(x + (i - 0.5) * 0.38, vals, 0.38, label=split.replace("_", " "), color=PALETTE[i])
        ax.axhline(0.5, color="grey", ls=":", lw=0.8)
        ax.set_xticks(x, models, rotation=15, ha="right")
        ax.set(ylim=(0.4, 1.0), ylabel="ROC-AUC", title=f"Model comparison - scenario {scen}")
        ax.legend()
        _save(fig, f"17_model_comparison_{scen}.png")

        hold = purchase["holdout"][scen]
        primary = purchase["primary"][scen]["model"]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for i, (name, (yt, yp)) in enumerate(hold.items()):
            lw = 2.4 if name == primary else 1.0
            fpr, tpr, _ = roc_curve(yt, yp)
            axes[0].plot(fpr, tpr, lw=lw, label=name, color=PALETTE[i % 6])
            pr, rc, _ = precision_recall_curve(yt, yp)
            axes[1].plot(rc, pr, lw=lw, color=PALETTE[i % 6])
            fr, mp = calibration_curve(yt, yp, n_bins=8, strategy="quantile")
            axes[2].plot(mp, fr, "o-", lw=lw, ms=3, color=PALETTE[i % 6])
        axes[0].plot([0, 1], [0, 1], "k:", lw=0.8)
        axes[0].set(title=f"ROC (scenario {scen}, customer held out)", xlabel="False positive rate",
                    ylabel="True positive rate")
        axes[0].legend(fontsize=7)
        axes[1].axhline(purchase["base_rate"], color="k", ls=":", lw=0.8)
        axes[1].set(title="Precision-recall", xlabel="Recall", ylabel="Precision")
        axes[2].plot([0, 1], [0, 1], "k:", lw=0.8)
        axes[2].set(title="Calibration (predicted vs observed)", xlabel="Mean predicted probability",
                    ylabel="Observed purchase rate")
        _save(fig, f"18_roc_pr_calibration_{scen}.png")

        yt, yp = hold[primary]
        cm = evaluation.confusion(yt, yp)
        fig, ax = plt.subplots(figsize=(4, 3.6))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], ["Actual 0", "Actual 1"])
        ax.set_title(f"Confusion matrix - {primary}\n(scenario {scen}, threshold {C.CLASS_THRESHOLD})", fontsize=8)
        _save(fig, f"19_confusion_matrix_{scen}.png")
