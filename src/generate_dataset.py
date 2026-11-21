"""Synthetic fitting-room dataset generator (men's formal shirts, ~1,630 customers / ~6,540 trials).

IMPORTANT: every record is SIMULATED. Nothing here is original research data. The generator only
reproduces the *scale and structure* of the fitting-room setting described in the project brief.

Data-generating process (all seeded):
  1. A garment catalogue (~420 designs) with attributes, a price, a latent "design appeal" and a
     latent "display popularity" (independent, so some garments are tried a lot but convert poorly).
  2. Customers belong to one of five latent taste segments; individual preferences are the segment
     preference plus noise, so segments overlap heavily.
  3. Each customer visits on one day and tries 1-8 garments, drawn without replacement with
     probability driven by popularity, taste affinity and price fit.
  4. Purchase probability = sigmoid(intercept + hidden buyer propensity + taste affinity + price fit
     + design appeal + size match + size x fit interaction + trend signal + noise); the intercept is
     solved so overall conversion is ~41 %, then labels are drawn Bernoulli.
  5. Trial_Count = number of EARLIER trials of the same garment (chronological order), so it never
     contains the current or any future trial -> no leakage.

Run:  python -m src.generate_dataset
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

import config as C
from src import trends as T

COLOR_APPEAL = {"Beige": -0.05, "Black": 0.15, "Blue": 0.20, "Brown": -0.15, "Cream": -0.10,
                "Green": -0.05, "Grey": 0.05, "Khaki": 0.00, "Light Blue": 0.12, "Lilac": -0.25}
PATTERN_APPEAL = {"Print": -0.10, "Patterned": -0.05, "Graphic": -0.20, "Checks": 0.10,
                  "Dots": -0.10, "Stripes": 0.10, "Solid": 0.15, "Textured": 0.05}
PATTERN_PREMIUM = {"Print": 0.10, "Patterned": 0.08, "Graphic": 0.10, "Checks": 0.03,
                   "Dots": 0.03, "Stripes": 0.02, "Solid": -0.06, "Textured": 0.12}

SEGMENTS = [
    dict(name="Classic professional", p=0.28, price=1800, sens=0.9,
         color={"warm": -0.1, "cool": 0.3, "neutral": 0.3},
         pattern={"solid": 0.5, "texture": 0.1, "geometric": 0.2, "abstract": -0.5},
         fit={"Slim": 0.3, "Ultra Slim": -0.3}, collar={"Regular": 0.4, "Button-down": 0.2, "Band": -0.5},
         sleeve={"Full": 0.4, "Half": -0.4}),
    dict(name="Trend-forward", p=0.24, price=1500, sens=0.8,
         color={"warm": 0.0, "cool": 0.3, "neutral": 0.1},
         pattern={"solid": -0.2, "texture": 0.1, "geometric": 0.3, "abstract": 0.5},
         fit={"Slim": 0.0, "Ultra Slim": 0.4}, collar={"Regular": -0.1, "Button-down": 0.2, "Band": 0.3},
         sleeve={"Full": 0.0, "Half": 0.1}),
    dict(name="Premium", p=0.14, price=2800, sens=0.5,
         color={"warm": 0.1, "cool": 0.1, "neutral": 0.4},
         pattern={"solid": 0.3, "texture": 0.5, "geometric": 0.0, "abstract": -0.3},
         fit={"Slim": 0.2, "Ultra Slim": -0.1}, collar={"Regular": 0.0, "Button-down": 0.5, "Band": -0.2},
         sleeve={"Full": 0.5, "Half": -0.5}),
    dict(name="Value seeker", p=0.20, price=1200, sens=1.3,
         color={"warm": 0.0, "cool": 0.2, "neutral": 0.2},
         pattern={"solid": 0.3, "texture": -0.2, "geometric": 0.2, "abstract": -0.1},
         fit={"Slim": 0.2, "Ultra Slim": -0.2}, collar={"Regular": 0.3, "Button-down": 0.0, "Band": -0.3},
         sleeve={"Full": 0.1, "Half": 0.1}),
    dict(name="Casual smart", p=0.14, price=1600, sens=0.9,
         color={"warm": 0.5, "cool": 0.0, "neutral": -0.2},
         pattern={"solid": -0.1, "texture": 0.2, "geometric": 0.5, "abstract": 0.1},
         fit={"Slim": 0.1, "Ultra Slim": 0.0}, collar={"Regular": 0.1, "Button-down": 0.3, "Band": 0.0},
         sleeve={"Full": -0.2, "Half": 0.5}),
]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _solve_intercept(z: np.ndarray, target: float) -> float:
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sigmoid(z + mid).mean() > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def _build_catalog(rng: np.random.Generator, n: int) -> pd.DataFrame:
    color = rng.choice(C.COLORS, n, p=C.COLOR_PROBS)
    pattern = rng.choice(C.PATTERNS, n, p=C.PATTERN_PROBS)
    sleeve = rng.choice(C.SLEEVES, n, p=C.SLEEVE_PROBS)
    collar = rng.choice(C.COLLARS, n, p=C.COLLAR_PROBS)
    fit = rng.choice(C.FITS, n, p=C.FIT_PROBS)
    premium = np.array([PATTERN_PREMIUM[p] + (0.05 if c == "Band" else 0) + (0.06 if f == "Ultra Slim" else 0)
                        + (0.07 if s == "Full" else 0) for p, c, f, s in zip(pattern, collar, fit, sleeve)])
    mean_log = math.log(1750) + premium
    prices = np.empty(n, dtype=int)
    for i in range(n):
        while True:
            x = float(np.exp(rng.normal(mean_log[i], 0.36)))
            u = rng.random()
            if u < 0.65:
                p = int(round(x / 100) * 100 - 1)       # 1299, 1499 ...
            elif u < 0.85:
                p = int(round(x / 50) * 50 - 1)         # 1349, 1449 ...
            else:
                p = int(round(x / 5) * 5 - 1)           # 934, 1274 ...
            if C.PRICE_MIN <= p <= C.PRICE_MAX:
                prices[i] = p
                break
    prices[0], prices[1] = C.PRICE_MIN, C.PRICE_MAX      # anchor the stated price range
    appeal = np.array([COLOR_APPEAL[c] + PATTERN_APPEAL[p] + (0.12 if f == "Slim" else 0.0)
                       + (-0.2 if cl == "Band" else 0.0) for c, p, f, cl in zip(color, pattern, fit, collar)])
    return pd.DataFrame({
        "Garment_ID": [f"G{i + 1:04d}" for i in range(n)],
        "Sleeve_Type": sleeve, "Color": color, "Collar_Type": collar, "Fit": fit,
        "Pattern": pattern, "Price": prices,
        "_quality": 2.0 * appeal + rng.normal(0, 0.35, n),    # design appeal: observable part + hidden part
        "_pop": 0.5 * appeal + rng.normal(0, 0.75, n),        # display popularity (largely independent of appeal)
    })


def generate(n_customers: int = C.N_CUSTOMERS, n_trials: int = C.N_TRIALS,
             n_garments: int = C.N_GARMENTS, seed: int = C.SEED) -> pd.DataFrame:
    if not n_customers <= n_trials <= 8 * n_customers:
        raise ValueError("n_trials must be between n_customers and 8 * n_customers")
    rng = np.random.default_rng(seed)
    cat = _build_catalog(rng, n_garments)
    G = len(cat)
    cg = np.array([C.COLOR_GROUP_ORDER.index(C.COLOR_TO_GROUP[c]) for c in cat["Color"]])
    pg = np.array([C.PATTERN_GROUP_ORDER.index(C.PATTERN_TO_GROUP[p]) for p in cat["Pattern"]])
    fi = np.array([C.FITS.index(f) for f in cat["Fit"]])
    co = np.array([C.COLLARS.index(c) for c in cat["Collar_Type"]])
    sl = np.array([C.SLEEVES.index(s) for s in cat["Sleeve_Type"]])
    ln_price = np.log(cat["Price"].to_numpy(dtype=float))
    quality, pop = cat["_quality"].to_numpy(), cat["_pop"].to_numpy()

    # trials per customer (1..8) summing exactly to n_trials
    k = np.clip(1 + rng.poisson(3.0, n_customers), 1, 8)
    diff = n_trials - int(k.sum())
    while diff != 0:
        i = int(rng.integers(0, n_customers))
        if diff > 0 and k[i] < 8:
            k[i] += 1; diff -= 1
        elif diff < 0 and k[i] > 1:
            k[i] -= 1; diff += 1

    seg_ids = rng.choice(len(SEGMENTS), n_customers, p=[s["p"] for s in SEGMENTS])
    day_w = np.where(np.arange(C.STUDY_DAYS) % 7 >= 5, 1.8, 1.0)
    day_w = day_w / day_w.sum()

    rows = []
    for ci in range(n_customers):
        s = SEGMENTS[seg_ids[ci]]
        pc = np.array([s["color"][g] for g in C.COLOR_GROUP_ORDER]) + rng.normal(0, 0.3, 3)
        pp = np.array([s["pattern"][g] for g in C.PATTERN_GROUP_ORDER]) + rng.normal(0, 0.3, 4)
        pf = np.array([s["fit"][f] for f in C.FITS]) + rng.normal(0, 0.3, 2)
        pco = np.array([s["collar"][c] for c in C.COLLARS]) + rng.normal(0, 0.3, 3)
        ps = np.array([s["sleeve"][x] for x in C.SLEEVES]) + rng.normal(0, 0.3, 2)
        mu = math.log(s["price"]) + rng.normal(0, 0.22)
        sens = s["sens"] * math.exp(rng.normal(0, 0.2))
        propensity = rng.normal(0, 0.4)                     # hidden "ready to buy" tendency
        own = int(rng.choice(len(C.SIZES), p=C.SIZE_PROBS))

        aff = pc[cg] + pp[pg] + pf[fi] + pco[co] + ps[sl]
        price_term = -0.5 * ((ln_price - mu) / 0.5) ** 2 * sens
        w = np.exp(pop + 0.7 * aff + 0.5 * price_term)
        chosen = rng.choice(G, size=int(k[ci]), replace=False, p=w / w.sum())

        day = int(rng.choice(C.STUDY_DAYS, p=day_w))
        minute = int(rng.integers(600, 1230))
        for g in chosen:
            idx = own if rng.random() < 0.82 else int(np.clip(own + rng.choice([-1, 1]), 0, len(C.SIZES) - 1))
            size = C.SIZES[idx]
            size_term = 0.0 if idx == own else -0.7
            fit_name = cat["Fit"].iat[g]
            if fit_name == "Ultra Slim" and size >= 44:
                size_term -= 0.6
            elif fit_name == "Slim" and size >= 46:
                size_term -= 0.3
            elif fit_name == "Ultra Slim" and size <= 38:
                size_term += 0.2
            z = (propensity + 0.8 * aff[g] + price_term[g] - 0.8 * (ln_price[g] - math.log(1800))
                 + quality[g] + size_term)
            rows.append((ci, g, size, day * 1440 + minute, z))
            minute += int(rng.integers(6, 14))

    t = pd.DataFrame(rows, columns=["ci", "g", "Size", "minutes", "z"])
    week = (t["minutes"] // 1440 // 7).to_numpy()
    n_weeks = math.ceil(C.STUDY_DAYS / 7)
    table = T.simulated_trends(n_weeks, seed)
    pat_grp = [C.PATTERN_TO_GROUP[p] for p in cat["Pattern"].to_numpy()[t["g"]]]
    fit_nm = cat["Fit"].to_numpy()[t["g"]]
    trend = np.array([T.trend_index(pg_, f_, table, w_) for pg_, f_, w_ in zip(pat_grp, fit_nm, week)])
    z = t["z"].to_numpy() + 0.012 * (trend - 50) + rng.normal(0, 0.4, len(t))
    z = z + _solve_intercept(z, C.TARGET_CONVERSION)
    purchased = (rng.random(len(t)) < _sigmoid(z)).astype(int)

    df = pd.DataFrame({
        "Customer_ID": [f"C{c + 1:04d}" for c in t["ci"]],
        "Garment_ID": cat["Garment_ID"].to_numpy()[t["g"]],
        "Size": t["Size"].astype(int).to_numpy(),
        "Sleeve_Type": cat["Sleeve_Type"].to_numpy()[t["g"]],
        "Color": cat["Color"].to_numpy()[t["g"]],
        "Collar_Type": cat["Collar_Type"].to_numpy()[t["g"]],
        "Fit": fit_nm,
        "Pattern": cat["Pattern"].to_numpy()[t["g"]],
        "Price": cat["Price"].to_numpy()[t["g"]].astype(int),
        "Purchased": purchased,
        "Trial_Timestamp": pd.Timestamp(C.STUDY_START) + pd.to_timedelta(t["minutes"].to_numpy(), unit="m"),
        "Trend_Index": trend.round(1),
    })
    df = df.sort_values(["Trial_Timestamp", "Customer_ID"], kind="stable").reset_index(drop=True)
    df["Trial_Count"] = df.groupby("Garment_ID").cumcount()      # EARLIER trials only -> no leakage
    df["Trial_Timestamp"] = df["Trial_Timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    return df[C.ALL_COLUMNS]


# --------------------------------------------------------------------------------------------
# Validation and documentation
# --------------------------------------------------------------------------------------------
def validate(df: pd.DataFrame) -> dict:
    checks = {
        "rows": len(df), "customers": df["Customer_ID"].nunique(), "garments": df["Garment_ID"].nunique(),
        "missing_values": int(df.isna().sum().sum()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_customer_garment_pairs": int(df.duplicated(["Customer_ID", "Garment_ID"]).sum()),
        "invalid_sizes": int((~df["Size"].isin(C.SIZES)).sum()),
        "invalid_sleeves": int((~df["Sleeve_Type"].isin(C.SLEEVES)).sum()),
        "invalid_colors": int((~df["Color"].isin(C.COLORS)).sum()),
        "invalid_collars": int((~df["Collar_Type"].isin(C.COLLARS)).sum()),
        "invalid_fits": int((~df["Fit"].isin(C.FITS)).sum()),
        "invalid_patterns": int((~df["Pattern"].isin(C.PATTERNS)).sum()),
        "price_min": int(df["Price"].min()), "price_max": int(df["Price"].max()),
        "invalid_prices": int(((df["Price"] < C.PRICE_MIN) | (df["Price"] > C.PRICE_MAX)).sum()),
        "invalid_target_values": int((~df["Purchased"].isin([0, 1])).sum()),
        "conversion_rate": round(float(df["Purchased"].mean()), 4),
    }
    ordered = df.sort_values("Trial_Timestamp", kind="stable")   # rows are stored in time order
    checks["trial_count_leakage_violations"] = int(
        (ordered["Trial_Count"].to_numpy() != ordered.groupby("Garment_ID").cumcount().to_numpy()).sum())
    const = df.groupby("Garment_ID")[["Sleeve_Type", "Color", "Collar_Type", "Fit", "Pattern", "Price"]].nunique()
    checks["garments_with_inconsistent_attributes"] = int((const > 1).any(axis=1).sum())
    hard = ["missing_values", "exact_duplicate_rows", "duplicate_customer_garment_pairs", "invalid_sizes",
            "invalid_sleeves", "invalid_colors", "invalid_collars", "invalid_fits", "invalid_patterns",
            "invalid_prices", "invalid_target_values", "trial_count_leakage_violations",
            "garments_with_inconsistent_attributes"]
    checks["passed"] = all(checks[h] == 0 for h in hard)
    return checks


DICTIONARY = [
    ("Customer_ID", "text", "Anonymous customer identifier (one fitting-room visit per customer)."),
    ("Garment_ID", "text", "Identifier of a garment design; its colour/pattern/collar/sleeve/fit/price never change."),
    ("Size", "int", "Size tried on by the customer (36, 38, 39, 40, 42, 44, 46, 48)."),
    ("Sleeve_Type", "category", "Half or Full."),
    ("Color", "category", "Beige, Black, Blue, Brown, Cream, Green, Grey, Khaki, Light Blue, Lilac."),
    ("Collar_Type", "category", "Button-down, Regular, Band."),
    ("Fit", "category", "Slim or Ultra Slim."),
    ("Pattern", "category", "Print, Patterned, Graphic, Checks, Dots, Stripes, Solid, Textured."),
    ("Price", "int (INR)", "Garment price, INR 934-3,999."),
    ("Trial_Count", "int", "Number of EARLIER fitting-room trials of this garment (current trial excluded). 0 = cold start."),
    ("Purchased", "0/1", "TARGET. 1 = bought after trying it on, 0 = did not."),
    ("Trial_Timestamp", "datetime", "Simulated trial time; defines the order used to compute Trial_Count."),
    ("Trend_Index", "float 0-100", "Weekly search-interest index for the garment's pattern group and fit (simulated here; PyTrends in live mode)."),
]


def write_documents(df: pd.DataFrame, checks: dict) -> None:
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    dd = ["# Data dictionary", "", "| Column | Type | Description |", "|---|---|---|"]
    dd += [f"| {a} | {b} | {c} |" for a, b, c in DICTIONARY]
    (C.DATA_DIR / "DATA_DICTIONARY.md").write_text("\n".join(dd) + "\n", encoding="utf-8")

    desc = f"""# Dataset description

**This dataset is SYNTHETIC.** It was generated by `src/generate_dataset.py` (seed {C.SEED}) to emulate the
scale and structure of a fitting-room study of men's formal shirts (about 1,630 customers and 6,540 garment
trials over three months). It is not the original research data and must not be presented as such.

* Rows (garment trials): {checks['rows']:,}
* Customers: {checks['customers']:,}   Garment designs observed: {checks['garments']:,}
* Overall purchase rate: {checks['conversion_rate']:.1%}  (target label `Purchased`)
* Price range (INR): {checks['price_min']:,} - {checks['price_max']:,}

Generation logic (see the module docstring for details): latent customer taste segments with overlapping
preferences, price-fit and budget sensitivity, size-match and size x fit effects, hidden garment design appeal
and display popularity (independent, so some garments are tried often but convert poorly), a weekly trend
signal, plus noise. `Trial_Count` is computed chronologically from earlier trials only.

Because the labels come from a simulated process, model performance on this data shows that the pipeline
works, not how well it would predict real shoppers.
"""
    (C.DATA_DIR / "README.md").write_text(desc, encoding="utf-8")

    lines = ["# Dataset validation report", "", f"Result: **{'PASSED' if checks['passed'] else 'FAILED'}**", "",
             "| Check | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in checks.items() if k != "passed"]
    lines += ["", "## Purchase rate by attribute (sanity check of relationships)", ""]
    for col in ["Fit", "Pattern", "Color", "Collar_Type", "Sleeve_Type"]:
        rate = df.groupby(col)["Purchased"].mean().sort_values(ascending=False)
        lines.append(f"**{col}**: " + ", ".join(f"{i} {v:.0%}" for i, v in rate.items()))
        lines.append("")
    (C.REPORTS_DIR / "dataset_validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> pd.DataFrame:
    ap = argparse.ArgumentParser(description="Generate the synthetic fitting-room dataset")
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()
    df = generate(seed=args.seed)
    checks = validate(df)
    C.RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.RAW_PATH, index=False, encoding="utf-8")
    write_documents(df, checks)
    print(f"Saved {len(df):,} rows -> {C.RAW_PATH}")
    print(f"Customers={checks['customers']:,} Garments={checks['garments']} "
          f"Conversion={checks['conversion_rate']:.1%} Validation={'PASSED' if checks['passed'] else 'FAILED'}")
    return df


if __name__ == "__main__":
    main()
