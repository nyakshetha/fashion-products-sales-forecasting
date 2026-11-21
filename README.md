# Fashion Products Sales Forecasting - Fitting Room Based Purchase Likelihood Prediction

> **Synthetic-data disclosure.** The original fitting-room data are not available. Every record here is
> **simulated** by `src/generate_dataset.py` (fixed seed). Model results on synthetic data show that the pipeline
> works; they **do not establish real-world predictive performance**.

## 1. Abstract
A fitting-room based demand-estimation system for men's formal shirts. It (a) reproduces a research-style workflow -
feature selection, feature aggregation, K-Means preference clusters, cluster weights and a Decision Tree that assigns new
garments to clusters - and (b) adds a **supervised purchase-likelihood extension** (XGBoost and alternatives) that outputs
`Purchase Likelihood (%)` for a garment, including a cold-start scenario. A Google-Trends (PyTrends) interest index is used
as an external signal. A Streamlit app turns the outputs into inventory decision support.

## 2. Introduction
Fashion retailers must stock garments that have no sales history. Sales records only show completed purchases, whereas
fitting-room data also show what customers wanted to try but did not buy.

## 3. Problem statement
Estimate, for a newly introduced garment, (i) which customer-preference segment it belongs to and how large that segment is,
and (ii) how likely a fitting-room trial of it is to end in a purchase.

## 4. Research objectives
1. Learn customer-preference clusters from garment attributes tried in the fitting room.
2. Weight clusters by their share of fitting-room records and use the weight as relative demand potential.
3. Classify new garments into clusters with a Decision Tree.
4. (Extension) Estimate purchase probability with a supervised model, with and without trial history.
5. Translate outputs into cautious inventory decision support.

## 5. Literature review (brief)
Classical forecasting (moving average, ARIMA) struggles with fashion's non-linear, fast-changing demand; gradient boosting is
widely used on tabular retail data, and search-interest signals are used as external features. The related IEEE paper below
forecasts *weekly sales from historical sales* - a time-series problem that needs sales history, which new garments lack. This
project instead follows the fitting-room approach, and borrows only the XGBoost + Google-Trends ideas as add-ons.

## 6. Research references
* **Methodology basis:** Anitha & Neelakandan (2024), fitting-room based cluster-weighted demand estimation. *I could not open
  this paper while building the project; the workflow follows the project brief. Please complete and verify the full citation
  and check each step against the paper.*
* **Related work:** Yakshetha N., Sherin Riya S., J. Gold Beulah Patturose, R. Priscilla, "Fashion Products Sales Forecasting,"
  2025 IEEE 7th Int. Conf. on Computing, Communication and Automation (ICCCA), DOI 10.1109/ICCCA66364.2025.11325123
  (details taken from the PDF you attached). Its reported XGBoost error figures are **not** reproduced here.

## 7. Original research methodology (as specified in the brief)
Fitting-room collection -> feature selection (Decision Tree / Random Forest / Lasso) -> feature aggregation (colour warm/cool/
neutral, pattern solid/texture/geometric/abstract, size groups, INR 500 price bands) -> K-Means (elbow + silhouette; K = 6)
-> cluster weights -> Decision Tree classification of new garments -> demand estimation.

## 8. Synthetic dataset methodology
`python -m src.generate_dataset` creates 6,540 trials by 1,630 customers over 90 simulated days: a 420-design garment
catalogue (prices INR 934-3,999), five overlapping customer taste segments, price-fit and size-match effects, size x fit
interaction, a weekly trend signal and noise. Garments carry independent hidden "popularity" and "appeal", so some are tried
often but convert poorly. Overall conversion is about 40 % (not forced to 50:50). `Trial_Count` counts **earlier** trials of
the garment only. Documentation is produced with the data: `data/DATA_DICTIONARY.md`, `data/README.md`,
`outputs/reports/dataset_validation_report.md`.

*Generator tuning disclosure:* my first generator version let hidden customer/garment effects dominate, giving near-chance
garment-held-out ROC-AUC (~0.53). I strengthened population-level attribute effects (price, fit, colour, size x fit) and reduced
latent noise. This is a design choice of the synthetic process, which is why results remain demonstrations only.

## 9. Dataset schema
`Customer_ID, Garment_ID, Size, Sleeve_Type, Color, Collar_Type, Fit, Pattern, Price, Trial_Count, Purchased` plus
`Trial_Timestamp` and `Trend_Index`. See `data/DATA_DICTIONARY.md`.

## 10. Feature descriptions
Size (36-48), Sleeve (Half/Full), Colour (10), Collar (Button-down/Regular/Band), Fit (Slim/Ultra Slim), Pattern (8), Price (INR),
Trial_Count (prior trials; 0 = cold start), Trend_Index (0-100 search interest for the pattern group and fit),
Purchased (target 0/1 - never used as an input; IDs are never used as features).

## 11. Data preprocessing
`src/preprocessing.py`: missing values, duplicates, IQR outlier report (reported, not removed), schema and target validation.
Encoders live in sklearn pipelines (`src/feature_engineering.py`) so training and inference use identical transforms; unseen
categories are ignored safely. K-Means inputs are one-hot (0/1) plus a min-max price band, so no feature dominates.

## 12. Feature selection
Decision Tree, Random Forest and Lasso importances, summed per original attribute (`outputs/metrics/feature_importance.csv`).
Screening only - not the final models. Screening target is `Purchased`; the brief does not state the original target.

## 13. Customer-profile aggregation
Group mappings in `config.py`. `data/processed/customer_profiles.csv` holds per-customer shares of each group. Clustering is
done at fitting-room-record level so that cluster weight = records / total records, as in the brief.

## 14. K-Means clustering
K = 2..10 evaluated (inertia, silhouette); **K = 6 is used because the brief reports six clusters**, not because this data
singles it out - the highest-silhouette K on the data is usually smaller and silhouette is low (~0.14), i.e. overlapping groups.

## 15. Cluster-weight calculation
`Cluster Weight (%) = records in cluster / total records x 100`; `Relative Demand Index = weight / (100 / K)`. Cluster weight
measures observed **preference**, not purchase probability (the plots show both, side by side).

## 16. Decision Tree classification
A Decision Tree maps raw garment attributes to a cluster. The clusters come from the same inputs the tree sees, so ~100 %
held-out accuracy is expected: it is an interpretable replica of the clustering, not independent validation.

## 17. Purchase-likelihood extension (not in the original methodology)
Supervised classifier for `Purchased`. **Scenario A** = new garment (attributes + trend index). **Scenario B** = plus prior
`Trial_Count`. Candidates: Logistic Regression, Decision Tree, Random Forest, **XGBoost**. XGBoost is the designated primary
model (project requirement); the comparison is reported openly and `summary.json` records any gap to the best model.
Evaluation is customer-held-out and garment-held-out (`GroupShuffleSplit`), so no customer/garment appears in both train and test.

## 18. Model evaluation
Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier score (vs the base-rate Brier), confusion matrix, classification
report, ROC / PR / calibration curves. Probability is not accuracy: a 70 % likelihood means 70 % of similar trials convert.

## 19. Experimental results (snapshot from my sandbox run)
Run in an environment **without** `xgboost` (a compatibility check in `model_training.py` falls back to a stand-in only if XGBoost is missing or broken, and prints a warning), so `HistGradientBoosting` stood in for XGBoost. **Re-run `python main.py` on your
machine to get XGBoost results** (`outputs/reports/results_report.md`). Scenario A, customer-held-out:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.651 | 0.567 | 0.373 | 0.450 | 0.652 | 0.221 |
| Decision Tree | 0.598 | 0.468 | 0.361 | 0.408 | 0.588 | 0.236 |
| Random Forest | 0.658 | 0.583 | 0.371 | 0.454 | 0.650 | 0.221 |
| HistGradientBoosting (stand-in) | 0.637 | 0.538 | 0.379 | 0.445 | 0.647 | 0.222 |

Brier of always predicting the base rate is 0.236. Garment-held-out ROC-AUC is lower (~0.56-0.61). `Trial_Count` adds
essentially nothing over attributes (scenario B ~ scenario A). Clustering: K = 6, silhouette 0.138, six clusters of 13-20 % each.

## 20. Inventory-management applications
`src/inventory_insights.py` combines the likelihood tier (relative to the base conversion rate +/- 10 points) with the cluster's
relative demand index. Output is qualitative advice (prioritise / monitor / review). **No stock quantity is produced**: that
needs store traffic, current stock, lead time and costs.

## 21. System architecture
`generate_dataset -> data_loader -> preprocessing/feature_engineering -> feature_selection -> clustering + cluster_weighting
-> model_training -> models/*.pkl -> purchase_prediction (+ inventory_insights) -> app.py`. `main.py` runs everything;
`src/trends.py` supplies the PyTrends signal (live, cached, or simulated fallback).

## 22. Installation (Windows, PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
If PowerShell blocks activation: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

## 23. Execution
```powershell
python -m src.generate_dataset      # optional: dataset is already included (same seed -> same data)
python main.py                      # trains everything, saves models, metrics, plots, reports
python main.py --live-trends        # same, and tries to download real Google Trends data first
python -m pytest tests/ -v          # unit tests
streamlit run app.py                # launch the dashboard (trains once on first launch if models/ is empty)
```

## 24. Streamlit dashboard (StockSense)
A retailer-facing decision-support tool. **By design it shows no research-paper details, author names, methodology or
dataset-generation information** - all of that lives only in this README and the `data/` documentation.

Pages: *Overview* (KPI cards, weekly demand, likelihood mix, this week's priorities) - *Product Scorecard* (purchase
likelihood % for every product, High / Medium / Low level, filters, CSV export) - *Demand & Popularity* (what shoppers try vs buy,
by colour / pattern / fit / collar / sleeve / size / price band) - *Search Trends* (Google Trends interest, products matching
rising searches) - *New Product Check* (likelihood for a new or already-tried product, with what drives the score) -
*Action Plan* (Stock up / Promote / Monitor / Review price-fit / Limit stock, CSV export).

Notes
* First launch: if `models/` is empty the app trains the models once (about a minute), then loads the saved models on every
  later start. `python main.py` does the same thing and also writes plots and reports.
* Sidebar shows a small "Demo data" tag; set `SHOW_DEMO_BADGE = False` in `config.py` when you switch to real store data.
* If live Google Trends is not connected the Search Trends page says so and shows sample interest data instead.
* Action rules (`src/retail_insights.py`): High likelihood + high traffic -> Stock up; High + low traffic -> Promote; Low + traffic
  -> Review price / fit; Low + low traffic -> Limit stock; otherwise Monitor (rising searches can upgrade a product).

## 25. Google Trends (PyTrends) notes
PyTrends is unofficial and Google rate-limits it. If a live fetch fails the code falls back to cached or simulated values and
labels the source. The dataset's `Trend_Index` is simulated; live values are only used as inference defaults, are on a different
(0-100 per keyword) scale, and were never part of the training data, so treat them as a demonstration of the integration.

## 26. Limitations
Synthetic labels (the model learns the generator); one category and one store; no customer features at prediction time (limits
accuracy); K = 6 not supported by silhouette alone; Decision-Tree accuracy is circular; PyTrends is unofficial.

## 27. Future enhancements
Validate on real fitting-room logs; add customer-level features; probability calibration (isotonic) for the chosen model;
size-availability and stock data; multi-category support.

## 28. Conclusion
The project delivers a reproducible, tested fitting-room pipeline that separates cluster-based demand estimation from the
supervised purchase-likelihood extension and states clearly what the synthetic results do and do not show.
