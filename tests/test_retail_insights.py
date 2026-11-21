import numpy as np
import pandas as pd
import pytest

import config as C
from src import retail_insights as R
from src import trends as T
from src.purchase_prediction import PurchasePredictor


@pytest.fixture(scope="module")
def scorecard(small_df, models_dir):
    return R.build_scorecard(small_df, PurchasePredictor(models_dir), T.simulated_trends(13))


def test_one_row_per_product(scorecard, small_df):
    assert len(scorecard) == small_df["Garment_ID"].nunique()
    assert scorecard["Product ID"].is_unique
    assert scorecard["Likelihood_Pct"].between(0, 100).all()
    assert scorecard["Trials"].sum() == len(small_df)


def test_levels_actions_and_reasons_valid(scorecard):
    assert set(scorecard["Likelihood"]) <= set(R.TIERS)
    assert set(scorecard["Action"]) <= set(R.ACTIONS)
    assert set(scorecard["Demand"]) <= {"High", "Medium", "Low"}
    assert scorecard["Reason"].str.len().gt(10).all()


def test_tiers_follow_probability(scorecard):
    high = scorecard.loc[scorecard["Likelihood"] == "High", "Likelihood_Pct"]
    low = scorecard.loc[scorecard["Likelihood"] == "Low", "Likelihood_Pct"]
    if len(high) and len(low):
        assert high.min() > low.max()


def test_sparse_products_use_attribute_only_estimate(scorecard):
    sparse = scorecard[scorecard["Trials"] < R.MIN_TRIALS_FOR_HISTORY]
    assert (sparse["Estimate_Basis"] == "Product attributes").all()


@pytest.mark.parametrize("tier,demand,trend,expected", [
    ("High", "High", "Stable", "Stock up"),
    ("High", "Medium", "Rising", "Stock up"),
    ("High", "Low", "Stable", "Promote"),
    ("Low", "High", "Stable", "Review price / fit"),
    ("Low", "Low", "Falling", "Limit stock"),
    ("Medium", "Medium", "Stable", "Monitor"),
    ("Medium", "Low", "Rising", "Promote"),
])
def test_action_rules(tier, demand, trend, expected):
    assert R.action_for(tier, demand, trend)[0] == expected


def test_trend_direction_and_changes():
    t = T.simulated_trends(13)
    ch = R.trend_changes(t)
    assert set(ch) == set(C.TREND_KEYWORDS)
    assert R.direction(20) == "Rising" and R.direction(-20) == "Falling" and R.direction(1) == "Stable"
    assert T.simulated_trends(13)["fit:Slim"].equals(t["fit:Slim"])        # deterministic
    assert "date" in t.columns


def test_weekly_activity_and_period_comparison(small_df):
    w = R.weekly_activity(small_df)
    assert w["Trials"].sum() == len(small_df) and w["Conversion_Pct"].between(0, 100).all()
    pc = R.period_comparison(small_df)
    assert pc["trials"] > 0


def test_attribute_summary(small_df):
    a = R.attribute_summary(small_df, "Fit")
    assert abs(a["Share_Pct"].sum() - 100) < 0.2
    assert a["Trials"].sum() == len(small_df)
