import pandas as pd

import config as C
from src import generate_dataset


def test_full_scale_shape_and_validation():
    df = generate_dataset.generate()
    checks = generate_dataset.validate(df)
    assert len(df) == C.N_TRIALS
    assert df["Customer_ID"].nunique() == C.N_CUSTOMERS
    assert checks["passed"], checks
    assert C.PRICE_MIN == df["Price"].min() and C.PRICE_MAX == df["Price"].max()


def test_reproducible_with_fixed_seed():
    a = generate_dataset.generate(n_customers=100, n_trials=400, n_garments=60, seed=3)
    b = generate_dataset.generate(n_customers=100, n_trials=400, n_garments=60, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_columns_and_target(small_df):
    assert list(small_df.columns) == C.ALL_COLUMNS
    assert set(small_df["Purchased"].unique()) <= {0, 1}
    assert 0.25 < small_df["Purchased"].mean() < 0.6          # realistic, not forced 50:50


def test_attribute_values_valid(small_df):
    assert small_df["Size"].isin(C.SIZES).all()
    assert small_df["Color"].isin(C.COLORS).all()
    assert small_df["Pattern"].isin(C.PATTERNS).all()
    assert small_df["Fit"].isin(C.FITS).all()
    assert small_df["Price"].between(C.PRICE_MIN, C.PRICE_MAX).all()


def test_no_duplicate_customer_garment_pairs(small_df):
    assert not small_df.duplicated(["Customer_ID", "Garment_ID"]).any()


def test_trial_count_has_no_leakage(small_df):
    ordered = small_df.sort_values("Trial_Timestamp", kind="stable")
    assert (ordered["Trial_Count"].to_numpy() == ordered.groupby("Garment_ID").cumcount().to_numpy()).all()
    assert (small_df["Trial_Count"] >= 0).all()
    assert (small_df.groupby("Garment_ID")["Trial_Count"].min() == 0).all()   # first trial is cold start


def test_garment_attributes_consistent(small_df):
    n = small_df.groupby("Garment_ID")[["Color", "Pattern", "Fit", "Collar_Type", "Sleeve_Type", "Price"]].nunique()
    assert (n == 1).all().all()
