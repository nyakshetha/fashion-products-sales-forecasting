import numpy as np
import pandas as pd

import config as C
from src import preprocessing
from src.feature_engineering import (add_groups, build_customer_profiles, make_cluster_encoder,
                                     make_purchase_preprocessor, purchase_feature_names)


def test_groupings_cover_every_category():
    assert set(C.COLORS) == set(C.COLOR_TO_GROUP)
    assert set(C.PATTERNS) == set(C.PATTERN_TO_GROUP)
    assert set(C.SIZES) == set(C.SIZE_TO_GROUP)


def test_add_groups_and_price_bands(small_df):
    g = add_groups(small_df)
    assert set(g["Color_Group"]) <= {"warm", "cool", "neutral"}
    assert set(g["Pattern_Group"]) <= {"solid", "texture", "geometric", "abstract"}
    assert (g["Price_Band_Idx"] == small_df["Price"] // 500).all()
    row = add_groups(pd.DataFrame([{"Size": 40, "Color": "Black", "Pattern": "Dots", "Price": 934}]))
    assert row["Price_Band"].iloc[0] == "500-999" and row["Pattern_Group"].iloc[0] == "geometric"


def test_cluster_encoder_range_and_inference(small_df):
    enc = make_cluster_encoder()
    X = enc.fit_transform(small_df[C.ATTR_COLS])
    assert X.min() >= 0 and X.max() <= 1                      # nothing dominates K-Means
    X_new = enc.transform(small_df[C.ATTR_COLS].head(3))
    assert X_new.shape[1] == X.shape[1]


def test_purchase_preprocessor_excludes_target_and_ids():
    for scen in "AB":
        feats = purchase_feature_names(scen)
        assert C.TARGET not in feats and "Customer_ID" not in feats and "Garment_ID" not in feats
    assert "Trial_Count" in purchase_feature_names("B") and "Trial_Count" not in purchase_feature_names("A")


def test_purchase_preprocessor_handles_unseen_category(small_df):
    prep = make_purchase_preprocessor("B").fit(small_df[purchase_feature_names("B")])
    row = small_df[purchase_feature_names("B")].head(1).copy()
    row["Color"] = "Magenta"
    assert np.isfinite(prep.transform(row)).all()


def test_quality_reports(small_df):
    rep = preprocessing.preprocessing_report(small_df)
    assert rep["schema_problems"] == []
    assert rep["target"]["missing_labels"] == 0
    assert rep["duplicates"]["exact_duplicate_rows"] == 0
    bad = small_df.copy()
    bad.loc[0, "Price"] = 99999
    assert preprocessing.schema_problems(bad)


def test_customer_profiles(small_df):
    prof = build_customer_profiles(small_df)
    assert len(prof) == small_df["Customer_ID"].nunique()
    share_cols = [c for c in prof.columns if c.startswith("Color_Group_")]
    assert np.allclose(prof[share_cols].sum(axis=1), 1.0)
