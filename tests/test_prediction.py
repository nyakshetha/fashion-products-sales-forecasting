import pytest

import config as C
from src import evaluation
from src.purchase_prediction import PurchasePredictor

GARMENT = dict(size=40, sleeve="Full", color="Blue", collar="Regular", fit="Slim", pattern="Solid", price=1499)


@pytest.fixture(scope="module")
def predictor(models_dir):
    return PurchasePredictor(models_dir)


def test_scenario_a_new_garment(predictor):
    r = predictor.predict(**GARMENT, trial_count=None)
    assert r["scenario"] == "A" and 0 <= r["probability"] <= 1
    assert r["likelihood_pct"] == round(r["probability"] * 100, 1)
    assert r["predicted_class"] in (0, 1) and 0 <= r["cluster"] < C.N_CLUSTERS
    assert 0 < r["cluster_weight_pct"] < 100 and r["recommendation"]["likelihood_tier"] in {"High", "Moderate", "Low"}
    assert r["explanation"]


def test_scenario_b_with_trials(predictor):
    r = predictor.predict(**GARMENT, trial_count=12)
    assert r["scenario"] == "B" and 0 <= r["probability"] <= 1


def test_probability_comes_from_model_and_varies(predictor):
    cheap = predictor.predict(**{**GARMENT, "price": 999}, explain=False)["probability"]
    pricey = predictor.predict(**{**GARMENT, "price": 3999, "pattern": "Graphic", "color": "Lilac"}, explain=False)["probability"]
    assert cheap != pricey


@pytest.mark.parametrize("bad", [dict(size=41), dict(color="Magenta"), dict(price=-5), dict(price="abc")])
def test_invalid_inputs_rejected(predictor, bad):
    with pytest.raises(ValueError):
        predictor.predict(**{**GARMENT, **bad})


def test_invalid_trial_count_rejected(predictor):
    with pytest.raises(ValueError):
        predictor.predict(**GARMENT, trial_count=-1)


def test_out_of_range_price_warns(predictor):
    assert predictor.predict(**{**GARMENT, "price": 6000}, explain=False)["warnings"]


def test_models_saved_and_loadable(models_dir):
    assert PurchasePredictor.models_available(models_dir)


def test_heldout_splits_have_no_entity_overlap(small_df):
    for col in ("Customer_ID", "Garment_ID"):
        tr, te = evaluation.grouped_split(small_df, col)
        assert not set(small_df.iloc[tr][col]) & set(small_df.iloc[te][col])


def test_metrics_reported_for_both_splits(purchase):
    m = purchase["metrics"]
    assert set(m["split"]) == {"customer_heldout", "garment_heldout"}
    assert m["roc_auc"].between(0.3, 0.95).all()                # sanity: not perfect, not broken
    assert {"accuracy", "precision", "recall", "f1", "roc_auc", "brier"} <= set(m.columns)
