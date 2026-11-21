import pandas as pd
import pytest

from src import generate_dataset, model_training


@pytest.fixture(scope="session")
def small_df() -> pd.DataFrame:
    """Small reproducible synthetic dataset so tests stay fast."""
    df = generate_dataset.generate(n_customers=400, n_trials=1600, n_garments=120, seed=7)
    df["Trial_Timestamp"] = pd.to_datetime(df["Trial_Timestamp"])
    return df


@pytest.fixture(scope="session")
def cluster_stack(small_df):
    return model_training.train_cluster_stack(small_df)


@pytest.fixture(scope="session")
def purchase(small_df):
    return model_training.train_purchase_models(small_df, fast=True)


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory, cluster_stack, purchase):
    d = tmp_path_factory.mktemp("models")
    model_training.save_artifacts(cluster_stack, purchase, models_dir=d)
    return d
