import numpy as np

import config as C
from src.cluster_weighting import compute_cluster_weights, demand_potential


def test_six_clusters_and_weights_sum_to_100(cluster_stack, small_df):
    labels = cluster_stack["labels"]
    assert len(labels) == len(small_df) and len(set(labels)) == C.N_CLUSTERS
    w = cluster_stack["weights"]
    assert abs(w["Weight_Pct"].sum() - 100) < 0.1
    assert w["Records"].sum() == len(small_df)
    assert w["Weight_Pct"].is_monotonic_decreasing              # clusters numbered largest -> smallest


def test_k_range_2_to_10(cluster_stack):
    ke = cluster_stack["k_eval"]
    assert list(ke["k"]) == list(range(2, 11))
    assert ke["inertia"].is_monotonic_decreasing
    assert ke["silhouette"].between(-1, 1).all()


def test_weight_formula():
    w = compute_cluster_weights(np.array([0, 0, 0, 1]), 2)
    assert list(w["Weight_Pct"]) == [75.0, 25.0]
    assert list(w["Relative_Demand_Index"]) == [1.5, 0.5]
    assert demand_potential(25, 4, 1.2, 0.8)[1] == "Average"


def test_decision_tree_replicates_clusters(cluster_stack):
    tm = cluster_stack["tree_metrics"]
    assert tm["holdout_accuracy"] > 0.9
    enc, tree = cluster_stack["encoder"], cluster_stack["tree"]
    pred = tree.predict(enc.transform(__import__("pandas").DataFrame(
        [{"Size": 40, "Sleeve_Type": "Full", "Color": "Blue", "Collar_Type": "Regular",
          "Fit": "Slim", "Pattern": "Solid", "Price": 1499}])[C.ATTR_COLS]))
    assert 0 <= int(pred[0]) < C.N_CLUSTERS
