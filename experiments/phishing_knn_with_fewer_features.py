import logging
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from learners.knn import knn_learner
from utils import get_dataset, plt_clear, export_obj_to_json_file, line_graph


logger = logging.getLogger()
RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)


def phishing_knn_with_five_features():
    """
    This experiment is designed to show if using 5 features out of 30 will yield an optimal k > 1 with similar
    performance to using all features. An 80/20 train/test split is used.

    Output for the paper are simple accuracy metrics.
    """
    df, X, y = get_dataset("phishing")
    columns_to_keep = ["Shortining_Service", "RightClick", "Iframe", "having_At_Symbol", "Favicon"]
    X = X[columns_to_keep]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    logger.info("=== phishing_knn_with_five_features... ===")
    ks = list(range(1, 51))
    output_dict = dict()

    for k in ks:
        logger.info(f"=== KNN (phishing, k={k})... ===")
        _, metrics_dict = knn_learner(X_train, y_train, X_test, y_test, n_neighbors=k)
        output_dict[k] = metrics_dict

    # plot:
    y = [output_dict[x]["acc"] for x in ks]
    line_graph(ks, y, "k", "Accuracy", f"Phishing - KNN w/ 5 Features - K vs. Accuracy", f"output/plots/phishing_knn_with_five_features_k_vs_acc.png")

    # write metrics to a file:
    export_obj_to_json_file(output_dict, "output/metrics/phishing_knn_with_five_features.json")


def phishing_knn_with_ten_features():
    """
    This experiment is designed to show if using 10 features out of 30 will yield an optimal k > 1 with similar
    performance to using all features. An 80/20 train/test split is used.

    Output for the paper are simple accuracy metrics.
    """
    df, X, y = get_dataset("phishing")
    columns_to_keep = [
        "Shortining_Service", "RightClick", "Iframe", "having_At_Symbol", "Favicon",
        "port", "Google_Index", "SFH", "Page_Rank", "web_traffic",
    ]
    X = X[columns_to_keep]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    logger.info("=== phishing_knn_with_ten_features... ===")
    ks = list(range(1, 51))
    output_dict = dict()

    for k in ks:
        logger.info(f"=== KNN (phishing, k={k})... ===")
        _, metrics_dict = knn_learner(X_train, y_train, X_test, y_test, n_neighbors=k)
        output_dict[k] = metrics_dict

    # plot:
    y = [output_dict[x]["acc"] for x in ks]
    line_graph(ks, y, "k", "Accuracy", f"Phishing - KNN w/ 10 Features - K vs. Accuracy", f"output/plots/phishing_knn_with_ten_features_k_vs_acc.png")

    # write metrics to a file:
    export_obj_to_json_file(output_dict, "output/metrics/phishing_knn_with_ten_features.json")
