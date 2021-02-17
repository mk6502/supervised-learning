import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from learners.dt import decision_tree_learner
from utils import get_dataset, plt_clear, export_obj_to_json_file


logger = logging.getLogger()
RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)


def phishing_decision_tree_with_five_features():
    """
    This experiment is designed to show if using 5 features out of 30 will yield a tree with similar performance to
    using all features. These five were selected at random. An 80/20 train/test split is used.

    Output for the paper are simple accuracy metrics and a PNG of the tree.
    """
    df, X, y = get_dataset("phishing")
    columns_to_keep = ["Shortining_Service", "RightClick", "Iframe", "having_At_Symbol", "Favicon"]
    df = df[columns_to_keep]
    X = X[columns_to_keep]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # decision tree:
    logger.info("=== phishing_decision_tree_with_five_features... ===")
    dt, dt_metrics_dict = decision_tree_learner(X_train, y_train, X_test, y_test, max_depth=None)

    # write metrics to a file:
    export_obj_to_json_file(dt_metrics_dict, "output/metrics/phishing_decision_tree_with_five_features.json")

    # export a mostly-useless plot to show complexity of the pruned tree:
    filename = "output/plots/phishing_decision_tree_with_five_features.png"
    logger.info(f"=== exporting {filename}... ===")
    plt_clear()
    plt.figure(figsize=(30, 30))  # need a lot of room
    plot_tree(dt, feature_names=df.columns)
    plt.savefig(filename)
    plt_clear()


def phishing_decision_tree_with_ten_features():
    """
    This experiment is designed to show if using 10 features out of 30 will yield a tree with similar performance to
    using all features. These ten were selected at random. An 80/20 train/test split is used.

    Output for the paper are simple accuracy metrics and a PNG of the tree.
    """
    df, X, y = get_dataset("phishing")
    columns_to_keep = [
        "Shortining_Service", "RightClick", "Iframe", "having_At_Symbol", "Favicon",
        "port", "Google_Index", "SFH", "Page_Rank", "web_traffic",
    ]

    df = df[columns_to_keep]
    X = X[columns_to_keep]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # decision tree:
    logger.info("=== phishing_decision_tree_with_ten_features... ===")
    dt, dt_metrics_dict = decision_tree_learner(X_train, y_train, X_test, y_test, max_depth=None)

    # write metrics to a file:
    export_obj_to_json_file(dt_metrics_dict, "output/metrics/phishing_decision_tree_with_ten_features.json")

    # export a mostly-useless plot to show complexity of the pruned tree:
    filename = "output/plots/phishing_decision_tree_with_ten_features.png"
    logger.info(f"===== exporting {filename}... =====")
    plt_clear()
    plt.figure(figsize=(30, 30))  # need a lot of room
    plot_tree(dt, feature_names=df.columns)
    plt.savefig(filename)
    plt_clear()
