import logging
import numpy as np
from sklearn.model_selection import train_test_split
from utils import get_dataset, export_obj_to_json_file, line_graph
from learners.knn import knn_learner


logger = logging.getLogger()
RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)


def knn_basics():
    """
    Run KNN algorithms for both datasets with default parameters. Use an 80/20 train/test split.

    Output for the paper is a table with dataset, accuracy.
    """
    output_dict = dict()
    for dataset in ["census", "phishing"]:
        df, X, y = get_dataset(dataset)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

        logger.info(f"===== KNN ({dataset})... =====")
        _, metrics_dict = knn_learner(X_train, y_train, X_test, y_test)
        output_dict[dataset] = metrics_dict

    export_obj_to_json_file(output_dict, f"output/metrics/knn_basics.json")


def knn_effect_of_k():
    """
    Run KNN on both datasets, increasing k. Use an 80/20 train/test split.

    Output for the paper are two plots (one per dataset) of k vs. accuracy.
    """
    ks = list(range(1, 51))
    datasets = ["census", "phishing"]
    output_dict = dict()

    for dataset in datasets:
        output_dict[dataset] = dict()

        df, X, y = get_dataset(dataset)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

        for k in ks:
            logger.info(f"===== KNN ({dataset}, k={k})... =====")
            _, metrics_dict = knn_learner(X_train, y_train, X_test, y_test, n_neighbors=k)
            output_dict[dataset][k] = metrics_dict

    export_obj_to_json_file(output_dict, f"output/metrics/knn_effect_of_k.json")

    # plot:
    for dataset in datasets:
        y = [output_dict[dataset][x]["acc"] for x in ks]
        line_graph(ks, y, "k", "Accuracy", f"{dataset.title()} - KNN - K vs. Accuracy", f"output/plots/{dataset}_knn_k_vs_acc.png")
