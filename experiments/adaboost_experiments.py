import logging
import numpy as np
from sklearn.model_selection import train_test_split
from utils import get_dataset, export_obj_to_json_file, line_graph
from learners.adaboost import adaboost_learner


logger = logging.getLogger()
RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)


def adaboost_basics():
    """
    Run AdaBoost algorithms for both datasets with default parameters. Use an 80/20 train/test split.

    Output for the paper is a table with dataset, accuracy.
    """
    output_dict = dict()
    for dataset in ["census", "phishing"]:
        df, X, y = get_dataset(dataset)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

        logger.info(f"===== AdaBoost ({dataset})... =====")
        _, metrics_dict = adaboost_learner(X_train, y_train, X_test, y_test)
        output_dict[dataset] = metrics_dict

    export_obj_to_json_file(output_dict, f"output/metrics/adaboost_basics.json")


def adaboost_effect_of_n_estimators():
    """
    Run AdaBoost on both datasets, increasing n_estimators. Use an 80/20 train/test split.

    Output for the paper are two plots (one per dataset) of n_estimators vs. accuracy.
    """
    n_estimators = [50, 100, 200, 300, 400, 500]
    datasets = ["census", "phishing"]
    output_dict = dict()

    for dataset in datasets:
        output_dict[dataset] = dict()

        df, X, y = get_dataset(dataset)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

        for n in n_estimators:
            logger.info(f"===== KNN ({dataset}, n_estimators={n})... =====")
            _, metrics_dict = adaboost_learner(X_train, y_train, X_test, y_test, n_estimators=n)
            output_dict[dataset][n] = metrics_dict

    export_obj_to_json_file(output_dict, f"output/metrics/adaboost_effect_of_n_estimators.json")

    # plot:
    for dataset in datasets:
        y = [output_dict[dataset][x]["acc"] for x in n_estimators]
        line_graph(n_estimators, y, "k", "Accuracy", f"{dataset.title()} - KNN - n_estimators vs. Accuracy", f"output/plots/{dataset}_adaboost_n_estimators_vs_acc.png")
