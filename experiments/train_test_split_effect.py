import logging
import numpy as np
from utils import get_dataset, export_obj_to_json_file
from sklearn.model_selection import train_test_split

from learners.adaboost import adaboost_learner
from learners.dt import decision_tree_learner
from learners.knn import knn_learner
from learners.nn import neural_network_learner
from learners.svm import svm_learner


logger = logging.getLogger()
RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)


def train_test_split_effect():
    """
    Compare the effects of using a train-test split of 0.85/0.15 and 0.75/0.25 and leaving all other params the same for
    all learners for both datasets.

    Output for the paper is a table with train-test split, dataset, learner, accuracy.
    """
    output_dict = dict()
    for dataset in ["census", "phishing"]:
        output_dict[dataset] = dict()
        logger.info(f"===== Running for dataset={dataset}... =====")
        df, X, y = get_dataset(dataset)

        for test_size in [0.15, 0.25]:
            output_dict[dataset][test_size] = dict()
            logger.info(f"===== Running for test_size={test_size}... =====")

            # train-test split:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

            _, dt_metrics_dict = decision_tree_learner(X_train, y_train, X_test, y_test)
            _, adaboost_metrics_dict = adaboost_learner(X_train, y_train, X_test, y_test)
            _, knn_metrics_dict = knn_learner(X_train, y_train, X_test, y_test)
            _, nn_metrics_dict = neural_network_learner(X_train, y_train, X_test, y_test)
            _, svm_metrics_dict = svm_learner(X_train, y_train, X_test, y_test)

            output_dict[dataset][test_size]["dt"] = dt_metrics_dict
            output_dict[dataset][test_size]["adaboost"] = adaboost_metrics_dict
            output_dict[dataset][test_size]["nn"] = nn_metrics_dict
            output_dict[dataset][test_size]["svm"] = svm_metrics_dict

    # write output to a JSON file:
    export_obj_to_json_file(output_dict, "output/metrics/train_test_split_effect.json")
