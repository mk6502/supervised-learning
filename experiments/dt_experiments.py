import logging
import numpy as np
from sklearn.model_selection import train_test_split
from utils import get_dataset, export_obj_to_json_file, line_graph
from learners.dt import decision_tree_learner


logger = logging.getLogger()
RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)


def dt_basics():
    """
    Run decision tree algorithms for both datasets with default parameters. Use an 80/20 train/test split.

    Output for the paper is a table with dataset, accuracy and two useless plots of the trees.
    """
    output_dict = dict()
    for dataset in ["census", "phishing"]:
        df, X, y = get_dataset(dataset)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

        logger.info(f"===== DT ({dataset})... =====")
        _, metrics_dict = decision_tree_learner(X_train, y_train, X_test, y_test)
        output_dict[dataset] = metrics_dict

    export_obj_to_json_file(output_dict, f"output/metrics/dt_basics.json")


def dt_effect_of_max_depth():
    """
    Run DT on both datasets, increasing the max_depth parameter. Use an 80/20 train/test split.

    Output for the paper are two plots (one per dataset) of max_depth vs. accuracy.
    """
    max_depths = [1, 3, 5, 10, 20, 50, 100]
    datasets = ["census", "phishing"]
    output_dict = dict()

    for dataset in datasets:
        output_dict[dataset] = dict()

        df, X, y = get_dataset(dataset)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

        for max_depth in max_depths:
            logger.info(f"===== DT ({dataset}, max_depth={max_depth})... =====")
            _, metrics_dict = decision_tree_learner(X_train, y_train, X_test, y_test, max_depth=max_depth)
            output_dict[dataset][max_depth] = metrics_dict

    export_obj_to_json_file(output_dict, f"output/metrics/dt_effect_of_max_depth.json")

    # plot:
    for dataset in datasets:
        y = [output_dict[dataset][x]["acc"] for x in max_depths]
        line_graph(max_depths, y, "max_depth", "Accuracy", f"{dataset.title()} - DT Max Iterations vs. Accuracy", f"output/plots/{dataset}_dt_max_depth_vs_acc.png")
