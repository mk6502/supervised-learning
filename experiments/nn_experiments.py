import logging
import numpy as np
from sklearn.model_selection import train_test_split
from utils import get_dataset, export_obj_to_json_file, line_graph
from learners.nn import neural_network_learner, neural_network_grid_search


logger = logging.getLogger()
RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)


def nn_basics():
    """
    Run NN algorithms for both datasets with default parameters. Use an 80/20 train/test split.

    Output for the paper is a table with dataset, accuracy.
    """
    output_dict = dict()
    for dataset in ["census", "phishing"]:
        df, X, y = get_dataset(dataset)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

        logger.info(f"===== NN ({dataset})... =====")
        _, metrics_dict = neural_network_learner(X_train, y_train, X_test, y_test)
        output_dict[dataset] = metrics_dict

    export_obj_to_json_file(output_dict, f"output/metrics/nn_basics.json")


def nn_effect_of_max_iter():
    """
    Run NN on both datasets, increasing the max_iter parameter. Use an 80/20 train/test split.

    Output for the paper are two plots (one per dataset) of max_iter vs. accuracy.
    """
    max_iters = [50, 100, 200, 500, 750]
    datasets = ["census", "phishing"]
    output_dict = dict()

    for dataset in datasets:
        output_dict[dataset] = dict()

        df, X, y = get_dataset(dataset)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

        for max_iter in max_iters:
            logger.info(f"===== NN ({dataset}, max_iter={max_iter})... =====")
            _, metrics_dict = neural_network_learner(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE, max_iter=max_iter)
            output_dict[dataset][max_iter] = metrics_dict

    export_obj_to_json_file(output_dict, f"output/metrics/nn_effect_of_max_iter.json")

    # plot:
    for dataset in datasets:
        y = [output_dict[dataset][x]["acc"] for x in max_iters]
        line_graph(max_iters, y, "max_iter", "Accuracy", f"{dataset.title()} - NN Max Iterations vs. Accuracy", f"output/plots/{dataset}_nn_max_iter_vs_acc.png")


def nn_grid_search_activation_and_alpha():
    """
    Run a grid search on both datasets to tune hyperparameters for alpha and the activation function using the defaults
    for all other parameters.

    Output for the paper is a table with dataset, optimal_alpha, optimal_activation, accuracy.
    """
    possible_activation = ["relu", "logistic"]
    possible_alpha = [0.0001, 0.00001, 0.001]

    output_dict = dict()
    for dataset in ["census", "phishing"]:
        df, X, y = get_dataset(dataset)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

        logger.info(f"===== NN Grid Search ({dataset})... =====")
        optimal_activation, optimal_alpha = neural_network_grid_search(X_train, y_train, X_test, y_test, RANDOM_STATE, possible_activation, possible_alpha)
        _, metrics_dict = neural_network_learner(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE, activation=optimal_activation, alpha=optimal_alpha)
        output_dict[dataset] = {"optimal_alpha": optimal_alpha, "optimal_activation": optimal_activation, "acc": metrics_dict["acc"]}

    export_obj_to_json_file(output_dict, f"output/metrics/nn_grid_search_activation_and_alpha.json")
