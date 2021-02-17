import logging
import numpy as np
from sklearn.model_selection import train_test_split
from utils import get_dataset, export_obj_to_json_file, line_graph
from learners.svm import svm_learner


logger = logging.getLogger()
RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)


def svm_rbf_vs_sigmoid():
    """
    Run SVM using the default RBF kernel function and then a sigmoid kernel function. Compare the outputs. Use an 80/20
    train/test split.

    Output for the paper is a table with dataset, kernel function, and accuracy.
    """
    output_dict = dict()
    for dataset in ["census", "phishing"]:
        output_dict[dataset] = dict()
        df, X, y = get_dataset(dataset)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

        for kernel in ["rbf", "sigmoid"]:
            logger.info(f"=== SVM ({dataset}, {kernel})... ===")
            _, svm_metrics_dict = svm_learner(X_train, y_train, X_test, y_test, kernel=kernel, random_state=RANDOM_STATE)
            output_dict[dataset][kernel] = svm_metrics_dict["acc"]

    export_obj_to_json_file(output_dict, f"output/metrics/svm_rbf_vs_sigmoid.json")


def svm_effect_of_max_iter():
    """
    Run SVM using the default RBF kernel on both datasets, increasing the max_iter parameter. Use an 80/20 train/test
    split.

    Output for the paper are two plots (one per dataset) of max_iter vs. accuracy.
    """
    max_iters = [50, 100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    datasets = ["census", "phishing"]
    output_dict = dict()

    for dataset in datasets:
        output_dict[dataset] = dict()

        df, X, y = get_dataset(dataset)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

        for max_iter in max_iters:
            logger.info(f"=== SVM ({dataset}, max_iter={max_iter})... ===")
            _, metrics_dict = svm_learner(X_train, y_train, X_test, y_test, kernel="rbf", random_state=RANDOM_STATE, max_iter=max_iter)
            output_dict[dataset][max_iter] = metrics_dict

    export_obj_to_json_file(output_dict, f"output/metrics/svm_effect_of_max_iter.json")

    # plot:
    for dataset in datasets:
        y = [output_dict[dataset][x]["acc"] for x in max_iters]
        line_graph(max_iters, y, "max_iter", "Accuracy", f"{dataset.title()} - SVM Max Iterations vs. Accuracy", f"output/plots/{dataset}_svm_max_iter_vs_acc.png")
