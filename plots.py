"""
This file is used to generate plots used in the paper.
"""
import json
from utils import accuracy_test_size_bar_chart, accuracy_two_learners_bar_chart


def make_beautiful_plots(dataset, combined_metrics_dict):
    """
    Make plots used in the paper. `combined_metrics_dict` is for a single dataset.
    """
    test_sizes = list(combined_metrics_dict.keys())  # [0.1, 0.2]

    # Accuracy for both test_sizes for pruned decision tree:
    accuracy_test_size_bar_chart(combined_metrics_dict, "dtp_metrics_dict", test_sizes, "Pruned Decision Tree Accuracy with different test sizes", f"plots/{dataset}_dtp_test_size_acc.png")

    # Accuracy for both test_sizes for (default parameter) NN:
    accuracy_test_size_bar_chart(combined_metrics_dict, "nn_metrics_dict", test_sizes, "Neural Network Accuracy with different test sizes", f"plots/{dataset}_nn_test_size_acc.png")

    # Accuracy for both test_sizes for (default parameter) AdaBoost:
    accuracy_test_size_bar_chart(combined_metrics_dict, "ab_metrics_dict", test_sizes, "AdaBoost Accuracy with different test sizes", f"plots/{dataset}_ab_test_size_acc.png")

    # Accuracy for both test_sizes for (default parameter) AdaBoost:
    accuracy_test_size_bar_chart(combined_metrics_dict, "svm_metrics_dict", test_sizes, "SVM Accuracy with different test sizes", f"plots/{dataset}_svm_test_size_acc.png")

    # Accuracy for both test_sizes for (default parameter) AdaBoost:
    accuracy_test_size_bar_chart(combined_metrics_dict, "knn_metrics_dict", test_sizes, "KNN Accuracy with different test sizes", f"plots/{dataset}_knn_test_size_acc.png")

    # Accuracy when using default parameters vs. grid search - DT:
    accuracy_two_learners_bar_chart(combined_metrics_dict, 0.2, "dt_metrics_dict", "dtp_metrics_dict", "DT Defaults", "DT Optimized", "DT Defaults vs. Optimized Accuracy", f"plots/{dataset}_dt_default_vs_opt_acc.png")

    # Accuracy when using default parameters vs. grid search - NN:
    accuracy_two_learners_bar_chart(combined_metrics_dict, 0.2, "nn_metrics_dict", "opt_nn_metrics_dict", "NN Defaults", "NN Optimized", "NN Defaults vs. Optimized Accuracy", f"plots/{dataset}_nn_default_vs_opt_acc.png")

    # TODO: Accuracy vs. number of iterations (n_estimator) for AdaBoost line:

    # TODO: Accuracy vs. number of iterations (max_iter) for NN line:
    # TODO: Accuracy vs. number of iterations (max_iter) for SVM line:
    return


def generate_plots():
    for dataset in ["adult", "phishing"]:
        try:
            with open(f"outputs/{dataset}_combined_metrics_dict.json") as f:
                combined_metrics_dict = json.load(f)
        except:
            raise Exception(f"Unable to load outputs/{dataset}_combined_metrics_dict.json - did you run main.py?")

        make_beautiful_plots(dataset, combined_metrics_dict)


if __name__ == "__main__":
    generate_plots()
