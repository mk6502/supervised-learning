"""
This file is used to generate plots used in the paper.
"""
import json
from utils import accuracy_test_size_bar_chart, accuracy_two_learners_bar_chart, accuracy_vs_param_line_chart


def make_beautiful_plots(dataset, combined_metrics_dict):
    """
    Make plots used in the paper. `combined_metrics_dict` is for a single dataset.
    """
    test_sizes = list(combined_metrics_dict.keys())  # [0.1, 0.2]
    test_size = str(0.2)  # only graph one of the test sizes, don't need both

    # Accuracy for both test_sizes for pruned decision tree:
    accuracy_test_size_bar_chart(combined_metrics_dict, "dtp_metrics_dict", test_sizes, "Pruned Decision Tree Accuracy with different test sizes", f"output/plots/{dataset}_dtp_test_size_acc.png")

    # Accuracy for both test_sizes for (default parameter) NN:
    accuracy_test_size_bar_chart(combined_metrics_dict, "nn_metrics_dict", test_sizes, "Neural Network Accuracy with different test sizes", f"output/plots/{dataset}_nn_test_size_acc.png")

    # Accuracy for both test_sizes for (default parameter) AdaBoost:
    accuracy_test_size_bar_chart(combined_metrics_dict, "ab_metrics_dict", test_sizes, "AdaBoost Accuracy with different test sizes", f"output/plots/{dataset}_ab_test_size_acc.png")

    # Accuracy for both test_sizes for (default parameter) AdaBoost:
    accuracy_test_size_bar_chart(combined_metrics_dict, "svm_metrics_dict", test_sizes, "SVM Accuracy with different test sizes", f"output/plots/{dataset}_svm_test_size_acc.png")

    # Accuracy for both test_sizes for (default parameter) AdaBoost:
    accuracy_test_size_bar_chart(combined_metrics_dict, "knn_metrics_dict", test_sizes, "KNN Accuracy with different test sizes", f"output/plots/{dataset}_knn_test_size_acc.png")

    # Accuracy when using default parameters vs. grid search - DT:
    accuracy_two_learners_bar_chart(combined_metrics_dict, test_size, "dt_metrics_dict", "dtp_metrics_dict", "DT Defaults", "DT Optimized", "DT Defaults vs. Optimized Accuracy", f"output/plots/{dataset}_dt_default_vs_opt_acc.png")

    # Accuracy when using default parameters vs. grid search - NN:
    accuracy_two_learners_bar_chart(combined_metrics_dict, test_size, "nn_metrics_dict", "opt_nn_metrics_dict", "NN Defaults", "NN Optimized", "NN Defaults vs. Optimized Accuracy", f"output/plots/{dataset}_nn_default_vs_opt_acc.png")

    # Accuracy vs. number of iterations (n_estimator) for AdaBoost line:
    x = [10, 50, 100, 200]
    y = [combined_metrics_dict[test_size]["ab10_metrics_dict"]["acc"], combined_metrics_dict[test_size]["ab50_metrics_dict"]["acc"], combined_metrics_dict[test_size]["ab100_metrics_dict"]["acc"], combined_metrics_dict[test_size]["ab200_metrics_dict"]["acc"]]
    accuracy_vs_param_line_chart(x, y, "n_estimator", "Accuracy", "AdaBoost Iterations vs. Accuracy", f"output/plots/{dataset}_ab_iter_vs_acc.png")

    # Accuracy vs. max number of iterations (max_iter) for NN line:
    x = [50, 100, 200, 500]
    y = [combined_metrics_dict[test_size]["nn50_metrics_dict"]["acc"], combined_metrics_dict[test_size]["nn100_metrics_dict"]["acc"], combined_metrics_dict[test_size]["nn200_metrics_dict"]["acc"], combined_metrics_dict[test_size]["nn500_metrics_dict"]["acc"]]
    accuracy_vs_param_line_chart(x, y, "max_iter", "Accuracy", "NN Max Iterations vs. Accuracy", f"output/plots/{dataset}_nn_max_iter_vs_acc.png")

    # Accuracy vs. max number of iterations (max_iter) for SVM line:
    x = [50, 100, 200, 500]
    y = [combined_metrics_dict[test_size]["svm50_metrics_dict"]["acc"], combined_metrics_dict[test_size]["svm100_metrics_dict"]["acc"], combined_metrics_dict[test_size]["svm200_metrics_dict"]["acc"], combined_metrics_dict[test_size]["svm500_metrics_dict"]["acc"]]
    accuracy_vs_param_line_chart(x, y, "max_iter", "Accuracy", "SVM Max Iterations vs. Accuracy", f"output/plots/{dataset}_svm_max_iter_vs_acc.png")


def generate_plots():
    """
    Main method for this file.
    """
    for dataset in ["adult", "phishing"]:
        try:
            with open(f"output/{dataset}_combined_metrics_dict.json") as f:
                combined_metrics_dict = json.load(f)
        except:
            raise Exception(f"Unable to load outputs/{dataset}_combined_metrics_dict.json - did you run main.py?")

        make_beautiful_plots(dataset, combined_metrics_dict)


if __name__ == "__main__":
    generate_plots()
