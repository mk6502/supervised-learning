"""
This file is used to train and test various supervised learners. It also generates all plots used in the paper.
"""
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import plot_tree

from plots import generate_plots
from utils import get_dataset, plt_clear
from supervised_learners import decision_tree_learner, decision_tree_grid_search, neural_network_learner, \
    neural_network_grid_search, adaboost_learner, adaboost_grid_search, svm_learner, knn_learner, knn_grid_search


# set random seed for reproducibility:
RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)


def main():
    """
    Main method that does all the training, hyperparameter tuning, metrics calculations, and plotting.
    """
    for dataset in ["adult", "phishing"]:
        print(f"===== Running for dataset={dataset}... =====")
        df, X, y = get_dataset(dataset)

        # encode X into float for sklearn:
        encoder_X = OrdinalEncoder()
        encoder_X.fit(X)
        X_encoded = encoder_X.transform(X)

        combined_metrics_dict = dict()  # all metrics for plotting this dataset
        for test_size in [0.1, 0.2]:
            combined_metrics_dict[test_size] = dict()  # 2-dimensional dict
            print(f"===== {dataset} Running for test_size={test_size}... =====")

            # same train-test split for all learners:
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=test_size, random_state=RANDOM_STATE)

            # decision tree (no pruning):
            print("===== Decision Tree (no pruning)... =====")
            dt, dt_y_pred, dt_metrics_dict = decision_tree_learner(X_train, y_train, X_test, y_test, max_depth=None)
            combined_metrics_dict[test_size]["dt_metrics_dict"] = dt_metrics_dict
            print()

            # decision tree (with hyperparameter grid search to find max_depth for pruning):
            print("===== Decision Tree (gridsearch for max_depth for pruning)... =====")
            dtp, dtp_y_pred, dtp_metrics_dict = decision_tree_grid_search(X_train, y_train, X_test, y_test)
            combined_metrics_dict[test_size]["dtp_metrics_dict"] = dtp_metrics_dict
            print()

            # export a mostly-useless plot to show complexity of the pruned tree:
            print(f"===== exporting plots/{dataset}_{test_size}_dt_with_pruning.png... =====")
            plt_clear()
            plt.figure(figsize=(30, 30))  # need a lot of room
            plot_tree(dtp, feature_names=df.columns)
            plt.savefig(f"plots/{dataset}_{test_size}_dt_with_pruning.png")
            plt_clear()
            print()

            # neural network:
            print("===== Neural Network (defaults)... =====")
            nn, nn_y_pred, nn_metrics_dict = neural_network_learner(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE)
            combined_metrics_dict[test_size]["nn_metrics_dict"] = nn_metrics_dict
            print()

            # neural network with max_iter=50:
            print("===== Neural Network (max_iter=50)... =====")
            nn50, nn50_y_pred, nn50_metrics_dict = neural_network_learner(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE, max_iter=50)
            combined_metrics_dict[test_size]["nn50_metrics_dict"] = nn50_metrics_dict
            print()

            # neural network with max_iter=100:
            print("===== Neural Network (max_iter=100)... =====")
            nn100, nn100_y_pred, nn100_metrics_dict = neural_network_learner(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE, max_iter=100)
            combined_metrics_dict[test_size]["nn100_metrics_dict"] = nn100_metrics_dict
            print()

            # neural network with max_iter=200:
            print("===== Neural Network (max_iter=200)... =====")
            combined_metrics_dict[test_size]["nn200_metrics_dict"] = nn_metrics_dict  # this is the default
            print()

            # neural network with max_iter=500:
            print("===== Neural Network (max_iter=500)... =====")
            nn500, nn500_y_pred, nn500_metrics_dict = neural_network_learner(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE, max_iter=500)
            combined_metrics_dict[test_size]["nn500_metrics_dict"] = nn500_metrics_dict
            print()

            # neural network grid search for some hyperparameters:
            print("===== Neural Network (grid search for hyperparameters)... =====")
            opt_nn, opt_nn_y_pred, opt_nn_metrics_dict = neural_network_grid_search(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE)
            combined_metrics_dict[test_size]["opt_nn_metrics_dict"] = opt_nn_metrics_dict
            print()

            # AdaBoost:
            print("===== AdaBoost (defaults)... =====")
            ab, ab_y_pred, ab_metrics_dict = adaboost_learner(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE)
            combined_metrics_dict[test_size]["ab_metrics_dict"] = ab_metrics_dict
            print()

            # AdaBoost with n_estimators=10:
            print("===== AdaBoost (n_estimators=10)... =====")
            ab10, ab10_y_pred, ab10_metrics_dict = adaboost_learner(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE, n_estimators=10)
            combined_metrics_dict[test_size]["ab10_metrics_dict"] = ab10_metrics_dict
            print()

            # AdaBoost with n_estimators=50:
            print("===== AdaBoost (n_estimators=50)... =====")
            combined_metrics_dict[test_size]["ab50_metrics_dict"] = ab_metrics_dict  # default is 50
            print()

            # AdaBoost with n_estimators=100:
            print("===== AdaBoost (n_estimators=100)... =====")
            ab100, ab100_y_pred, ab100_metrics_dict = adaboost_learner(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE, n_estimators=100)
            combined_metrics_dict[test_size]["ab100_metrics_dict"] = ab100_metrics_dict
            print()

            # AdaBoost with n_estimators=200:
            print("===== AdaBoost (n_estimators=200)... =====")
            ab200, ab200_y_pred, ab200_metrics_dict = adaboost_learner(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE, n_estimators=200)
            combined_metrics_dict[test_size]["ab200_metrics_dict"] = ab200_metrics_dict
            print()

            # AdaBoost grid search to find optimal n_estimator:
            print("===== AdaBoost (grid search for n_estimators)... =====")
            opt_ab, opt_ab_y_pred, opt_ab_metrics_dict = adaboost_grid_search(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE)
            combined_metrics_dict[test_size]["opt_ab_metrics_dict"] = opt_ab_metrics_dict
            print()

            # SVM (default rbf kernel):
            print("===== SVM (default rbf kernel)... =====")
            svm, svm_y_pred, svm_metrics_dict = svm_learner(X_train, y_train, X_test, y_test, kernel="rbf", random_state=RANDOM_STATE)
            combined_metrics_dict[test_size]["svm_metrics_dict"] = svm_metrics_dict
            print()

            # SVM (default rbf kernel) with max_iter=10:
            print("===== SVM (default rbf kernel and max_iter=10)... =====")
            svm10, svm10_y_pred, svm10_metrics_dict = svm_learner(X_train, y_train, X_test, y_test, kernel="rbf", random_state=RANDOM_STATE, max_iter=10)
            combined_metrics_dict[test_size]["svm10_metrics_dict"] = svm10_metrics_dict
            print()

            # SVM (default rbf kernel) with max_iter=10:
            print("===== SVM (default rbf kernel and max_iter=20)... =====")
            svm20, svm20_y_pred, svm20_metrics_dict = svm_learner(X_train, y_train, X_test, y_test, kernel="rbf", random_state=RANDOM_STATE, max_iter=20)
            combined_metrics_dict[test_size]["svm20_metrics_dict"] = svm20_metrics_dict
            print()

            # SVM (default rbf kernel) with max_iter=50:
            print("===== SVM (default rbf kernel and max_iter=50)... =====")
            svm50, svm50_y_pred, svm50_metrics_dict = svm_learner(X_train, y_train, X_test, y_test, kernel="rbf", random_state=RANDOM_STATE, max_iter=50)
            combined_metrics_dict[test_size]["svm50_metrics_dict"] = svm50_metrics_dict
            print()

            # SVM (default rbf kernel) with max_iter=100:
            print("===== SVM (default rbf kernel and max_iter=100)... =====")
            svm100, svm100_y_pred, svm100_metrics_dict = svm_learner(X_train, y_train, X_test, y_test, kernel="rbf", random_state=RANDOM_STATE, max_iter=100)
            combined_metrics_dict[test_size]["svm100_metrics_dict"] = svm100_metrics_dict
            print()

            # SVM (sigmoid kernel):
            print("===== SVM (sigmoid kernel)... =====")
            svm_sigmoid, svm_sigmoid_y_pred, svm_sigmoid_metrics_dict = svm_learner(X_train, y_train, X_test, y_test, kernel="sigmoid", random_state=RANDOM_STATE)
            combined_metrics_dict[test_size]["svm_sigmoid_metrics_dict"] = svm_sigmoid_metrics_dict
            print()

            # KNN:
            print("===== KNN (defaults)... =====")
            knn, knn_y_pred, knn_metrics_dict = knn_learner(X_train, y_train, X_test, y_test)
            combined_metrics_dict[test_size]["knn_metrics_dict"] = knn_metrics_dict
            print()

            # KNN search for optimal value of K:
            print("===== KNN (grid search for K)... =====")
            opt_knn, opt_knn_y_pred, opt_knn_metrics_dict = knn_grid_search(X_train, y_train, X_test, y_test, fig_filename=f"plots/{dataset}_{test_size}_knn_gridsearch.png")
            combined_metrics_dict[test_size]["opt_knn_metrics_dict"] = opt_knn_metrics_dict
            print()

            print(f"===== finished test_size={test_size}! =====")

        # export combined_metrics_dict to JSON for out-of-band exploration:
        with open(f"output/{dataset}_combined_metrics_dict.json", "w+") as f:
            json.dump(combined_metrics_dict, f)

        print(f"===== finished dataset: {dataset} =====")

    # make plots used in the paper:
    print("===== Done with learners, generating plots... =====")
    generate_plots()


if __name__ == "__main__":
    main()
