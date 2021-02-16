"""
This file is used to train and test various supervised learners. It also generates all plots used in the paper.
"""
import json
import logging
import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import plot_tree

from plots import generate_plots
from utils import get_dataset, plt_clear
from supervised_learners import decision_tree_learner, decision_tree_grid_search, neural_network_learner, \
    neural_network_grid_search, adaboost_learner, svm_learner, knn_learner, knn_grid_search


# set random seed for reproducibility:
RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)


# configure logging to stdout and file:
logging.getLogger("matplotlib").setLevel(level=logging.CRITICAL)  # quiet down matplotlib warnings
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.addHandler(logging.FileHandler("output/main.log"))


def phishing_decision_tree_with_fewer_features():
    """
    This experiment is deisnged to show if using 5 features out of 30 will yield a tree with similar performance to using all features.
    """
    dataset = "phishing"
    test_size = 0.2

    df, X, y = get_dataset(dataset)
    columns_to_keep = ["Shortining_Service", "RightClick", "Iframe", "having_At_Symbol", "Favicon"]
    df = df[columns_to_keep]
    X = X[columns_to_keep]


    combined_metrics_dict = dict()  # all metrics for plotting this dataset
    combined_metrics_dict[test_size] = dict()  # 2-dimensional dict
    logger.info(f"===== {dataset} Running for test_size={test_size}... =====")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)

    # decision tree (with hyperparameter grid search to find max_depth for pruning):
    logger.info("===== Decision Tree (gridsearch for max_depth for pruning)... =====")
    dtp, dtp_y_pred, dtp_metrics_dict = decision_tree_grid_search(X_train, y_train, X_test, y_test)
    combined_metrics_dict[test_size]["dtp_metrics_dict"] = dtp_metrics_dict

    # export a mostly-useless plot to show complexity of the pruned tree:
    logger.info(f"===== exporting output/plots/{dataset}_{test_size}_fewer_features_dt_with_pruning.png... =====")
    plt_clear()
    plt.figure(figsize=(30, 30))  # need a lot of room
    plot_tree(dtp, feature_names=df.columns)
    plt.savefig(f"output/plots/{dataset}_{test_size}_fewer_features_dt_with_pruning.png")
    plt_clear()


def main(algo):
    """
    Main method that does all the training, hyperparameter tuning, metrics calculations, and plotting.
    """
    for dataset in ["census", "phishing"]:
        logger.info(f"===== Running for dataset={dataset}... =====")
        df, X, y = get_dataset(dataset)

        # encode X into float for sklearn:
        encoder_X = OrdinalEncoder()
        encoder_X.fit(X)
        X_encoded = encoder_X.transform(X)

        combined_metrics_dict = dict()  # all metrics for plotting this dataset
        for test_size in [0.1, 0.2]:
            combined_metrics_dict[test_size] = dict()  # 2-dimensional dict
            logger.info(f"===== {dataset} Running for test_size={test_size}... =====")

            # same train-test split for all learners:
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=test_size, random_state=RANDOM_STATE)

            if algo in ["dt", "all"]:
                # decision tree (no pruning):
                logger.info("===== Decision Tree (no pruning)... =====")
                dt, _, dt_metrics_dict = decision_tree_learner(X_train, y_train, X_test, y_test, max_depth=None)
                combined_metrics_dict[test_size]["dt_metrics_dict"] = dt_metrics_dict

                # decision tree (with hyperparameter grid search to find max_depth for pruning):
                logger.info("===== Decision Tree (gridsearch for max_depth for pruning)... =====")
                dtp, _, dtp_metrics_dict = decision_tree_grid_search(X_train, y_train, X_test, y_test)
                combined_metrics_dict[test_size]["dtp_metrics_dict"] = dtp_metrics_dict

                # export a mostly-useless plot to show complexity of the pruned tree:
                logger.info(f"===== exporting output/plots/{dataset}_{test_size}_dt_with_pruning.png... =====")
                plt_clear()
                plt.figure(figsize=(30, 30))  # need a lot of room
                plot_tree(dtp, feature_names=df.columns)
                plt.savefig(f"output/plots/{dataset}_{test_size}_dt_with_pruning.png")
                plt_clear()

                # run the phishing_decision_tree_with_fewer_features experiment:
                if test_size == 0.2 and dataset == "phishing":
                    phishing_decision_tree_with_fewer_features()

            if algo in ["nn", "all"]:
                # neural network:
                logger.info("===== Neural Network (defaults)... =====")
                _, _, nn_metrics_dict = neural_network_learner(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE)
                combined_metrics_dict[test_size]["nn_metrics_dict"] = nn_metrics_dict

                # neural network with max_iter=50:
                logger.info("===== Neural Network (max_iter=50)... =====")
                _, _, nn50_metrics_dict = neural_network_learner(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE, max_iter=50)
                combined_metrics_dict[test_size]["nn50_metrics_dict"] = nn50_metrics_dict

                # neural network with max_iter=100:
                logger.info("===== Neural Network (max_iter=100)... =====")
                _, _, nn100_metrics_dict = neural_network_learner(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE, max_iter=100)
                combined_metrics_dict[test_size]["nn100_metrics_dict"] = nn100_metrics_dict

                # neural network with max_iter=200:
                combined_metrics_dict[test_size]["nn200_metrics_dict"] = nn_metrics_dict  # NN default is 200

                # neural network with max_iter=500:
                logger.info("===== Neural Network (max_iter=500)... =====")
                _, _, nn500_metrics_dict = neural_network_learner(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE, max_iter=500)
                combined_metrics_dict[test_size]["nn500_metrics_dict"] = nn500_metrics_dict

                # neural network with max_iter=750:
                logger.info("===== Neural Network (max_iter=750)... =====")
                _, _, nn750_metrics_dict = neural_network_learner(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE, max_iter=750)
                combined_metrics_dict[test_size]["nn750_metrics_dict"] = nn750_metrics_dict

                # neural network grid search for some hyperparameters:
                logger.info("===== Neural Network (grid search for hyperparameters)... =====")
                _, _, opt_nn_metrics_dict = neural_network_grid_search(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE)
                combined_metrics_dict[test_size]["opt_nn_metrics_dict"] = opt_nn_metrics_dict

            if algo in ["ab", "all"]:
                # AdaBoost:
                logger.info("===== AdaBoost (defaults)... =====")
                _, _, ab_metrics_dict = adaboost_learner(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE)
                combined_metrics_dict[test_size]["ab_metrics_dict"] = ab_metrics_dict

                # AdaBoost with n_estimators=50:
                combined_metrics_dict[test_size]["ab50_metrics_dict"] = ab_metrics_dict  # AdaBoost default is 50

                # AdaBoost with n_estimators=100:
                logger.info("===== AdaBoost (n_estimators=100)... =====")
                _, _, ab100_metrics_dict = adaboost_learner(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE, n_estimators=100)
                combined_metrics_dict[test_size]["ab100_metrics_dict"] = ab100_metrics_dict

                # AdaBoost with n_estimators=200:
                logger.info("===== AdaBoost (n_estimators=200)... =====")
                _, _, ab200_metrics_dict = adaboost_learner(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE, n_estimators=200)
                combined_metrics_dict[test_size]["ab200_metrics_dict"] = ab200_metrics_dict

                # AdaBoost with n_estimators=300:
                logger.info("===== AdaBoost (n_estimators=300)... =====")
                _, _, ab300_metrics_dict = adaboost_learner(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE, n_estimators=300)
                combined_metrics_dict[test_size]["ab300_metrics_dict"] = ab300_metrics_dict

                # AdaBoost with n_estimators=400:
                logger.info("===== AdaBoost (n_estimators=400)... =====")
                _, _, ab400_metrics_dict = adaboost_learner(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE, n_estimators=400)
                combined_metrics_dict[test_size]["ab400_metrics_dict"] = ab400_metrics_dict

                # AdaBoost with n_estimators=500:
                logger.info("===== AdaBoost (n_estimators=500)... =====")
                _, _, ab500_metrics_dict = adaboost_learner(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE, n_estimators=500)
                combined_metrics_dict[test_size]["ab500_metrics_dict"] = ab500_metrics_dict

            if algo in ["svm", "all"]:
                # SVM (default rbf kernel):
                logger.info("===== SVM (default rbf kernel - no max_iter limit)... =====")
                _, _, svm_metrics_dict = svm_learner(X_train, y_train, X_test, y_test, kernel="rbf", random_state=RANDOM_STATE)
                combined_metrics_dict[test_size]["svm_metrics_dict"] = svm_metrics_dict

                # SVM (default rbf kernel) with max_iter=50:
                logger.info("===== SVM (default rbf kernel and max_iter=50)... =====")
                _, _, svm50_metrics_dict = svm_learner(X_train, y_train, X_test, y_test, kernel="rbf", random_state=RANDOM_STATE, max_iter=50)
                combined_metrics_dict[test_size]["svm50_metrics_dict"] = svm50_metrics_dict

                # SVM (default rbf kernel) with max_iter=100:
                logger.info("===== SVM (default rbf kernel and max_iter=100)... =====")
                _, _, svm100_metrics_dict = svm_learner(X_train, y_train, X_test, y_test, kernel="rbf", random_state=RANDOM_STATE, max_iter=100)
                combined_metrics_dict[test_size]["svm100_metrics_dict"] = svm100_metrics_dict

                # SVM (default rbf kernel) with max_iter=200:
                logger.info("===== SVM (default rbf kernel and max_iter=200)... =====")
                _, _, svm200_metrics_dict = svm_learner(X_train, y_train, X_test, y_test, kernel="rbf", random_state=RANDOM_STATE, max_iter=200)
                combined_metrics_dict[test_size]["svm200_metrics_dict"] = svm200_metrics_dict

                # SVM (default rbf kernel) with max_iter=500:
                logger.info("===== SVM (default rbf kernel and max_iter=500)... =====")
                _, _, svm500_metrics_dict = svm_learner(X_train, y_train, X_test, y_test, kernel="rbf", random_state=RANDOM_STATE, max_iter=500)
                combined_metrics_dict[test_size]["svm500_metrics_dict"] = svm500_metrics_dict

                # SVM (default rbf kernel) with max_iter=750:
                logger.info("===== SVM (default rbf kernel and max_iter=750)... =====")
                _, _, svm750_metrics_dict = svm_learner(X_train, y_train, X_test, y_test, kernel="rbf", random_state=RANDOM_STATE, max_iter=750)
                combined_metrics_dict[test_size]["svm750_metrics_dict"] = svm750_metrics_dict

                # SVM (sigmoid kernel):
                logger.info("===== SVM (sigmoid kernel - no max_iter limit)... =====")
                _, _, svm_sigmoid_metrics_dict = svm_learner(X_train, y_train, X_test, y_test, kernel="sigmoid", random_state=RANDOM_STATE)
                combined_metrics_dict[test_size]["svm_sigmoid_metrics_dict"] = svm_sigmoid_metrics_dict

            if algo in ["knn", "all"]:
                # KNN:
                logger.info("===== KNN (defaults)... =====")
                knn, knn_y_pred, knn_metrics_dict = knn_learner(X_train, y_train, X_test, y_test)
                combined_metrics_dict[test_size]["knn_metrics_dict"] = knn_metrics_dict

                # KNN search for optimal value of K:
                logger.info("===== KNN (grid search for K)... =====")
                opt_knn, opt_knn_y_pred, opt_knn_metrics_dict = knn_grid_search(X_train, y_train, X_test, y_test, fig_filename=f"output/plots/{dataset}_{test_size}_knn_gridsearch.png")
                combined_metrics_dict[test_size]["opt_knn_metrics_dict"] = opt_knn_metrics_dict

            logger.info(f"===== finished test_size={test_size}! =====")

        # export combined_metrics_dict to JSON for out-of-band exploration:
        with open(f"output/{dataset}_combined_metrics_dict.json", "w+") as f:
            json.dump(combined_metrics_dict, f)

        logger.info(f"===== finished dataset: {dataset} =====")

    if algo == "all":
        # make plots used in the paper:
        logger.info("===== Done with learners, generating plots... =====")
        generate_plots(algo)


if __name__ == "__main__":
    algo = "all"
    if len(sys.argv) > 1:
        algo = sys.argv[1]
    main(algo)
