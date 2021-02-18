"""
This file is used to train and test various supervised learners. It also generates all plots used in the paper.
"""
import logging
import sys
import numpy as np

from experiments.adaboost_experiments import adaboost_basics, adaboost_effect_of_n_estimators
from experiments.dt_experiments import dt_basics, dt_effect_of_max_depth
from experiments.knn_experiments import knn_basics, knn_effect_of_k
from experiments.nn_experiments import nn_grid_search_activation_and_alpha, nn_effect_of_max_iter, nn_basics
from experiments.svm_experiments import svm_rbf_vs_sigmoid, svm_effect_of_max_iter
from experiments.phishing_decision_tree_with_fewer_features import phishing_decision_tree_with_five_features, phishing_decision_tree_with_ten_features
from experiments.phishing_knn_with_fewer_features import phishing_knn_with_five_features, phishing_knn_with_ten_features
from experiments.train_test_split_effect import train_test_split_effect


# set random seed for reproducibility:
RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)


# configure logging to stdout and file:
logging.getLogger("matplotlib").setLevel(level=logging.CRITICAL)  # quiet down matplotlib warnings
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def main():
    """
    Main entry point. Runs everything.
    """
    # Decision tree experiments:
    dt_basics()
    dt_effect_of_max_depth()

    # KNN experiments:
    knn_basics()
    knn_effect_of_k()
    phishing_knn_with_five_features()
    phishing_knn_with_ten_features()

    # AdaBoost experiments:
    adaboost_basics()
    adaboost_effect_of_n_estimators()

    # SVM experiments:
    svm_rbf_vs_sigmoid()
    svm_effect_of_max_iter()

    # NN experiments:
    nn_basics()
    nn_effect_of_max_iter()
    nn_grid_search_activation_and_alpha()

    # Feature selection:
    phishing_decision_tree_with_five_features()
    phishing_decision_tree_with_ten_features()

    # Effect of an 85/15 vs. 75/25 train/test split:
    train_test_split_effect()


if __name__ == "__main__":
    main()
