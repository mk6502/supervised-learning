import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import plot_tree

from utils import get_dataset, plt_clear
from supervised_learners import decision_tree_learner, neural_network_learner, neural_network_grid_search, \
    adaboost_learner, adaboost_grid_search, svm_learner, knn_learner, knn_grid_search


RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)


def main():
    for dataset in ["adult", "phishing"]:
        print(f"running for dataset: {dataset}")

        df, X, y = get_dataset(dataset)

        # encode X into float for sklearn:
        encoder_X = OrdinalEncoder()
        encoder_X.fit(X)
        X_encoded = encoder_X.transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.20, random_state=RANDOM_STATE)

        # decision tree (no pruning):
        dt, dt_y_pred, dt_acc, dt_fpr, dt_tpr, dt_thresholds, dt_auc, dt_precision, dt_recall, dt_average_precision = \
            decision_tree_learner(X_train, y_train, X_test, y_test, max_depth=None)
        print("====================\n")

        # NOTE: there's no point in plotting this tree - it's insanely complex!

        # decision tree (with pruning):
        dtp, dtp_y_pred, dtp_acc, dtp_fpr, dtp_tpr, dtp_thresholds, dtp_auc, dtp_precision, dtp_recall, dtp_average_precision = \
            decision_tree_learner(X_train, y_train, X_test, y_test, max_depth=5)

        plt_clear()
        plt.figure(figsize=(12, 12))
        plot_tree(dtp, feature_names=df.columns)
        plt.savefig(f"plots/{dataset}_dt_with_pruning.png", dpi=1000)
        plt_clear()
        print("====================\n")

        # neural network:
        nn, nn_y_pred, nn_acc, nn_fpr, nn_tpr, nn_thresholds, nn_auc, nn_precision, nn_recall, nn_average_precision = \
            neural_network_learner(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE)
        print("====================\n")

        # neural network grid search for some hyperparameters:
        opt_nn, opt_nn_y_pred, opt_nn_acc, opt_nn_fpr, opt_nn_tpr, opt_nn_thresholds, opt_nn_auc, opt_nn_precision, opt_nn_recall, opt_nn_average_precision = \
            neural_network_grid_search(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE)
        print("====================\n")

        # boosting:
        ab, ab_y_pred, ab_acc, ab_fpr, ab_tpr, ab_thresholds, ab_auc, ab_precision, ab_recall, ab_average_precision = \
            adaboost_learner(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE)
        print("====================\n")

        # boosting grid search to find optimal n_estimator:
        opt_ab, opt_ab_y_pred, opt_ab_acc, opt_ab_fpr, opt_ab_tpr, opt_ab_thresholds, opt_ab_auc, opt_ab_precision, opt_ab_recall, opt_ab_average_precision = \
            adaboost_grid_search(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE)
        print("====================\n")

        # SVM (polynomial kernel):
        svm, svm_y_pred, svm_acc, svm_fpr, svm_tpr, svm_thresholds, svm_auc, svm_precision, svm_recall, svm_average_precision = \
            svm_learner(X_train, y_train, X_test, y_test, kernel="poly", random_state=RANDOM_STATE)
        print("====================\n")

        # SVM (sigmoid kernel):
        svm_sigmoid, svm_sigmoid_y_pred, svm_sigmoid_acc, svm_sigmoid_fpr, svm_sigmoid_tpr, svm_sigmoid_thresholds, svm_sigmoid_auc, svm_sigmoid_precision, svm_sigmoid_recall, svm_sigmoid_average_precision = \
            svm_learner(X_train, y_train, X_test, y_test, kernel="sigmoid", random_state=RANDOM_STATE)
        print("====================\n")

        # KNN:
        knn, knn_y_pred, knn_acc, knn_fpr, knn_tpr, knn_thresholds, knn_auc, knn_precision, knn_recall, knn_average_precision = \
            knn_learner(X_train, y_train, X_test, y_test)
        print("====================\n")

        # KNN search for optimal value of K:
        opt_knn, opt_knn_y_pred, opt_knn_acc, opt_knn_fpr, opt_knn_tpr, opt_knn_thresholds, opt_knn_auc, opt_knn_precision, opt_knn_recall, opt_knn_average_precision = \
            knn_grid_search(X_train, y_train, X_test, y_test, fig_filename=f"plots/{dataset}_knn_gridsearch.png")
        print("====================\n")

    print("DONE!")


if __name__ == "__main__":
    main()
