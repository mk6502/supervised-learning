import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import plot_tree

from utils import get_dataset, plt_clear
from supervised_learners import decision_tree_learner, decision_tree_grid_search, neural_network_learner, \
    neural_network_grid_search, adaboost_learner, adaboost_grid_search, svm_learner, knn_learner, knn_grid_search


# set random seed for reproducibility:
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

        # same train-test split for all learners:
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.20, random_state=RANDOM_STATE)

        # decision tree (no pruning):
        print("===== Decision Tree (no pruning) =====")
        dt, dt_y_pred, dt_metrics_dict = decision_tree_learner(X_train, y_train, X_test, y_test, max_depth=None)
        print()

        # decision tree (with hyperparameter grid search to find max_depth for pruning):
        print("===== Decision Tree (gridsearch for max_depth for pruning) =====")
        dtp, dtp_y_pred, dtp_metrics_dict = decision_tree_grid_search(X_train, y_train, X_test, y_test)
        print()

        print(f"===== exporting plots/{dataset}_dt_with_pruning.png =====")
        plt_clear()
        plt.figure(figsize=(30, 30))  # need a lot of room
        plot_tree(dtp, feature_names=df.columns)
        plt.savefig(f"plots/{dataset}_dt_with_pruning.png")
        plt_clear()
        print()

        # neural network:
        print("===== Decision Tree (no pruning) =====")
        nn, nn_y_pred, nn_metrics_dict = neural_network_learner(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE)
        print()

        # neural network grid search for some hyperparameters:
        print("===== Neural Network (grid search for hyperparameters) =====")
        opt_nn, opt_nn_y_pred, opt_nn_metrics_dict = neural_network_grid_search(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE)
        print()

        # boosting:
        print("===== AdaBoost (defaults) =====")
        ab, ab_y_pred, ab_metrics_dict = adaboost_learner(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE)
        print()

        # boosting grid search to find optimal n_estimator:
        print("===== AdaBoost (grid search for n_estimator) =====")
        opt_ab, opt_ab_y_pred, opt_ab_metrics_dict = adaboost_grid_search(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE)
        print()

        # SVM (polynomial kernel):
        print("===== SVM (polynomial kernel) =====")
        svm, svm_y_pred, svm_metrics_dict = svm_learner(X_train, y_train, X_test, y_test, kernel="poly", random_state=RANDOM_STATE)
        print()

        # SVM (sigmoid kernel):
        print("===== SVM (sigmoid kernel) =====")
        svm_sigmoid, svm_sigmoid_y_pred, svm_sigmoid_metrics_dict = \
            svm_learner(X_train, y_train, X_test, y_test, kernel="sigmoid", random_state=RANDOM_STATE)
        print()

        # KNN:
        print("===== KNN (defaults) =====")
        knn, knn_y_pred, knn_metrics_dict = knn_learner(X_train, y_train, X_test, y_test)
        print()

        # KNN search for optimal value of K:
        print("===== KNN (grid search for K) =====")
        opt_knn, opt_knn_y_pred, opt_knn_metrics_dict = knn_grid_search(X_train, y_train, X_test, y_test, fig_filename=f"plots/{dataset}_knn_gridsearch.png")
        print()

        print(f"===== finished dataset: {dataset} =====")

    print("DONE!")


if __name__ == "__main__":
    main()
