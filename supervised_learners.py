from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

import matplotlib.pyplot as plt
from utils import plt_clear


def basic_metrics(y_test, y_pred):
    acc = metrics.accuracy_score(y_test, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    auc = metrics.auc(fpr, tpr)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    average_precision = metrics.average_precision_score(y_test, y_pred)
    print(f"Accuracy: {acc}")
    print(f"AUC: {auc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Average Precision: {average_precision}")
    return acc, fpr, tpr, thresholds, auc, precision, recall, average_precision


def decision_tree_learner(X_train, y_train, X_test, y_test, max_depth=None):
    # Fit to training data:
    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train, y_train)

    # Predict:
    y_pred = dt.predict(X_test)

    # Stats:
    if max_depth:
        print(f"DT Learner with max_depth={max_depth}:")
    else:
        print("DT Learning with no max_depth:")

    dt_acc, dt_fpr, dt_tpr, dt_thresholds, dt_auc, dt_precision, dt_recall, dt_average_precision = basic_metrics(y_test, y_pred)

    # Precision-Recall Curve (from https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#plot-the-precision-recall-curve)
    # disp = metrics.plot_precision_recall_curve(dt, X_test, y_test)
    # disp.ax_.set_title('2-class Precision-Recall curve: AP={0:0.2f}'.format(dt_average_precision))

    return dt, y_pred, dt_acc, dt_fpr, dt_tpr, dt_thresholds, dt_auc, dt_precision, dt_recall, dt_average_precision


def neural_network_learner(X_train, y_train, X_test, y_test, random_state=123, activation="relu", alpha=0.0001):
    """
    Neural network classifier.
    """
    # Fit to training data:
    nn = MLPClassifier(random_state=random_state, activation=activation, alpha=alpha)
    nn.fit(X_train, y_train)

    # Predict:
    y_pred = nn.predict(X_test)

    # Stats:
    print(f"NN Learner with activation={activation} and alpha={alpha}:")
    nn_acc, nn_fpr, nn_tpr, nn_thresholds, nn_auc, nn_precision, nn_recall, nn_average_precision = basic_metrics(y_test, y_pred)

    return nn, y_pred, nn_acc, nn_fpr, nn_tpr, nn_thresholds, nn_auc, nn_precision, nn_recall, nn_average_precision


def neural_network_grid_search(X_train, y_train, X_test, y_test, random_state=123):
    # some code taken from: https://www.ritchieng.com/machine-learning-efficiently-search-tuning-param/
    possible_activation = ["relu", "logistic"]
    possible_alpha = [0.0001, 0.00001, 0.001]
    param_grid = {"activation": possible_activation, "alpha": possible_alpha, "random_state": [random_state]}

    nn = MLPClassifier()

    grid = GridSearchCV(nn, param_grid, cv=10, scoring="accuracy")  # 10 folds
    grid.fit(X_train, y_train)

    optimal_activation = grid.best_params_["activation"]
    optimal_alpha = grid.best_params_["alpha"]
    print(f"Optimal activation={optimal_activation}, alpha={optimal_alpha}")

    return neural_network_learner(X_train, y_train, X_test, y_test, random_state=random_state, activation=optimal_activation, alpha=optimal_alpha)


def adaboost_learner(X_train, y_train, X_test, y_test, n_estimators=1000, random_state=123):
    # Fit to training data:
    ada = AdaBoostClassifier(n_estimators=n_estimators, random_state=random_state)
    ada.fit(X_train, y_train)

    # Predict:
    y_pred = ada.predict(X_test)

    # Stats:
    print(f"AdaBoost Learner with n_estimators={n_estimators}:")
    ada_acc, ada_fpr, ada_tpr, ada_thresholds, ada_auc, ada_precision, ada_recall, ada_average_precision = basic_metrics(y_test, y_pred)

    return ada, y_pred, ada_acc, ada_fpr, ada_tpr, ada_thresholds, ada_auc, ada_precision, ada_recall, ada_average_precision


def svm_learner(X_train, y_train, X_test, y_test, kernel="polynomial", random_state=123):
    # Fit to training data:
    svm = SVC(kernel=kernel, random_state=random_state)
    svm.fit(X_train, y_train)

    # Predict:
    y_pred = svm.predict(X_test)

    # Stats:
    print(f"SVM Learner with kernel={kernel}:")
    svm_acc, svm_fpr, svm_tpr, svm_thresholds, svm_auc, svm_precision, svm_recall, svm_average_precision = basic_metrics(
        y_test, y_pred)

    return svm, y_pred, svm_acc, svm_fpr, svm_tpr, svm_thresholds, svm_auc, svm_precision, svm_recall, svm_average_precision


def knn_learner(X_train, y_train, X_test, y_test, n_neighbors=5):
    # Fit to training data:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

    # Predict:
    y_pred = knn.predict(X_test)

    # Stats:
    print(f"KNN Learner with n_neighbors={n_neighbors}:")
    knn_acc, knn_fpr, knn_tpr, knn_thresholds, knn_auc, knn_precision, knn_recall, knn_average_precision = basic_metrics(y_test, y_pred)

    return knn, y_pred, knn_acc, knn_fpr, knn_tpr, knn_thresholds, knn_auc, knn_precision, knn_recall, knn_average_precision


def knn_grid_search(X_train, y_train, X_test, y_test, fig_filename):
    # some code taken from: https://www.ritchieng.com/machine-learning-efficiently-search-tuning-param/
    possible_k = list(range(1, 26))
    param_grid = {"n_neighbors": possible_k}

    knn = KNeighborsClassifier()

    grid = GridSearchCV(knn, param_grid, cv=10, scoring="accuracy")  # 10 folds
    grid.fit(X_train, y_train)

    optimal_k = grid.best_params_["n_neighbors"]
    grid_mean_scores = [x for x in grid.cv_results_["mean_test_score"]]
    plt_clear()
    plt.plot(possible_k, grid_mean_scores)
    plt.title(f"Grid Search for optimal K: {optimal_k}")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.savefig(fig_filename)
    plt_clear()

    return knn_learner(X_train, y_train, X_test, y_test, n_neighbors=optimal_k)
