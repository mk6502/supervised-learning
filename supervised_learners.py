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
    d = dict()
    d["acc"] = metrics.accuracy_score(y_test, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    d["auc"] = metrics.auc(fpr, tpr)
    print(f"Accuracy: {d['acc']}")
    print(f"AUC: {d['auc']}")
    return d


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

    metrics_dict = basic_metrics(y_test, y_pred)

    return dt, y_pred, metrics_dict


def decision_tree_grid_search(X_train, y_train, X_test, y_test):
    # some code taken from: https://www.ritchieng.com/machine-learning-efficiently-search-tuning-param/
    possible_max_depth = [1, 5, 10, 25, 100]
    param_grid = {"max_depth": possible_max_depth}

    dt = DecisionTreeClassifier()

    grid = GridSearchCV(dt, param_grid, cv=10, scoring="accuracy")  # 10 folds
    grid.fit(X_train, y_train)

    optimal_max_depth = grid.best_params_["max_depth"]
    print(f"Optimal max_depth={optimal_max_depth}")

    return decision_tree_learner(X_train, y_train, X_test, y_test, max_depth=optimal_max_depth)


def neural_network_learner(X_train, y_train, X_test, y_test, random_state=123, activation="relu", alpha=0.0001, max_iter=200):
    """
    Neural network classifier.
    """
    # Fit to training data:
    nn = MLPClassifier(random_state=random_state, activation=activation, alpha=alpha, max_iter=max_iter)
    nn.fit(X_train, y_train)

    # Predict:
    y_pred = nn.predict(X_test)

    # Stats:
    print(f"NN Learner with activation={activation} and alpha={alpha}:")
    metrics_dict = basic_metrics(y_test, y_pred)

    return nn, y_pred, metrics_dict


def neural_network_grid_search(X_train, y_train, X_test, y_test, random_state=123):
    # some code taken from: https://www.ritchieng.com/machine-learning-efficiently-search-tuning-param/
    possible_activation = ["relu", "logistic"]
    possible_alpha = [0.0001, 0.00001, 0.001]
    param_grid = {"activation": possible_activation, "alpha": possible_alpha}

    nn = MLPClassifier(random_state=random_state)

    grid = GridSearchCV(nn, param_grid, cv=10, scoring="accuracy")  # 10 folds
    grid.fit(X_train, y_train)

    optimal_activation = grid.best_params_["activation"]
    optimal_alpha = grid.best_params_["alpha"]
    print(f"Optimal activation={optimal_activation}, alpha={optimal_alpha}")

    return neural_network_learner(X_train, y_train, X_test, y_test, random_state=random_state, activation=optimal_activation, alpha=optimal_alpha)


def adaboost_learner(X_train, y_train, X_test, y_test, n_estimators=50, random_state=123):
    # Fit to training data:
    ada = AdaBoostClassifier(n_estimators=n_estimators, random_state=random_state)
    ada.fit(X_train, y_train)

    # Predict:
    y_pred = ada.predict(X_test)

    # Stats:
    print(f"AdaBoost Learner with n_estimators={n_estimators}:")
    metrics_dict = basic_metrics(y_test, y_pred)

    return ada, y_pred, metrics_dict


def adaboost_grid_search(X_train, y_train, X_test, y_test, random_state=123):
    # some code taken from: https://www.ritchieng.com/machine-learning-efficiently-search-tuning-param/
    possible_n_estimators = [10, 50, 100, 200]
    param_grid = {"n_estimators": possible_n_estimators}

    ada = AdaBoostClassifier(random_state=random_state)

    grid = GridSearchCV(ada, param_grid, cv=10, scoring="accuracy")  # 10 folds
    grid.fit(X_train, y_train)

    optimal_n_estimators = grid.best_params_["n_estimators"]
    print(f"Optimal activation={optimal_n_estimators}")

    return adaboost_learner(X_train, y_train, X_test, y_test, random_state=random_state, n_estimators=optimal_n_estimators)


def svm_learner(X_train, y_train, X_test, y_test, kernel="rbf", random_state=123, max_iter=-1):
    # Fit to training data:
    svm = SVC(kernel=kernel, random_state=random_state, max_iter=max_iter)
    svm.fit(X_train, y_train)

    # Predict:
    y_pred = svm.predict(X_test)

    # Stats:
    print(f"SVM Learner with kernel={kernel}:")
    metrics_dict = basic_metrics(y_test, y_pred)

    return svm, y_pred, metrics_dict


def knn_learner(X_train, y_train, X_test, y_test, n_neighbors=5):
    # Fit to training data:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

    # Predict:
    y_pred = knn.predict(X_test)

    # Stats:
    print(f"KNN Learner with n_neighbors={n_neighbors}:")
    metrics_dict = basic_metrics(y_test, y_pred)

    return knn, y_pred, metrics_dict


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
    plt.plot(possible_k, grid_mean_scores, marker="o")
    plt.title(f"KNN Grid Search for optimal K ({optimal_k})")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig(fig_filename)
    plt_clear()

    return knn_learner(X_train, y_train, X_test, y_test, n_neighbors=optimal_k)
