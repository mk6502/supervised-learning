import logging
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from utils import basic_metrics


logger = logging.getLogger()


def neural_network_learner(X_train, y_train, X_test, y_test, random_state=123, activation="relu", alpha=0.0001, max_iter=200):
    """
    Neural network classifier.
    """
    nn = MLPClassifier(random_state=random_state, activation=activation, alpha=alpha, max_iter=max_iter)
    nn.fit(X_train, y_train)
    y_pred = nn.predict(X_test)
    return nn, basic_metrics(y_test, y_pred)


def neural_network_grid_search(X_train, y_train, X_test, y_test, random_state, possible_activation, possible_alpha):
    """
    Grid search for neural networks.
    """
    param_grid = {"activation": possible_activation, "alpha": possible_alpha}

    nn = MLPClassifier(random_state=random_state)

    grid = GridSearchCV(nn, param_grid, cv=10, scoring="accuracy")  # 10 folds
    grid.fit(X_train, y_train)

    optimal_activation = grid.best_params_["activation"]
    optimal_alpha = grid.best_params_["alpha"]
    logger.info(f"Optimal activation={optimal_activation}, alpha={optimal_alpha}")

    return optimal_activation, optimal_alpha
