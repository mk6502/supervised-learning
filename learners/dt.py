import logging
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from utils import basic_metrics


logger = logging.getLogger()


def decision_tree_learner(X_train, y_train, X_test, y_test, max_depth=None):
    """
    Wrapper around DecisionTreeClassifier.
    """
    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    return dt, basic_metrics(y_test, y_pred)


def decision_tree_grid_search(X_train, y_train, X_test, y_test, possible_max_depth=[3, 5, 10, 25, 50]):
    """
    GridSearch for DecisionTreeClassifier.
    """
    param_grid = {"max_depth": possible_max_depth}

    dt = DecisionTreeClassifier()

    grid = GridSearchCV(dt, param_grid, cv=10, scoring="accuracy")  # 10 folds
    grid.fit(X_train, y_train)

    optimal_max_depth = grid.best_params_["max_depth"]
    logger.info(f"Optimal max_depth={optimal_max_depth}")

    return decision_tree_learner(X_train, y_train, X_test, y_test, max_depth=optimal_max_depth)
