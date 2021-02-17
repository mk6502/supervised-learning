import logging
from sklearn.ensemble import AdaBoostClassifier
from utils import basic_metrics


logger = logging.getLogger()


def adaboost_learner(X_train, y_train, X_test, y_test, n_estimators=50, random_state=123):
    """
    Wrapper around AdaBoostClassifier.
    """
    ada = AdaBoostClassifier(n_estimators=n_estimators, random_state=random_state)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    return ada, basic_metrics(y_test, y_pred)
