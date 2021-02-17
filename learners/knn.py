import logging
from sklearn.neighbors import KNeighborsClassifier
from utils import basic_metrics


logger = logging.getLogger()


def knn_learner(X_train, y_train, X_test, y_test, n_neighbors=5):
    """
    Wrapper around KNeighborsClassifier.
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return knn, basic_metrics(y_test, y_pred)
