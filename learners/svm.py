import logging
from sklearn.svm import SVC
from utils import basic_metrics


logger = logging.getLogger()


def svm_learner(X_train, y_train, X_test, y_test, kernel="rbf", random_state=123, max_iter=-1):
    """
    Wrapper for SVC (SVM Classifier).
    """
    svm = SVC(kernel=kernel, random_state=random_state, max_iter=max_iter)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    return svm, basic_metrics(y_test, y_pred)
