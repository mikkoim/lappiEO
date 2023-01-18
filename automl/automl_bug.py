"""
Self-contained example of the autosklearn script not working on CSC
"""

import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

if __name__ == "__main__":
    X, y = sklearn.datasets.load_digits(return_X_y=True)
    print(X.shape, y.shape)
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)
    automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=60)
    print(automl)
    automl.fit(X_train, y_train)
    y_hat = automl.predict(X_test)
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))