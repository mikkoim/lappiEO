"""
Made for testing if the AutoSkLearn training script works on CSC clusters
"""

import sys
import sklearn.datasets
import sklearn.metrics

import autosklearn.classification
from datetime import datetime, timedelta

if __name__ == '__main__':
    TIME = int(sys.argv[1])

    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)
    print(X_train.shape)
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=TIME,
        per_run_time_limit=30
    )
    print("starting...")
    print(f"Started at {datetime.strftime(datetime.now(), '%H:%M:%S')}")
    print(f"Stops approx. at {datetime.strftime(datetime.now() + timedelta(seconds = TIME), '%H:%M:%S')}")
    automl.fit(X_train, y_train, dataset_name='breast_cancer')
    print(automl.leaderboard())
