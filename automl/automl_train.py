"""
Runs AutoSKLearn AutoML search for tabular data
"""

import argparse
import os
import pickle
from datetime import datetime, timedelta

import autosklearn.classification
import numpy as np
import pandas as pd
from autosklearn.experimental.askl2 import AutoSklearn2Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score)
from sklearn.model_selection import StratifiedKFold


def evaluate_rf(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    print(clf)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0, average='weighted')
    f1 = f1_score(y_test, y_pred, zero_division=0, average='weighted')
    print(f"Accuracy: {acc:.3f}\nPrecision: {prec:.3f}\nF1: {f1:.3f}")
    return acc, prec, f1

def precision_custom(solution, prediction):
    return precision_score(solution, prediction, zero_division=0, average='weighted')

def f1_custom(solution, prediction):
    return f1_score(solution, prediction, zero_division=0, average='weighted')

def print_results(y_true, y_pred):
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}\n" + \
        f"Precision: {precision_score(y_true, y_pred, zero_division=0, average='weighted'):.3f}\n" + \
        f"F1: {f1_score(y_true, y_pred, zero_division=0, average='weighted'):.3f}")

precision_auto = autosklearn.metrics.make_scorer(
    name='precision',
    score_func=precision_custom,
    optimum=1,
    greater_is_better=True,
    needs_proba=False,
    needs_threshold=False
)

f1_auto = autosklearn.metrics.make_scorer(
    name='f1',
    score_func=f1_custom,
    optimum=1,
    greater_is_better=True,
    needs_proba=False,
    needs_threshold=False
)

BOLD = '\x1B[1m'
RESET = "\x1b[0m"

parser = argparse.ArgumentParser()
parser.add_argument("--input",
                    type=str,
                    required=True,
                    help="csv for model training")

parser.add_argument("--output",
                    type=str,
                    required=True,
                    help="output file prefix")

parser.add_argument("--max_time",
                    type=int,
                    required=True,
                    help="max time in seconds")

parser.add_argument("--autosklearn_version",
                    type=int,
                    required=False,
                    default=1,
                    help="autosklearn version")

parser.add_argument("--n_jobs",
                    type=int,
                    required=False,
                    default=-1,
                    help="n jobs for processing")

parser.add_argument("--tmpdir", 
                    type=str,
                    required=False,
                    default=None,
                    help="possible tmpdir")

parser.add_argument("--sep",
                    type=str,
                    required=False,
                    default=',',
                    help="csv separator")

parser.add_argument("--decimal",
                    type=str,
                    required=False,
                    default='.',
                    help="decimal separator")

args = parser.parse_args()

def main(args):
    uid = datetime.now().strftime("%y%m%dT%H%M%S")
    basename, ext = os.path.splitext(os.path.basename(args.input))

    # Read csv
    df = pd.read_csv(args.input, sep=args.sep, decimal=args.decimal)

    dfY = df.iloc[:,0]
    dfX = df.iloc[:,1:]

    # Print info
    print(BOLD + "Columns. First one is chosen as target" + RESET)
    print("Index\t\tColumn")
    for i, col in enumerate(df.columns):
        print(f"{i}\t\t{col}")
    print()

    print(BOLD + "\nTarget class distribution" + RESET)
    print("label\tcount")
    print(dfY.value_counts())
    print()

    # Classes smaller than 6 are removed
    drop_classes = dfY.value_counts()[dfY.value_counts()<6].index.values
    drop_series = ~dfY.isin(drop_classes)

    dfY = dfY.loc[drop_series]
    dfX = dfX.loc[drop_series,:]

    print("Classes smaller than 6 are removed:")
    print(drop_classes)
    print()

    # Final dataset
    X = dfX.to_numpy()
    y = dfY.to_numpy()

    print(f"Shape of X: {X.shape}")

    # Actual training

    seed = 42

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    train, test = next(skf.split(X,y))
    X_train = X[train,:]
    X_test = X[test,:]
    y_train = y[train]
    y_test = y[test]

    print("Processing...")
    print(f"Started at {datetime.strftime(datetime.now(), '%H:%M:%S')}")
    print(f"Stops approx. at {datetime.strftime(datetime.now() + timedelta(seconds = args.max_time), '%H:%M:%S')}")
    if args.autosklearn_version == 1:
        clf = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=args.max_time,
                                                            resampling_strategy=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
                                                            initial_configurations_via_metalearning=0,
                                                            n_jobs=args.n_jobs,
                                                            tmp_folder=args.tmpdir,
                                                            metadata_directory=args.tmpdir,
                                                            scoring_functions=[autosklearn.metrics.accuracy, 
                                                                                autosklearn.metrics.balanced_accuracy, 
                                                                                precision_auto, 
                                                                                f1_auto])
    elif args.autosklearn_version == 2:
        clf = AutoSklearn2Classifier(time_left_for_this_task=args.max_time,
                                    scoring_functions=[autosklearn.metrics.accuracy,
                                                        autosklearn.metrics.balanced_accuracy, 
                                                        precision_auto,
                                                        f1_auto])

    else:
        raise Exception('Invalid auto-sklearn version!')
    clf.fit(X_train, y_train, dataset_name='lappieo')
    print("Done")

    print("\n RUN STATISTICS:")
    print(clf.sprint_statistics())

    y_true = []
    y_pred_rf = []
    y_pred_automl = []

    for i, (train,test) in enumerate(skf.split(X, y)):
        X_train = X[train,:]
        X_test = X[test,:]
        y_train = y[train]
        y_test = y[test]

        print(BOLD + f"\nFold {i}:" + RESET)
        rf = RandomForestClassifier()

        rf.fit(X_train, y_train)
        clf.refit(X_train, y_train)

        print("\nRF:")
        acc, prec, f1 = evaluate_rf(rf, X_test, y_test)
        print("\nAutoSklearn:")
        acc, prec, f1 = evaluate_rf(clf, X_test, y_test)

        y_pred_fold_rf = rf.predict(X_test)
        y_pred_fold_automl = clf.predict(X_test)

        y_true = np.concatenate((y_true, y_test))
        y_pred_rf = np.concatenate((y_pred_rf, y_pred_fold_rf))
        y_pred_automl = np.concatenate((y_pred_automl, y_pred_fold_automl))

    print(BOLD + "\n\nFINAL RESULTS:" + RESET)
    print("\nRF results:")
    print_results(y_true, y_pred_rf)

    print("\nAutoSklearn results:")
    print_results(y_true, y_pred_automl)

    acc = accuracy_score(y_true, y_pred_automl)

    # Save the model
    output_name = f"{args.output}_{args.max_time}sec_acc{acc:.4f}_{uid}.pkl"
    print(f"\nSaving model to {output_name}")
    with open(output_name, "wb") as open_file:
        pickle.dump(clf, open_file)

if __name__=="__main__":
    main(args)
