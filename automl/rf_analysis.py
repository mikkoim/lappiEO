import os
import argparse
import pickle
import sys
from datetime import datetime, timedelta
from distutils.util import strtobool
from pprint import pprint
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score)
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

"""
python rf_analysis-py \
    --input train.csv

"""

def add_rf_args(parser):
    parser.add_argument('--n_estimators',
                        type=int,
                        default=100,
                        help="The number of trees in the forest.")
    parser.add_argument('--criterion',
                        type=str,
                        default='gini',
                        help="The function to measure the quality of a split. " \
                        "{“gini”, “entropy”, “log_loss”}, default=”gini”")
    parser.add_argument('--max_depth',
                        type=int,
                        default=None,
                        help="The maximum depth of the tree. If None, " \
                        "then nodes are expanded until all leaves are pure " \
                        "or until all leaves contain less than min_samples_split samples.")
    return parser

def build_tpot(tpot_fname):
    import importlib
    uid = datetime.now().strftime("%Y%m%dT%H%M%S")
    temp_module_name = Path(f'model{uid}.py')

    with open(tpot_fname, 'r') as f:
        lines = f.readlines()

    with open(temp_module_name, 'w') as f:
        for line in lines:
            if line.startswith('tpot_data'):
                continue
            elif line.startswith('features ='):
                continue
            elif line.startswith('training_features'):
                continue
            elif line.startswith('exported_pipeline.fit'):
                continue
            elif line.startswith('results ='):
                continue
            elif line.strip().startswith('train_test_split'):
                continue
            else:
                f.write(line)
    vals = {}
    exec(temp_module_name.read_text(), vals)
    os.remove(temp_module_name)
    return vals['exported_pipeline']


def build_rf(args):
    rf = RandomForestClassifier(n_estimators=args.n_estimators,
                                criterion=args.criterion,
                                max_depth=args.max_depth,
                                n_jobs=-1)
    return rf

def evaluate_rf(clf, X_test, y_test, confmat=False):
    y_pred = clf.predict(X_test)
    if confmat:
        plt.figure()
        sns.heatmap(confusion_matrix(y_test, y_pred, normalize='true'))

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0, average='weighted')
    f1 = f1_score(y_test, y_pred, zero_division=0, average='weighted')
    logging.info(f"Accuracy: {acc:.3f}\nPrecision: {prec:.3f}\nF1: {f1:.3f}")
    return acc, prec, f1

def array_to_longform(a, columns):
    dfa = pd.DataFrame(data=a, columns=columns)
    dfa = dfa.reset_index()
    return dfa.melt(id_vars=['index'], ignore_index=False)


def classification_reportX(*args, figsize=None, **kwargs):
    """Extends classification report by adding an useful plot for the performance across classes"""
    r = classification_report(*args, **kwargs, output_dict=True)
    rdf0 = pd.DataFrame(r)
    rdf = rdf0.T.iloc[:-3,:].sort_values('support',ascending=False)
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    xbar = range(len(rdf))
    ax1.bar(xbar,rdf['support'], alpha=0.2)
    ax2.plot(rdf['f1-score'], 'ro', label='f1-score')
    ax2.plot(rdf['precision'], 'g*', label='precision', alpha=0.5)
    ax2.plot(rdf['recall'], 'b*', label='recall', alpha=0.5)

    ax2.vlines(xbar, rdf['f1-score'], rdf['precision'], 'g')
    ax2.vlines(xbar, rdf['f1-score'], rdf['recall'], 'b')

    ax2.hlines(rdf0['weighted avg']['f1-score'], 0,len(rdf), color='r', linestyle='--', label='weighted f1-score')
    ax2.hlines(rdf0['macro avg']['f1-score'], 0,len(rdf), color='r', linestyle='-.', label='macro f1-score')
    ax2.hlines(rdf0['accuracy']['recall'], 0,len(rdf), color='b', linestyle='--', label='accuracy')

    fig.legend()
    return classification_report(*args, **kwargs)

def confusion_matrixX(y_true, y_pred, clf, figsize=(15,15)):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=figsize)

    mask = np.zeros_like(cm)
    mask[cm==0] = 1
    sns.heatmap(cm*100, 
                annot=True, 
                fmt='.0f', 
                cmap="YlGnBu",
                xticklabels=clf.classes_, 
                yticklabels=clf.classes_,
                mask=mask,
                square=True, 
                cbar=False,
                ax=ax)
    ax.set_ylabel('True label', size=18)
    ax.set_xlabel('Predicted label', size=18)


def main():
    uid = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    
    parser = argparse.ArgumentParser(
                description="This program trains a random forest model on a csv dataset and "\
                "prints useful information on model performance and feature importance. " \
                "It takes a csv-file as an input, where the first column " \
                "must be the target variable."
    )

    parser.add_argument("--input",
                        type=str,
                        required=True,
                        help="The csv-file containing the training data")
    parser.add_argument('--out_prefix',
                        type=str,
                        required=True,
                        help="The prefix that is added to saved files and figures")
    parser.add_argument("--separator",
                        type=str,
                        required=False,
                        default=";",
                        help="The csv separator character. Default ';'") 
    parser.add_argument("--decimal",
                        type=str,
                        required=False,
                        default=",",
                        help="The csv decimal character. Default ','")
    parser.add_argument('--tpot_model',
                        type=str,
                        required=False,
                        default=None,
                        help='The path to a tpot model definition file to be used instead of random forest')
    parser.add_argument("--n_splits",
                        type=int,
                        default=5,
                        help="The number of cross-validation folds. If set as 1, calculates a single " \
                        "train-test-split validation. Default 5")
    parser.add_argument("--remove_classes_smaller_than",
                        type=int,
                        required=False,
                        default=None,
                        help="Classes smaller than this value are removed. Default None")
    parser.add_argument("--no_permutation_importance",
                        default=False, 
                        action="store_true",
                        help="If this flag is set, no permutation importances are calculated")
    parser.add_argument("--random_seed",
                        type=int,
                        required=False,
                        default=None,
                        help="Random seed. Set a value for deterministic output.")
    parser = add_rf_args(parser)
    args = parser.parse_args()
    """
    args = parser.parse_args(["--input", "automl/train.csv",
                                "--separator", ",", 
                                "--decimal", "."])
    """

    # Logging
    out_folder = Path('model_analysis')
    out_folder.mkdir(exist_ok=True)
    out_stem = Path(f'{args.out_prefix}__{args.input}__{uid}')

    Log_Format = "%(message)s"

    logging.basicConfig(filename = out_folder / out_stem.with_suffix('.log'),
                        filemode = "w",
                        format = Log_Format, 
                        level = logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    # strings for colored output
    if sys.platform == 'win32':
        green = red = blue = bold = RESET = ''
    else:
        green = '\033[92m'
        red = '\033[91m'
        blue = '\033[94m'
        bold = '\x1B[1m'
        RESET = "\x1b[0m"

    # Read the csv
    logging.info(f'Input csv: {args.input}')
    if args.tpot_model:
        logging.info(f'Model: {args.tpot_model}')
    else:
        logging.info(f"Model: RF")
    df = pd.read_csv(args.input,
                        sep=args.separator,
                        decimal=args.decimal)

    dfX = df.iloc[:,1:]
    dfY = df.iloc[:,0]
    feature_names = dfX.columns

    # Print column info
    logging.info( "\n\n### Data info ###\n")
    logging.info(bold + green + "Columns. First one is chosen as target" + RESET)
    logging.info("Index\t\tColumn")
    logging.info(green + f"{0}\t\t{dfY.name}")
    for i, col in enumerate(df.columns[1:]):
        logging.info(blue + f"{i+1}\t\t{col}" + RESET)
    logging.info('\n')
    logging.info(f"Shape of data table:\nrows: {df.shape[0]}\ncolumns: {df.shape[1]}")


    # If argument 'remove_small_classes' is set as 'True', remove classes smaller than the value

    if args.remove_classes_smaller_than:
        logging.info(bold + red + f"\nREMOVING CLASSES SMALLER THAN {args.remove_classes_smaller_than} SAMPLES" + RESET)
        drop_classes = dfY.value_counts()[dfY.value_counts()<6].index.values
        drop_series = ~dfY.isin(drop_classes)
        logging.info(drop_classes)

        dfY = dfY.loc[drop_series]
        dfX = dfX.loc[drop_series,:]

    logging.info(bold + green + "\nTarget class distribution" + RESET)
    logging.info("label\tcount")
    logging.info(dfY.value_counts())
    logging.info('\n')

    # Convert to numpy arrays for processing
    X = dfX.to_numpy()
    y = dfY.to_numpy()

    y_true = []
    y_pred = []

    # Initialize cross-validation
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_seed)

    logging.info(bold + red + "\n\n### Starting cross-validation ###\n" + RESET)

    logging.info(bold + green + "Random forest parameters:" + RESET)
    model = build_rf(args)
    logging.info(pprint(model.get_params()))

    # Perform cross validation with intermediate outputs
    dfs= []
    for i, (train,test) in enumerate(skf.split(X, y)):
        X_train = X[train,:]
        X_test = X[test,:]
        y_train = y[train]
        y_test = y[test]

        logging.info(blue + f"\nRF Fold {i}:" + RESET)
        if args.tpot_model:
            model = build_tpot(args.tpot_model)
        else:
            model = build_rf(args)
        model.fit(X_train, y_train)

        acc, prec, f1 = evaluate_rf(model, X_test, y_test)

        y_pred_fold = model.predict(X_test)
        y_true = np.concatenate((y_true, y_test))
        y_pred = np.concatenate((y_pred, y_pred_fold))

        # Calculate permutation importance
        result = permutation_importance(
                    model, 
                    X_test, 
                    y_test, 
                    n_repeats=10, 
                    random_state=42, 
                    n_jobs=-1,
                    scoring='f1_weighted'
                    )
        dfmelt_perm = array_to_longform(result.importances.T, feature_names)
        dfs.append(dfmelt_perm)

    dfmelt_perm = pd.concat(dfs)

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    classes = model.classes_

    logging.info(bold + green + f"\nOverall results:")
    logging.info(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}\n" + \
        f"Precision: {precision_score(y_true, y_pred, zero_division=0, average='weighted'):.3f}\n" + \
        f"F1: {f1_score(y_true, y_pred, zero_division=0, average='weighted'):.3f}")
    logging.info(RESET)

    # Plotting reports
    logging.info(classification_reportX(y_true.astype(int), y_pred.astype(int), zero_division=0, figsize=(12,5)))
    plt.title(f"Name: {args.out_prefix}\nDataset: {args.input} \nTimestamp: {uid}")
    outname = out_folder / f"{out_stem}_metrics.png"
    plt.savefig(outname)
    logging.info(green + f"Saved classification graph to {outname}"+ RESET)

    confusion_matrixX(y_true, y_pred, model)
    plt.title(f"Name: {args.out_prefix}\nDataset: {args.input} \nTimestamp: {uid}")
    outname = out_folder / f"{out_stem}_confusion.png"
    plt.savefig(outname)
    logging.info(green + f"Saved confusion matrix to {outname}"+ RESET)
    

    # Permutation importance
    plt.figure(figsize=(5,len(feature_names)//3))
    sns_plot = sns.boxplot(data=dfmelt_perm,
                x='value',
                y='variable')
    plt.title(f"Name: {args.out_prefix}\nDataset: {args.input} \nTimestamp: {uid}")
    outname = out_folder / f"{out_stem}_permutation_importance.png"
    plt.tight_layout()
    sns_plot.get_figure().savefig(outname)
    logging.info(green + f"Saved permutation importance to {outname}"+ RESET)


if __name__=='__main__':
    main()
