import lappieo.utils as ut
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--csv", type=str)
parser.add_argument("--splits", type=int)

args = parser.parse_args()

for split in range(args.splits):
    ut.csv_train_test_split(args.csv, split_n=split, n_splits=args.splits, shuffle=True, random_state=666)

