# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 11:04:06 2020

@author: E1007914
"""

import pandas as pd
import src.utils as ut

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, required=True,
                    help="Location of the csv containing filenames and labels")
parser.add_argument("--out", type=str)
parser.add_argument("--level", type=int)

args = parser.parse_args()

label_map = lambda x: x[:args.level]

def main(args):
    fnames, labels = ut.read_fname_csv(args.csv)

    labels = list(map(label_map, labels))

    assert len(fnames) == len(labels)

    fname_df = pd.DataFrame(list(zip(fnames,labels)), columns=['fname', 'label'])
    fname_df.to_csv(args.out, index=False)

    print(labels[:5])

if __name__=='__main__':
    main(args)
    exit()
