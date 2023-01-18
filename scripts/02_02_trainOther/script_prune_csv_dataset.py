# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 11:04:06 2020

@author: E1007914
"""

import os
import pandas as pd
import geopandas as gpd
import src.utils as ut

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, required=True,
                    help="Location of the csv containing filenames and labels")
parser.add_argument("--out", type=str)
parser.add_argument("--label", type=str)

args = parser.parse_args()

if args.label == 'inv':
    PRUNELIST = ['103 Kalliorotkot',
                '231 Jäkälä (karukkokangas)',
                '261 Jäkäläinen heinä-sara',
                '324 Räme-vesipintasuot',
                '334 Vesipintasuot',
                '422 Puro (leveys < 2 m)',
                '424 Leveä joki (> 5 m)',
                '522 Tuore niitty',
                '523 Kostea niitty']
elif args.label == 'nat':
    PRUNELIST = ['3160 - Humuspitoiset järvet ja lammet',
                '3220 - Tunturijoet ja purot',
                '6270 - Runsaslajiset kuivat ja tuoreet niityt',
                '7240 - Tuntureiden rehevät puronvarsisuot',
                '8210 - Kalkkikalliot']
else: 
    raise Exception()

def main(args):
    fnames, labels = ut.read_fname_csv(args.csv)
    fnames, labels = ut.prune_dataset_by_label(fnames, labels, PRUNELIST)

    assert len(fnames) == len(labels)

    fname_df = pd.DataFrame(list(zip(fnames,labels)), columns=['fname', 'label'])
    fname_df.to_csv(args.out, index=False)

    print(PRUNELIST)

if __name__=='__main__':
    main(args)
    exit()
