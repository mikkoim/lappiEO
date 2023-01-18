"""
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_modelname_atts(modelname, splitchar):
    return os.path.splitext(os.path.split(modelname)[1])[0].split(splitchar)

parser = argparse.ArgumentParser()

parser.add_argument('--csv', type=str, required=True)
parser.add_argument("--column", type=str, required=True)

args = parser.parse_args()

df = pd.read_csv(args.csv, delimiter=';', decimal=',')
modelpaths = np.unique(df[args.column])

attributes = [parse_modelname_atts(model, '_') for model in modelpaths]
all_attributes = np.unique(np.concatenate(attributes))

for att in all_attributes:
    df.insert(len(df.columns), att,False)

for model in modelpaths:
    atts = parse_modelname_atts(model, '_')
    for a in atts:
        df.loc[df['model']==model, a] = True

modelbase = os.path.splitext(os.path.split(args.csv)[1])[0]
df.to_csv(f'{modelbase}_attributes.csv', index=False, decimal=',', sep=';')
print('Attributes: ', all_attributes)
print('Done! output:', f'{modelbase}_attributes.csv')
