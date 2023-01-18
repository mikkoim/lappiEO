import os
from pathlib import Path
import numpy as np
import geopandas as gpd

from sklearn.model_selection import StratifiedKFold

import argparse

def diff_classes(gdf0: gpd.GeoDataFrame,
                gdf1: gpd.GeoDataFrame,
                col: str) -> tuple:
    u0 = set(np.unique(gdf0[col].values))
    u1 = set(np.unique(gdf1[col].values))
    return u0-u1, u1-u0

def remove_attribute_rows(df: gpd.GeoDataFrame, s, a: str) -> gpd.GeoDataFrame:
    for label in s:
        df = df[~(df[a]==label)]
    return df

def find_overlapping_geoms(gdf0: gpd.GeoDataFrame, 
                            gdf1: gpd.GeoDataFrame) -> np.ndarray: 
    n_0 = len(gdf0)
    n_1 = len(gdf1)
    T = np.zeros((n_0, n_1))
    assert(n_0 >= n_1)
        
    for i,g1 in enumerate(gdf0.geometry):
        for j,g2 in enumerate(gdf1.geometry):
            if g1.touches(g2) or g1.overlaps(g2) or g1.intersects(g2) or g1.crosses(g2):
                T[i,j] = 1

    return T

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--gdf_fname", type=str, required=True)
    parser.add_argument("--fold_prefix", type=str)
    parser.add_argument("--outfolder", type=str, default='.')
    parser.add_argument("--column", type=str)
    parser.add_argument("--n_splits", type=int, default=5)

    args = parser.parse_args()
    print(os.getcwd())
    gdf_o = gpd.read_file(args.gdf_fname)

    attribute = args.column

    gdf_o[attribute].value_counts()
    print(gdf_o[attribute].value_counts().sum(), len(gdf_o))
    gdf_o[attribute].value_counts().plot(kind='bar')

    gdf = gdf_o.copy()
    labels = gdf[attribute]
    gdf = gdf[gdf[attribute].notna()]

    too_small = gdf[attribute].value_counts() < args.n_splits

    gdf = remove_attribute_rows(gdf, too_small[too_small].index, attribute)

    print(gdf[attribute].value_counts())
    print(gdf[attribute].value_counts().sum(), len(gdf))
    gdf[attribute].value_counts().plot(kind='bar')

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=666)

    index_list = [x for x in skf.split(gdf.index, gdf[attribute].values)]

    train_gdfs = []
    test_gdfs = []

    for split in index_list:
        train_idx, test_idx = split

        train_gdf = gdf.iloc[train_idx]
        train_gdfs.append(train_gdf)
        test_gdf = gdf.iloc[test_idx]
        test_gdfs.append(test_gdf)

        train_labels = train_gdf[attribute]
        test_labels = test_gdf[attribute]

        assert(np.all(np.unique(train_labels.values) == np.unique(test_labels.values)))

        print(train_labels.value_counts())
        print(test_labels.value_counts())

    basename = Path(args.gdf_fname).stem
    outfolder = Path(args.outfolder)
    outfolder.mkdir(exist_ok=True)

    for i in range(len(index_list)):
        train_gdf = train_gdfs[i]
        test_gdf = test_gdfs[i]

        T = find_overlapping_geoms(train_gdf, test_gdf)
        
        train_new = train_gdf[np.all(T==0, axis=1)]
        class_diffs = diff_classes(train_new, test_gdf, attribute)
        print(class_diffs)

        train_new = remove_attribute_rows(train_new, class_diffs[0], attribute)
        test_gdf = remove_attribute_rows(test_gdf, class_diffs[1], attribute)

        assert(np.all(np.unique(train_new[attribute].values) == np.unique(test_gdf[attribute].values)))

        print('Train')
        print(train_new[attribute].value_counts())
        print('\nTest')
        print(test_gdf[attribute].value_counts())
        print()

        train_new.to_file(outfolder / f'{basename}_{args.fold_prefix}{i}_train.shp')
        test_gdf.to_file(outfolder / f'{basename}_{args.fold_prefix}{i}_test.shp')
