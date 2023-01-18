# -*- coding: utf-8 -*-
"""
Mostly obsolete
"""
import geopandas as gpd
import numpy as np

import lappieo.utils as ut


def read_ilmakuvatulkinta_dataset(gdf_fname, tif_folder0, tif_folder1):
    
    # Ilmakuvatulkinta
    gdf0 = gpd.read_file(gdf_fname, layer='S2')
    gdf0 = ut.preprocess_gdf(gdf0, 'OBJECTID')
    gdf0 = ut.preprocess_gdf(gdf0, 'Natura2000')

    fnames0, labels0 = ut.read_tif_filenames(gdf0, fid_col="OBJECTID", label_col='Natura2000', tif_folder=tif_folder0)
    
    # Tunturiniityt
    gdf1 = gpd.read_file(gdf_fname, layer='S2_tunturiniityt')
    gdf1 = ut.preprocess_gdf(gdf1, 'OBJECTID')
    gdf1 = ut.preprocess_gdf(gdf1, 'natura')
    
    fnames1, labels1 = ut.read_tif_filenames(gdf1, fid_col="OBJECTID", label_col='natura', tif_folder=tif_folder1)
    
    # Combine
    fnames = fnames0+fnames1
    labels = labels0+labels1
    return fnames, labels


def load_ilmakuvatulkinta_df_features(fname, layer):
    """Loads features from the dataframe itself
    """
    
    # Ilmakuvat 1
    gdf0 = gpd.read_file(fname, layer=layer)
    X0, labels0 = ut.read_sampled_data(gdf0, range(-12,-1), 'Natura2000')
    
    # Tunturiniityt
    gdf1 = gpd.read_file(fname, layer=layer+'_tunturiniityt')
    X1, labels1 = ut.read_sampled_data(gdf1, range(-12,-1), 'natura')

    # Combine and label
    X = np.concatenate((X0,X1),axis=0)
    labels = labels0+labels1
    
    # Remove wrong labels
    wrong_labels = ['101', '104', '90100', '91']
    wrong_inds = [(x in wrong_labels) for x in labels]

    X = np.delete(X,wrong_inds, axis=0)
    labels = np.delete(labels,wrong_inds)
    
    # Produce dataframe

    y, le = ut.encode_labels(labels)
    return X, y, le, [gdf0, gdf1]
    