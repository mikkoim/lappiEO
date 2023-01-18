"""

"""

import argparse
import os

import geopandas as gpd
import lappieo.utils as ut
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdf_fname", type=str, required=True,
                        help="location of the geodataframe with files and labels")
    parser.add_argument("--label_col", type=str, required=True,
                        help="dataframe column with label information")
    parser.add_argument("--tif_folder", type=str, 
                        help="location of the tif-files")
    parser.add_argument("--out_fname", type=str, required=True,
                        help="output csv name")
    args = parser.parse_args()

    tif_folder = os.path.abspath(args.tif_folder)
    
    gdf = gpd.read_file(args.gdf_fname)
    gdf = ut.preprocess_gdf(gdf, 'fid')
    gdf = ut.preprocess_gdf(gdf, args.label_col, func=str)

    fnames, labels = ut.read_tif_filenames(gdf, fid_col='fid', label_col=args.label_col, tif_folder=tif_folder)
    
    fnames, labels = ut.prune_dataset_by_label(fnames, labels, ['', ' '])
        
    assert len(fnames) == len(labels)
    
    fname_df = pd.DataFrame(list(zip(fnames,labels)), columns=['fname', 'label'])
    
    fname_df.to_csv(args.out_fname, index=False)
    
    print("Done!")

if __name__=='__main__':
    main()
    exit()
