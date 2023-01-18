"""
Reads absolute filename locations and labels according to a gdf shapefile and
saves these to a csv
"""

import argparse

import geopandas as gpd
import lappieo.utils as ut
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdf_fname", type=str, required=True,
                        help="location of the geodataframe with files and labels")
    parser.add_argument("--label_col", type=str, required=True,
                        help="dataframe column with label information")
    parser.add_argument("--fname_col", type=str, required=True,
                        help="dataframe column with filename information")
    parser.add_argument("--tif_folder", type=str, 
                        help="location of the tif-files")
    parser.add_argument("--out_fname", type=str, required=True,
                        help="output csv name")
    args = parser.parse_args()

    gdf = gpd.read_file(args.gdf_fname)
    gdf = ut.preprocess_gdf(gdf, args.fname_col)
    gdf = ut.preprocess_gdf(gdf, args.label_col)

    fnames, labels = ut.read_tif_filenames(gdf, args.fname_col, args.label_col, args.tif_folder)
    
    assert len(fnames) == len(labels)
    
    fname_df = pd.DataFrame(list(zip(fnames,labels)), columns=['fname', 'label'])
    
    fname_df.to_csv(args.out_fname, index=False)
    
    print("Done!")

if __name__=='__main__':
    main()
    exit()
