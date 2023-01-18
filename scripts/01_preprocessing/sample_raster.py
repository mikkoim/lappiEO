import argparse
from datetime import datetime
import geopandas as gpd
import rasterio as rio
import os
from pathlib import Path
from pprint import pprint

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_shp', type=str)
    parser.add_argument('--input_raster', type=str)
    parser.add_argument('--target', type=str, help='target column')
    parser.add_argument('--band_names', type=str, default=None, help='file with band names as rows')
    parser.add_argument('--dropna', type=int, default=None, help='drops rows with all values as this value')
    parser.add_argument('--out_folder', type=str)
    
    args = parser.parse_args()
    """
    args = parser.parse_args(["--input_shp", 
                                "./data/processed/vectors/samples_CORINE/Clc2018_FI20m__large-ds_500m__id.shp" ,
                                "--input_raster",
                                "./data/raw/rasters/clc2018_fi20m/Clc2018_FI20m.tif",
                                "--target",
                                "band0" ,
                                "--out_folder",
                                "./data/processed/vectors/samples_CORINE"])
    """
    
    # Read files
    gdf = gpd.read_file(args.input_shp)
    src = rio.open(args.input_raster, windowed=True)
    print(f"Sampling raster {args.input_raster} using points from {args.input_shp}")
    
    # Sample points
    coords = [(x,y) for x,y in zip(gdf.geometry.x,gdf.geometry.y)]
    gdf['rvalue'] = [x for x in src.sample(coords)]
    
    # Fix dataframe
    bands = [f'band{i}' for i in range(len(gdf['rvalue'][0]))]
    
    if args.band_names:
        with open(args.band_names) as f:
            lines = f.readlines()
        bandnames = [l.strip() for l in lines]
        print("Using bandnames:")
        pprint(bandnames)
        if len(bands) != len(bandnames):
            raise Exception(f"Mismatch in band names in file ({len(bandnames)})"\
                f" and number of bands ({len(bands)})")
        else:
            bands = bandnames
            
    gdf[bands] = gdf['rvalue'].values.tolist()
    gdf[bands] = gdf[bands].astype(src.meta['dtype'])
    gdf = gdf.drop(['rvalue'],axis=1)
    
    # Create df for csv
    df = gdf[[args.target]+bands]
    
    if args.dropna:
        df = df.loc[~(df[bands]==0).all(axis=1)] #drop rows where all values zeros
    df = df.dropna().reset_index().drop(['index'],axis=1)
                        
    # Saving
    
    shp_stem = Path(args.input_shp).stem
    raster_stem = Path(args.input_raster).stem

    out_folder = Path(args.out_folder)
    out_folder.mkdir(exist_ok=True)
    out_stem = Path(f"{raster_stem}__{shp_stem}__{args.target}")
    
    gdf.to_file(out_folder / out_stem.with_suffix('.shp'))
    df.to_csv(out_folder / out_stem.with_suffix('.csv'), index=False)
    print(f"Saved outputs to {str(out_folder / out_stem)}")