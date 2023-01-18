"""
Takes a vector file and creates a buffer around it
"""

import rasterio
import geopandas as gpd
import pathlib
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument('--buffer', type=int, required=True)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    gdf = gpd.read_file(args.input)
    out_gdf = gdf.copy()
    out_gdf.geometry = gdf.geometry.buffer(args.buffer, cap_style=3)
    out_gdf.to_file(args.output)
    print('Done')