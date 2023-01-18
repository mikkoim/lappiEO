"""
Clips a raster based on a vector, saving the raster area under each
vector feature to a separate file. Parallelizes to use all available cores
"""

import argparse
import os
import pathlib

import dask
import geopandas as gpd
import rasterio
import rioxarray
import xarray as xr
from dask.diagnostics import ProgressBar
from rasterio.mask import mask

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--shp_fname", type=str, required=True)
    parser.add_argument("--raster_fname", type=str, required=True)
    parser.add_argument("--id_col", type=str, required=True)
    parser.add_argument("--outfolder", type=str, required=True)


    args = parser.parse_args()

    shp_fname = pathlib.Path(args.shp_fname)
    raster_fname = pathlib.Path(args.raster_fname)
    outfolder = pathlib.Path(args.outfolder) / f'{raster_fname.stem}__{shp_fname.stem}'
    if not outfolder.exists():
        outfolder.mkdir(parents=True)

    # Read dataframe
    gdf = gpd.read_file(args.shp_fname)
    print(gdf)

    # Open raster for parallel processing
    src = rasterio.open(args.raster_fname, windowed=True)
    chunk_s = 2**11
    xds = rioxarray.open_rasterio(raster_fname, 
                                chunks={'band': -1, 'x': chunk_s, 'y': chunk_s},
                                lock=False,
                                parallel=True)
    print(xds)

    def save_clip(feature, id):
        """Saves a gdf feature
        """
        try:
            clip = xds.rio.clip([feature], from_disk=True)
            clip.rio.to_raster(outfolder / f'{id}.tif')
        except ValueError:
            pass

    list_of_delayed_functions = []
    for idx, row in gdf.iterrows():
        feature = row.geometry
        id = str(int(row[args.id_col]))
        list_of_delayed_functions.append(dask.delayed(save_clip)(feature, id))

    ### This starts the execution with the resources available
    with ProgressBar():
        dask.compute(list_of_delayed_functions)
