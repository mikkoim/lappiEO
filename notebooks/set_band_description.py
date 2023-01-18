"""
Set Band descriptions
Usage:
    python set_band_desc.py raster bands
Where:
    raster = raster filename
    bands = file containing band names in order
Example:
    python set_band_desc.py raster.tif bandnames.txt

"""
import sys
from osgeo import gdal

def set_band_descriptions(filepath, bands):
    """
    filepath: path/virtual path/uri to raster
    bands:    ((band, description), (band, description),...)
    """
    ds = gdal.Open(filepath, gdal.GA_Update)
    for band, desc in enumerate(bands):
        rb = ds.GetRasterBand(band+1)
        rb.SetDescription(desc)
    del ds

if __name__ == '__main__':
    filepath = sys.argv[1]
    descpath = sys.argv[2]
    with open(descpath, 'r') as f:
        bands = f.readlines()
    set_band_descriptions(filepath, bands)