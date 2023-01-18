# General workflow

## 01. Add source rasters and vectors to Allas

Needed source data:
- Raw source data rasters
- Extent of the area in an .shp file
- Annotated data points in an .shp file

### Upload data to Allas
```
module load allas
allas-conf
a-put -b impiomik/lahtodata/rasters <your_raster>
a-put -b impiomik/lahtodata/vectors <your_vector>
```

### Source data in allas currently (17.1.2022):
Rasters:        
- clc2018_fi20m.zip
    - Corine land cover raster - entire Finland
- h_nd-2-0_dm.zip
    - kasvillisuuden korkeus
- latvuspeitto_nd-2-0.zip
    - latvuspeitto
- max_ndbi-ndmi_min-mea-max_ndvi_2017-2020.zip
    - NDVI, NDBI, NDMI
- ndvi.zip
    - NDVI mean, max, amplitude
- s2_9bands_2018_markusinput_lappieo.zip
    - Sentinel 2 9 bands
- s2_july2021_pca1235_mxndvi_ndwipc1_lashdm_suot2.zip
    - Sentinel 2, ndvi -> 9 channel PCA + suomaski

- datacube_20loka_nattype.tar
    - Latest datacube

Vectors:
- Maastotieto_v7_koonti_korj.zip
    - 2020 samples
- koealat_20_21_A_220222.zip
    - 2020 + 2021 samples
- fsp_Natty_ok1_1701.zip
    - 2020-2022 samples
- hankealue_eireikia.zip
    - area extent

## 02. Get source rasters from Allas (~5min)
```
module load allas
allas-conf
a-get impiomik/lahtodata/rasters/<your_raster> -d data/raw/rasters
a-get impiomik/lahtodata/vectors/<your_vector> -d data/raw/vectors
```

Current commands (5.9.2022)
```
module load allas
allas-conf
a-get impiomik/lahtodata/rasters/clc2018_fi20m.zip -d data/raw/rasters
a-get impiomik/lahtodata/rasters/h_nd-2-0_dm.zip -d data/raw/rasters
a-get impiomik/lahtodata/rasters/latvuspeitto_nd-2-0.zip -d data/raw/rasters
a-get impiomik/lahtodata/rasters/ndvi.zip -d data/raw/rasters
a-get impiomik/lahtodata/rasters/s2_9bands_2018_markusinput_lappieo.zip -d data/raw/rasters

a-get impiomik/lahtodata/rasters/datacube_20loka_nattype.tar -d data/raw/rasters

a-get impiomik/lahtodata/vectors/fsp_Natty_ok1_1701.zip -d data/raw/vectors
a-get impiomik/lahtodata/vectors/hankealue_eireikia.zip -d data/raw/vectors
```

Unzip source data
```
unzip data/raw/rasters/clc2018_fi20m.zip -d data/raw/rasters/clc2018_fi20m
unzip data/raw/rasters/h_nd-2-0_dm.zip -d data/raw/rasters/h_nd-2-0_dm
unzip data/raw/rasters/latvuspeitto_nd-2-0.zip -d data/raw/rasters/latvuspeitto_nd-2-0
unzip data/raw/rasters/ndvi.zip -d data/raw/rasters/ndvi
unzip data/raw/rasters/s2_9bands_2018_markusinput_lappieo.zip -d data/raw/rasters/s2_9bands_2018_markusinput_lappieo

tar -xvf impiomik/lahtodata/rasters/datacube_20loka_nattype.tar -C data/raw/rasters/datacube_20loka_nattype

unzip data/raw/vectors/fsp_Natty_ok1_1701.zip -d data/raw/vectors/fsp_Natty_ok1_1701
unzip data/raw/vectors/hankealue_eireikia.zip -d data/raw/vectors/hankealue_eireikia
```

## 03. Process the source data

### 3.0 Manual workstep: Create the large-ds sample
Sample random points from the extent file using QGIS and the Vector > Random points inside polygons tool
to create the points for larger unannotated dataset

Used parameters
File: hankealue_eireikia.shp
Point count: 100 000
Distance between points: 500m

Save output to data/processed/vectors/large-ds_XXXm.shp

### 3.1 Check the Data types and pixel sizes (~5min)
```
module load geoconda
cpushell 2 32G 0

gdalinfo -nogcp -nomd -norat -noct -mm data/raw/rasters/h_nd-2-0_dm/h_nd-2-0_dm.img | grep 'Pixel Size\|Type='
gdalinfo -nogcp -nomd -norat -noct -mm data/raw/rasters/latvuspeitto_nd-2-0/latvuspeitto_nd-2-0.img | grep 'Pixel Size\|Type='
gdalinfo -nogcp -nomd -norat -noct -mm data/raw/rasters/ndvi/MaxNDVI_2020.tif | grep 'Pixel Size\|Type='
gdalinfo -nogcp -nomd -norat -noct -mm data/raw/rasters/ndvi/NDVI_2020_sum.tif | grep 'Pixel Size\|Type='
gdalinfo -nogcp -nomd -norat -noct -mm data/raw/rasters/ndvi/NDVI_amp_2020.tif | grep 'Pixel Size\|Type='
gdalinfo -nogcp -nomd -norat -noct -mm data/raw/rasters/s2_9bands_2018_markusinput_lappieo/s2_9bands_2018_markusinput_lappieo.img | grep 'Pixel Size\|Type='

gdalinfo -nogcp -nomd -norat -noct -mm data/raw/rasters/datacube_20loka_nattype/datacube_20loka_nattype.img | grep 'Pixel Size\|Type='
```

### 3.2 Check the columns of the vectors

```
ogrinfo -so -al data/processed/vectors/large-ds_500m.shp
ogrinfo -so -al data/raw/vectors/Maastotieto_v7_koonti_korj/Maastotieto_v7_koonti_korj.shp
ogrinfo -so -al data/raw/vectors/fsp_Natty_ok1_1701/fsp_Natty_ok1_1701.shp
```

With this data we can see that the target columns for annotations are 'Inventoint' and 'NaturaTyyp' 
and the feature identifiers are 'id' and 'OBJECTID_1'


### 3.3 Merge and clip the source raster into a datacube (15-30min + 10min)
```
gdal_merge.py -o data/processed/source0.tif \
            -separate \
            -co COMPRESS=DEFLATE \
            -co BIGTIFF=YES \
            -co TILED=YES \
            -ot UInt16 \
            -ps 10 10 \
            data/raw/rasters/h_nd-2-0_dm/h_nd-2-0_dm.img \
            data/raw/rasters/latvuspeitto_nd-2-0/latvuspeitto_nd-2-0.img \
            data/raw/rasters/ndvi/MaxNDVI_2020.tif \
            data/raw/rasters/ndvi/NDVI_2020_sum.tif \
            data/raw/rasters/ndvi/NDVI_amp_2020.tif \
            data/raw/rasters/s2_9bands_2018_markusinput_lappieo/s2_9bands_2018_markusinput_lappieo.img

gdalwarp \
    -cutline data/raw/vectors/hankealue_eireikia/hankealue_eireikia.shp \
    -crop_to_cutline \
    -co COMPRESS=DEFLATE \
    -co BIGTIFF=YES \
    -co TILED=YES \
    -t_srs EPSG:3067 \
    -dstnodata 65535 \
    data/processed/source0.tif \
    data/processed/source.tif


```

If the datacube is ready, just make sure the nodata values are ok and convert to a compressed tif
```
gdalwarp \
    -cutline data/raw/vectors/hankealue_eireikia/hankealue_eireikia.shp \
    -crop_to_cutline \
    -co COMPRESS=DEFLATE \
    -co BIGTIFF=YES \
    -co TILED=YES \
    -t_srs EPSG:3067 \
    -dstnodata 0 \
    data/raw/rasters/datacube_20loka_nattype/datacube_20loka_nattype.img \
    data/processed/source.tif
```

Create the bandnames.txt -file in data/processed, containing the bandnames of the raw rasters

## 04. Produce buffers for large dataset and field annotations (30sec)

Parameters: 
- 50m

```
python scripts/01_preprocessing/make_buffer.py \
        --input data/processed/vectors/large-ds_500m.shp \
        --buffer 50 \
        --output data/processed/vectors/buffer50_large-ds_500m.shp

python scripts/01_preprocessing/make_buffer.py \
        --input data/raw/vectors/fsp_Natty_ok1_1701/fsp_Natty_ok1_1701.shp \
        --buffer 50 \
        --output data/processed/vectors/buffer50_fsp_Natty_ok1_1701.shp
```



## 05. Clip the source raster based on buffers and create CNN datasets (~60min)
```
sbatch batchjobs/00_01_clip_raster_fsp_Natty_ok1_1701.sh
sbatch batchjobs/00_01_clip_raster_large-ds.sh
```

## 06. Sample the source raster (~10min)

Sampling is done for three different targets:
1. Field-collected Natura2000 class (NaturaTyyp)
2. Field-collected GCS class (Inventoin)
3. CORINE class for the large dataset

```
# Field-collected
sbatch batchjobs/00_02_sample_raster_fsp_Natty_ok1_1701.sh
# CORINE
sbatch batchjobs/00_02_sample_raster_large-ds.sh
```

## 07. Create non-overlapping train-test-splits

```
python scripts/01_preprocessing/script_create_nonoverlapping_splits.py \
        --gdf_fname data/processed/vectors/buffer50_fsp_Natty_ok1_1701.shp \
        --fold_prefix nat \
        --outfolder data/processed/vectors/splits \
        --column NaturaTyyp \
        --n_splits 5

python scripts/01_preprocessing/script_create_nonoverlapping_splits.py \
        --gdf_fname data/processed/vectors/buffer50_fsp_Natty_ok1_1701.shp \
        --fold_prefix inv \
        --outfolder data/processed/vectors/splits \
        --column Inventoint \
        --n_splits 5
```

## 08. Train 
