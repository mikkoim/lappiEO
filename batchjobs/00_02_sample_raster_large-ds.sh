#!/bin/bash
#SBATCH --job-name=large-ds
#SBATCH --account=Project_2002593
#SBATCH --partition=small
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o "o_sampleraster_large-ds.txt"
#SBATCH -e "e_sampleraster_large-ds.txt"

module load geoconda

# Samples classes from the CORINE raster
python scripts/01_preprocessing/sample_raster.py \
    --input_shp "./data/processed/vectors/large-ds_500m.shp" \
    --input_raster "./data/raw/rasters/clc2018_fi20m/Clc2018_FI20m.tif" \
    --target "rand_point" \
    --out_folder "./data/processed/samples_CORINE"

# Samples the source raster with CORINE-class
python scripts/01_preprocessing/sample_raster.py \
    --input_shp "./data/processed/samples_CORINE/Clc2018_FI20m__large-ds_500m__rand_point.shp" \
    --input_raster "./data/processed/source.tif" \
    --target "band0" \
    --dropna 0 \
    --out_folder "./data/processed/samples_CORINE"