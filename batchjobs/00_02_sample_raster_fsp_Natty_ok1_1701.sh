#!/bin/bash
#SBATCH --job-name=annot
#SBATCH --account=Project_2002593
#SBATCH --partition=small
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -o "o_sampleraster_annot.txt"
#SBATCH -e "e_sampleraster_annot.txt"

module load geoconda

python scripts/01_preprocessing/sample_raster.py \
    --input_shp "./data/raw/vectors/fsp_Natty_ok1_1701/fsp_Natty_ok1_1701.shp" \
    --input_raster "./data/processed/source.tif" \
    --target "NaturaTyyp" \
    --dropna 0 \
    --out_folder "./data/processed/samples"

python scripts/01_preprocessing/sample_raster.py \
    --input_shp "./data/raw/vectors/fsp_Natty_ok1_1701/fsp_Natty_ok1_1701.shp" \
    --input_raster "./data/processed/source.tif" \
    --target "Inventoint" \
    --dropna 0 \
    --out_folder "./data/processed/samples"