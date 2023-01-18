#!/bin/bash
#SBATCH --job-name=clip
#SBATCH --account=Project_2002593
#SBATCH --partition=small
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --gres=nvme:32
#SBATCH -o "o_clipraster.txt"
#SBATCH -e "e_clipraster.txt"

module load geoconda

python scripts/01_preprocessing/clip_raster.py \
    --shp_fname "data/processed/vectors/buffer50_large-ds_500m.shp" \
    --raster_fname "data/processed/source.tif" \
    --id_col "rand_point" \
    --outfolder "$TMPDIR"

zip -q -j source__buffer50_large-ds_500m.zip \
        $TMPDIR/source__buffer50_large-ds_500m/*
mv source__buffer50_large-ds_500m.zip data/processed/datasets