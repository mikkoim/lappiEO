#!/bin/bash
#SBATCH --job-name=clip
#SBATCH --account=Project_2002593
#SBATCH --partition=small
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=nvme:32
#SBATCH -o "o_clipraster.txt"
#SBATCH -e "e_clipraster.txt"

module load geoconda

python scripts/01_preprocessing/clip_raster.py \
    --shp_fname "./data/processed/vectors/buffer50_Maastotieto_v7_koonti_korj.shp" \
    --raster_fname "./data/processed/source.tif" \
    --id_col "OBJECTID_1" \
    --outfolder "$TMPDIR"

zip -q -j source__buffer50_Maastotieto_v7_koonti_korj.zip \
        $TMPDIR/source__buffer50_Maastotieto_v7_koonti_korj/*
mv source__buffer50_Maastotieto_v7_koonti_korj.zip data/processed/datasets