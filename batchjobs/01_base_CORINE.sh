#!/bin/bash
#SBATCH --job-name=b_cor
#SBATCH --account=Project_2002593
#SBATCH --partition=small
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=nvme:32
#SBATCH -o "o_01_base_corine.txt"
#SBATCH -e "e_01_base_corine.txt"

echo "Extracting data..."
unzip -q data/processed/datasets/source__buffer50_large-ds_500m.zip -d $TMPDIR/dataset
echo "Done"

singularity run --nv -B $SCRATCH:$SCRATCH -B $TMPDIR:$TMPDIR -H ~ lappieo.sif /opt/conda/envs/lappieo/bin/python \
    scripts/01_preprocessing/script_load_unsupervised_fnames.py \
        --gdf_fname="data/processed/samples_CORINE/Clc2018_FI20m__large-ds_500m__id.csv" \
        --tif_folder="$TMPDIR/dataset" \
        --fname_col="id" \
        --label_col="band0" \
        --out_fname="v3_corine.csv"

singularity run --nv -B $SCRATCH:$SCRATCH -B $TMPDIR:$TMPDIR -H ~ lappieo.sif /opt/conda/envs/lappieo/bin/python \
    scripts/01_preprocessing/script_load_maastotieto_fnames.py \
        --gdf_fname="../../vectors/buffers/buffer_maastotieto_100m_inv0_test.shp" \
        --tif_folder="$TMPDIR/v3_maastotieto" \
        --label_col='Inventoint' \
        --out_fname="$TMPDIR/test.csv"