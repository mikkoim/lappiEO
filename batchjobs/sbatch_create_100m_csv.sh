#!/bin/bash
#SBATCH --job-name=csv
#SBATCH --account=Project_2002593
#SBATCH --partition=small
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -o "o.txt"
#SBATCH -e "e.txt"

source /scratch/project_2002593/impiomik/miniconda3/etc/profile.d/conda.sh
conda activate lappieo

for i in {0..5}
do
    srun python ../script_load_maastotieto_fnames.py \
        --gdf_fname="../../vectors/buffers/buffer_maastotieto_100m_nat${i}_train.shp" \
        --tif_folder="../../datasets/v3/v3_maastotieto" \
        --label_col='NaturaTyyp' \
        --out_fname="100m_nat${i}_train.csv"

    srun python ../script_load_maastotieto_fnames.py \
        --gdf_fname="../../vectors/buffers/buffer_maastotieto_100m_nat${i}_test.shp" \
        --tif_folder="../../datasets/v3/v3_maastotieto" \
        --label_col='NaturaTyyp' \
        --out_fname="100m_nat${i}_test.csv"
done