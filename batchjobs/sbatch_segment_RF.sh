#!/bin/bash
#SBATCH --job-name=rf_saariselka
#SBATCH --account=Project_2002593
#SBATCH --partition=small
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH -o "output_paalaki.txt"
#SBATCH -e "error_paalaki.txt"

source /scratch/project_2002593/impiomik/miniconda3/etc/profile.d/conda.sh

conda activate lappieo

for i in {1..4} 
do
    srun python ../script_segment_RF.py \
        --train_csv="../train_test_splits/maastotieto/v3_maastotieto_nat.csv" \
        --test_csv="../train_test_splits/maastotieto/v3_maastotieto_nat.csv" \
        --dataset="v3" \
        --tif_folder="../../rasters/v3_paalaki/set$i" \
        --img_size=49 \
        --channels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13] \
        --out_folder="v3_paalaki_RF_natF" \
        --out_appendix="rf_nat"
done