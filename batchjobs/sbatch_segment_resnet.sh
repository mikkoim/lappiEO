#!/bin/bash
#SBATCH --job-name=invF_kasivarsi
#SBATCH --account=Project_2002593
#SBATCH --partition=gpu
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1
#SBATCH -o "o_invF_kasivarsi.txt"
#SBATCH -e "e_invF_kasivarsi.txt"

source /scratch/project_2002593/impiomik/miniconda3/etc/profile.d/conda.sh

conda activate lappieo

for i in {2..4} 
do
    srun python ../script_segment_resnet.py \
        --model_fname="v3_100m_trans_invF.pt" \
        --tif_folder="../../rasters/v3_kasivarsi/set$i" \
        --dataset="v3" \
        --n_channels=14 \
        --n_classes=36 \
        --block_size=100 \
        --input_size=19 \
        --out_folder="./seg_v3_trans_invF" \
        --out_appendix="seg_trans_invF"
done

