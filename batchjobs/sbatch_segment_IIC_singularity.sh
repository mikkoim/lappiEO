#!/bin/bash
#SBATCH --job-name=seg
#SBATCH --account=Project_2002593
#SBATCH --partition=gpu
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH -o "output_seg.txt"
#SBATCH -e "error_seg.txt"

for i in {1..4}
do
    singularity run --nv -B $SCRATCH:$SCRATCH -H ~ ../lappieo.sif /opt/conda/envs/lappieo/bin/python \
    ../script_segment_tif.py \
        --model_fname="v3_IIC_10c70o_9x9.pt" \
        --tif_folder="../../rasters/v3_kasivarsi/set$i" \
	--dataset="v3" \
        --n_cluster=10 \
	--n_overcluster=70 \
	--n_channels=14 \
        --block_size=500 \
        --center_crop=9 \
        --out_folder="./iic_10c70o_9x9" \
        --out_appendix="iic_10c70o_9x9"
done
