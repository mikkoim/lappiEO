#!/bin/bash
#SBATCH --job-name=segment_resnet
#SBATCH --account=Project_2002593
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH -o "output_IIC_finetune.txt"
#SBATCH -e "error_IIC_finetune.txt"

source /scratch/project_2002593/impiomik/miniconda3/etc/profile.d/conda.sh

conda activate lappieo

srun python ../script_IIC_finetune.py \
    --train_csv="../train_test_splits/maastotieto/v3_maastotieto_nat0_train.csv" \
    --test_csv="../train_test_splits/maastotieto/v3_maastotieto_nat0_test.csv" \
    --dataset="v3" \
    --img_size=49 \
    --batch_size=128 \
    --epochs=1000 \
    --learning_rate=1e-4 \
    --channels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13] \
    --use_gpu \
    --freeze_base \
    --load_dataset_to_memory \
    --pretrained_model='../../models/v3/v3_IIC_10c70o_49x49.pt' \
    --output_model_name="v3_IIC_10c70o_49x49_finetune_nat0.pt" \
    --n_clusters=10 \
    --n_overcluster=70 \
    --center_crop=49 \
    --wandb_project="v3"
