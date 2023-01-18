#!/bin/bash
#SBATCH --job-name=IIC_30c_9x9
#SBATCH --account=Project_2002593
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1,nvme:32
#SBATCH -o "o_IIC_30c70o_9x9.txt"
#SBATCH -e "e_IIC_30c70o_9x9.txt"

source /scratch/project_2002593/impiomik/miniconda3/etc/profile.d/conda.sh

conda activate lappieo

cp ../../datasets/v3/v3_35k.zip $TMPDIR
unzip $TMPDIR/v3_35k.zip -d $TMPDIR

srun python ../script_IIC_train.py \
    --folder="$TMPDIR/v3_35k" \
    --dataset="v3" \
    --img_size=49 \
    --batch_size=128 \
    --epochs=15 \
    --resnet_model="resnet18" \
    --learning_rate=1e-4 \
    --load_dataset_to_memory \
    --channels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13] \
    --output_model_name="v3_IIC_30c70o_9x9.pt" \
    --n_clusters=30 \
    --n_overcluster=70 \
    --n_transforms=5 \
    --center_crop=19 \
    --wandb_project="v3_100m_IIC"