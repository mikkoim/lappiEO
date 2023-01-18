#!/bin/bash
#SBATCH --job-name=iic_nat4
#SBATCH --account=Project_2002593
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1,nvme:16
#SBATCH -o "o_iic_nat4.txt"
#SBATCH -e "e_iic_nat4.txt"

source /scratch/project_2002593/impiomik/miniconda3/etc/profile.d/conda.sh

conda activate lappieo

cp ../../datasets/v3/v3_maastotieto.zip $TMPDIR
unzip $TMPDIR/v3_maastotieto.zip -d $TMPDIR

srun python ../script_load_maastotieto_fnames.py \
    --gdf_fname="../../vectors/buffers/buffer_maastotieto_100m_nat4_train.shp" \
    --tif_folder="$TMPDIR/v3_maastotieto" \
    --label_col='NaturaTyyp' \
    --out_fname="$TMPDIR/train.csv"

srun python ../script_load_maastotieto_fnames.py \
    --gdf_fname="../../vectors/buffers/buffer_maastotieto_100m_nat4_test.shp" \
    --tif_folder="$TMPDIR/v3_maastotieto" \
    --label_col='NaturaTyyp' \
    --out_fname="$TMPDIR/test.csv"

srun python ../script_resnet_train.py \
    --train_csv="$TMPDIR/train.csv" \
    --test_csv="$TMPDIR/test.csv" \
    --dataset="v3" \
    --img_size=49 \
    --center_crop=19 \
    --batch_size=128 \
    --epochs=500 \
    --learning_rate=1e-4 \
    --channels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13] \
    --load_dataset_to_memory \
    --freeze_base \
    --resnet_model='resnet18' \
    --pretrained_model="v3_IIC_10c70o_19x19.pt" \
    --N_channels_source=14 \
    --N_classes_source=25 \
    --pretrain_clusters=10 \
    --pretrain_overclusters=70 \
    --output_model_name="v3_100m_iic_nat4.pt" \
    --wandb_project="v3_100m"