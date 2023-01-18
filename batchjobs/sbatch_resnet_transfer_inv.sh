#!/bin/bash
#SBATCH --job-name=trans_inv0
#SBATCH --account=Project_2002593
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1,nvme:16
#SBATCH -o "o_trans_inv0.txt"
#SBATCH -e "e_trans_inv0.txt"

source /scratch/project_2002593/impiomik/miniconda3/etc/profile.d/conda.sh

conda activate lappieo

cp ../../datasets/v3/v3_maastotieto.zip $TMPDIR
unzip $TMPDIR/v3_maastotieto.zip -d $TMPDIR

srun python ../script_load_maastotieto_fnames.py \
    --gdf_fname="../../vectors/buffers/buffer_maastotieto_100m_inv0_train.shp" \
    --tif_folder="$TMPDIR/v3_maastotieto" \
    --label_col='Inventoint' \
    --out_fname="$TMPDIR/train.csv"

srun python ../script_load_maastotieto_fnames.py \
    --gdf_fname="../../vectors/buffers/buffer_maastotieto_100m_inv0_test.shp" \
    --tif_folder="$TMPDIR/v3_maastotieto" \
    --label_col='Inventoint' \
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
    --pretrained_model="v3_100m_corine1_base.pt" \
    --N_channels_source=14 \
    --N_classes_source=31 \
    --output_model_name="v3_100m_trans_inv0_test.pt" \
    --wandb_project="v3_100m"