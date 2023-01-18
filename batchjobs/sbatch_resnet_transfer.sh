#!/bin/bash
#SBATCH --job-name=inv0_tr_rn18
#SBATCH --account=Project_2002593
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1,nvme:16
#SBATCH -o "o_trans_rn18_inv0.txt"
#SBATCH -e "e_trans_rn18_inv0.txt"

source /scratch/project_2002593/impiomik/miniconda3/etc/profile.d/conda.sh

conda activate lappieo

cp ../../datasets/v3/v3_maastotieto.zip $TMPDIR
unzip $TMPDIR/v3_maastotieto.zip -d $TMPDIR

srun python ../script_load_maastotieto_fnames.py \
    --gdf_fname="../../vectors/buffers/buffer_Maastotieto_v7_koonti_korj.shp" \
    --tif_folder="$TMPDIR/v3_maastotieto" \
    --label_col="Inventoint" \
    --out_fname="$TMPDIR/v3_maastotieto_inv.csv"

srun python ../make_splits.py \
    --csv="$TMPDIR/v3_maastotieto_inv.csv" \
    --splits=5

srun python ../script_resnet_train.py \
    --train_csv="$TMPDIR/v3_maastotieto_inv0_train.csv" \
    --test_csv="$TMPDIR/v3_maastotieto_inv0_test.csv" \
    --dataset="v3" \
    --img_size=49 \
    --batch_size=128 \
    --epochs=1000 \
    --learning_rate=1e-4 \
    --channels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13] \
    --use_gpu \
    --load_dataset_to_memory \
    --freeze_base \
    --pretrained_model="../../models/v3/v3_corine1_base_rn18.pt" \
    --N_channels_source=14 \
    --N_classes_source=31 \
    --output_model_name="v3_trans_rn18_inv0.pt" \
    --wandb_project="v3"
    
