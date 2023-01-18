#!/bin/bash
#SBATCH --job-name=inv4
#SBATCH --account=Project_2002593
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1,nvme:16
#SBATCH -o "o_distill_crop_maasto_inv4.txt"
#SBATCH -e "e_distill_crop_maasto_inv4.txt"

source /scratch/project_2002593/impiomik/miniconda3/etc/profile.d/conda.sh

conda activate lappieo

cp ../../datasets/v3/*.zip $TMPDIR
unzip $TMPDIR/v3_maastotieto.zip -d $TMPDIR
unzip $TMPDIR/v3_35k.zip -d $TMPDIR

srun python ../script_load_unsupervised_fnames.py \
    --gdf_fname="../../vectors/points/corine_35k.shp" \
    --tif_folder="$TMPDIR/v3_35k" \
    --fname_col="fid" \
    --label_col="rvalue_1" \
    --out_fname="$TMPDIR/v3_corine.csv"

srun python ../script_load_maastotieto_fnames.py \
    --gdf_fname="../../vectors/buffers/buffer_Maastotieto_v7_koonti_korj.shp" \
    --tif_folder="$TMPDIR/v3_maastotieto" \
    --label_col="Inventoint" \
    --out_fname="$TMPDIR/v3_maastotieto_inv.csv"

srun python ../make_splits.py \
    --csv="$TMPDIR/v3_maastotieto_inv.csv" \
    --splits=5
 
srun python ../script_distillation.py \
    --train_csv="$TMPDIR/v3_maastotieto_inv4_train.csv" \
    --test_csv="$TMPDIR/v3_maastotieto_inv4_test.csv" \
    --dataset="v3" \
    --external_data_csv="$TMPDIR/v3_corine.csv" \
    --img_size=49 \
    --batch_size=128 \
    --epochs=500 \
    --iterations=1 \
    --learning_rate=1e-4 \
    --channels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13] \
    --use_gpu \
    --load_dataset_to_memory \
    --random_crop \
    --pretrained_model="v3_trans_crop_maasto_inv4.pt" \
    --N_channels_source=14 \
    --N_classes_source=36 \
    --output_model_name="v3_distill_crop_maasto_inv4" \
    --wandb_project="v3"
