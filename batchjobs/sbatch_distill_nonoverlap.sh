#!/bin/bash
#SBATCH --job-name=nat0_dist
#SBATCH --account=Project_2002593
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1,nvme:32
#SBATCH -o "o_dist_crop_nofrz_nat0.txt"
#SBATCH -e "e_dist_crop_nofrz_nat0.txt"

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
    --gdf_fname="../../vectors/buffers/buffer_maastotieto_100m_nat0_train.shp" \
    --tif_folder="$TMPDIR/v3_maastotieto" \
    --label_col='NaturaTyyp' \
    --out_fname="$TMPDIR/train.csv"

srun python ../script_load_maastotieto_fnames.py \
    --gdf_fname="../../vectors/buffers/buffer_maastotieto_100m_nat0_test.shp" \
    --tif_folder="$TMPDIR/v3_maastotieto" \
    --label_col='NaturaTyyp' \
    --out_fname="$TMPDIR/test.csv"
 
srun python ../script_distillation.py \
    --train_csv="$TMPDIR/train.csv" \
    --test_csv="$TMPDIR/test.csv" \
    --dataset="v3" \
    --external_data_csv="$TMPDIR/v3_corine.csv" \
    --img_size=49 \
    --center_crop=19 \
    --batch_size=128 \
    --epochs=500 \
    --iterations=1 \
    --learning_rate=1e-4 \
    --channels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13] \
    --use_gpu \
    --load_dataset_to_memory \
    --resnet_model='resnet18' \
    --student_model='resnet18' \
    --pretrained_model="delet_v3_100m_trans_crop_nofrz_nat0.pt" \
    --N_channels_source=14 \
    --N_classes_source=19 \
    --output_model_name="delet_v3_100m_distill_crop_nofrz_nat0" \
    --wandb_project="v3_100m"
