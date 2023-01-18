#!/bin/bash
#SBATCH --job-name=nat0_bc
#SBATCH --account=Project_2002593
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1,nvme:32
#SBATCH -o "o_100m_base.txt"
#SBATCH -e "e_100m_base.txt"

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
    
 srun python ../make_splits.py \
     --csv="$TMPDIR/v3_corine.csv" \
     --splits=20

srun python ../script_resnet_train.py \
    --train_csv="$TMPDIR/v3_corine1_train.csv" \
    --test_csv="$TMPDIR/v3_corine1_test.csv" \
    --dataset="v3" \
    --img_size=49 \
    --center_crop=19 \
    --batch_size=128 \
    --epochs=20 \
    --learning_rate=1e-4 \
    --channels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13] \
    --resnet_model='resnet18' \
    --load_dataset_to_memory \
    --output_model_name="v3_100m_corine1_base_2linear.pt" \
    --wandb_project="v3_100m"
