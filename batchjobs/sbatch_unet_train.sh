#!/bin/bash
#SBATCH --job-name=unet
#SBATCH --account=Project_2002593
#SBATCH --partition=gpu
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1,nvme:64
#SBATCH -o "o_unet_lr1e-3.txt"
#SBATCH -e "e_unet_lr1e-3.txt"

source /scratch/project_2002593/impiomik/miniconda3/etc/profile.d/conda.sh

conda activate lappieo

cp ../../datasets/v3/v3_segtrain_35k.zip $TMPDIR
cp ../../datasets/v3/v3_segtrain_trans_inv0.zip $TMPDIR

unzip $TMPDIR/v3_segtrain_35k.zip -d $TMPDIR
unzip $TMPDIR/v3_segtrain_trans_inv0.zip -d $TMPDIR

srun python ../script_unet_train.py \
    --config='config_unet_lr-1e-3.yml' \
    --file_dir="$TMPDIR/v3_segtrain_35k" \
    --label_dir="$TMPDIR/v3_segtrain_trans_inv0" \
    --train_csv="../train_test_splits/maastotieto/v3_maastotieto_inv0_train.csv" \
    --test_csv="../train_test_splits/maastotieto/v3_maastotieto_inv0_test.csv"
