#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --account=Project_2002593
#SBATCH --partition=small
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=nvme:16
#SBATCH -o "o_evaluation.txt"
#SBATCH -e "e_evaluation.txt"

source /scratch/project_2002593/impiomik/miniconda3/etc/profile.d/conda.sh
conda activate lappieo

cp ../../datasets/v3/*.zip $TMPDIR
unzip $TMPDIR/v3_maastotieto.zip -d $TMPDIR

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

srun python ../script_evaluate_single.py \
    --train_csv="$TMPDIR/train.csv" \
    --test_csv="$TMPDIR/test.csv" \
	--dataset='v3' \
	--model='v3_100m_distill_crop_nofrz_nat0_00.pt' \
	--img_size=49 \
    --center_crop=19 \
	--batch_size=128 \
	--load_dataset_to_memory \
	--crop \
	--resnet_model='resnet18' \
	--channels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13] \
	--N_classes_source=19 \
	--tta_guesses=5