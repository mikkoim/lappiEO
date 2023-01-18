#!/bin/bash
#SBATCH --job-name=b_cor_randinit
#SBATCH --account=Project_2002593
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1,nvme:32
#SBATCH -o "o_01_base_corine_randinit.txt"
#SBATCH -e "e_01_base_corine_randinit.txt"

echo "Extracting data..."
unzip -q ../../datasets/v3/v3_35k.zip -d $TMPDIR
echo "Done."

singularity run --nv -B $SCRATCH:$SCRATCH -B $TMPDIR:$TMPDIR -H ~ ../lappieo.sif /opt/conda/envs/lappieo/bin/python \
    ../script_load_unsupervised_fnames.py \
        --gdf_fname="../../vectors/points/corine_35k.shp" \
        --tif_folder="$TMPDIR/v3_35k" \
        --fname_col="fid" \
        --label_col="rvalue_1" \
        --out_fname="$TMPDIR/v3_corine.csv"
        
singularity run --nv -B $SCRATCH:$SCRATCH -B $TMPDIR:$TMPDIR -H ~ ../lappieo.sif /opt/conda/envs/lappieo/bin/python \
    ../script_make_splits.py \
        --csv="$TMPDIR/v3_corine.csv" \
        --splits=20

singularity run --nv -B $SCRATCH:$SCRATCH -B $TMPDIR:$TMPDIR -H ~ ../lappieo.sif /opt/conda/envs/lappieo/bin/python \
    ../script_resnet_train.py \
        --train_csv="$TMPDIR/v3_corine0_train.csv" \
        --test_csv="$TMPDIR/v3_corine0_test.csv" \
        --dataset="v3" \
        --img_size=49 \
        --center_crop=19 \
        --batch_size=128 \
        --epochs=20 \
        --learning_rate=1e-4 \
        --channels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13] \
        --resnet_model='resnet18' \
        --random_init_weights \
        --output_model_name="v3_base_randinit_corine0.pt" \
        --wandb_project="lappi_new"
