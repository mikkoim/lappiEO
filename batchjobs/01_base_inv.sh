#!/bin/bash
#SBATCH --job-name=inv0_bc
#SBATCH --account=Project_2002593
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1,nvme:32
#SBATCH -o "o_base_inv0.txt"
#SBATCH -e "e_base_inv0.txt"

cp ../../datasets/v3/v3_maastotieto.zip $TMPDIR
unzip $TMPDIR/v3_maastotieto.zip -d $TMPDIR

singularity run --nv -B $SCRATCH:$SCRATCH -B $TMPDIR:$TMPDIR -H ~ ../lappieo.sif /opt/conda/envs/lappieo/bin/python \
    ../script_load_maastotieto_fnames.py \
        --gdf_fname="../../vectors/buffers/buffer_maastotieto_100m_inv0_train.shp" \
        --tif_folder="$TMPDIR/v3_maastotieto" \
        --label_col='Inventoint' \
        --out_fname="$TMPDIR/train.csv"

singularity run --nv -B $SCRATCH:$SCRATCH -B $TMPDIR:$TMPDIR -H ~ ../lappieo.sif /opt/conda/envs/lappieo/bin/python \
    ../script_load_maastotieto_fnames.py \
        --gdf_fname="../../vectors/buffers/buffer_maastotieto_100m_inv0_test.shp" \
        --tif_folder="$TMPDIR/v3_maastotieto" \
        --label_col='Inventoint' \
        --out_fname="$TMPDIR/test.csv"

singularity run --nv -B $SCRATCH:$SCRATCH -B $TMPDIR:$TMPDIR -H ~ ../lappieo.sif /opt/conda/envs/lappieo/bin/python \
    ../script_resnet_train.py \
        --train_csv="$TMPDIR/train.csv" \
        --test_csv="$TMPDIR/test.csv" \
        --dataset="v3" \
        --img_size=49 \
        --center_crop=19 \
        --batch_size=128 \
        --epochs=500 \
        --learning_rate=1e-4 \
        --channels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13] \
        --resnet_model='resnet18' \
        --load_dataset_to_memory \
        --output_model_name="v3_base_inv0.pt" \
        --wandb_project="lappi_new"
