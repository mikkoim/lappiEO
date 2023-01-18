#!/bin/bash
#SBATCH --job-name=trans_nat
#SBATCH --account=Project_2002593
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1,nvme:16
#SBATCH -o "o_trans_nat.txt"
#SBATCH -e "e_trans_nat.txt"

echo "Extracting data..."
unzip -q ../../datasets/v3/v3_maastotieto.zip -d $TMPDIR
echo "Done."

for i in {0..4}
do
singularity run --nv -B $SCRATCH:$SCRATCH -B $TMPDIR:$TMPDIR -H ~ ../lappieo.sif /opt/conda/envs/lappieo/bin/python \
    ../script_load_maastotieto_fnames.py \
        --gdf_fname="../../vectors/buffers/buffer_maastotieto_100m_nat${i}_train.shp" \
        --tif_folder="$TMPDIR/v3_maastotieto" \
        --label_col="NaturaTyyp" \
        --out_fname="$TMPDIR/train.csv"

singularity run --nv -B $SCRATCH:$SCRATCH -B $TMPDIR:$TMPDIR -H ~ ../lappieo.sif /opt/conda/envs/lappieo/bin/python \
    ../script_load_maastotieto_fnames.py \
        --gdf_fname="../../vectors/buffers/buffer_maastotieto_100m_nat${i}_test.shp" \
        --tif_folder="$TMPDIR/v3_maastotieto" \
        --label_col="NaturaTyyp" \
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
        --load_dataset_to_memory \
        --freeze_base \
        --resnet_model='resnet18' \
        --pretrained_model="v3_base_randinit_corine0.pt" \
        --N_channels_source=14 \
        --N_classes_source=31 \
        --output_model_name="v3_transfer_nat${i}" \
        --wandb_project="lappi_new"
done