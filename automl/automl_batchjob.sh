#!/bin/bash
#SBATCH --job-name=autosklearn
#SBATCH --account=Project_2002593
#SBATCH --partition=small
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=nvme:32
#SBATCH -o "o_autosklearn.txt"
#SBATCH -e "e_autosklearn.txt"


module load python-data

python automl_train.py \
    --input 'train.csv' \
    --output 'autosklearn' \
    --max_time 60 \
    --autosklearn_version 1 \
    --n_jobs -1 \
    --tmpdir $TMPDIR/autosklearn2