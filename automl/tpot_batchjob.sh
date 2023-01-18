#!/bin/bash
#SBATCH --job-name=tpot
#SBATCH --account=Project_2002593
#SBATCH --partition=small
#SBATCH --time=18:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=nvme:32
#SBATCH -o "o_tpot.txt"
#SBATCH -e "e_tpot.txt"


module load python-data

python tpot_train.py \
    --input 'train.csv' \
    --output 'tpot_100gen_50pop' \
    --generations 100 \
    --population_size 50 \
    --scoring 'f1_weighted'
