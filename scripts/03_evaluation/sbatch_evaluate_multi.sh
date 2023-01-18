#!/bin/bash
#SBATCH --job-name=eval_multi
#SBATCH --account=Project_2002593
#SBATCH --partition=small
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -o "o_eval_multi.txt"
#SBATCH -e "e_eval_multi.txt"

source /scratch/project_2002593/impiomik/miniconda3/etc/profile.d/conda.sh
conda activate lappieo

srun python script_evaluate_multi.py