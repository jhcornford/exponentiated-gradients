#!/bin/bash
#SBATCH --array=0-4%5
#SBATCH --partition=main,long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=20GB
#SBATCH --time=3:30:00
#SBATCH --ntasks-per-gpu=1
#SBATCH --cpus-per-task=4
#SBATCH --output=sbatch_out/exp_cos_sim_anal.%A.%a.out
#SBATCH --error=sbatch_err/exp_cos_sim_anal.%A.%a.err
#SBATCH --job-name=exp_cos_sim_anal

# Load modules
printenv | grep SLURM*

echo "Loading conda"

# . load_conda.sh
source load_conda.sh

export SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:=0}
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

python analysis_1_solutions.py 
