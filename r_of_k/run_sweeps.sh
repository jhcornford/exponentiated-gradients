#!/bin/bash
#SBATCH --array=0-5%6
#SBATCH --partition=main,long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=24:00:00
#SBATCH --mem=12G
#SBATCH --job-name=rofk
#SBATCH --output=sbatch_out/rofk.%A.%a.out
#SBATCH --error=sbatch_err/rofk.%A.%a.err

# Submit from inside r_of_k/. The sbatch_out/ and sbatch_err/ dirs must exist
# before submitting (mkdir -p sbatch_out sbatch_err), or SLURM kills the job.
. ../load_conda.sh
SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:=0}
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

# Relative to r_of_k/ (submit from there). 
project_results_dir="results"
# note change wandb tags in main_config.yaml

sweep_name_arr=(mlp_gd mlp_eg)
n_settings_arr=(200 2000 20000)
run_config_strs=()
for sweep_name in ${sweep_name_arr[@]}; do
    for n in ${n_settings_arr[@]}; do
        run_config_strs+=("+sweep=$sweep_name dataset.n=$n")
    done
done

# run the python file
echo $SLURM_ARRAY_TASK_ID
echo ${run_config_strs[$SLURM_ARRAY_TASK_ID]}
python train.py -m ${run_config_strs[$SLURM_ARRAY_TASK_ID]} project_results_dir=$project_results_dir