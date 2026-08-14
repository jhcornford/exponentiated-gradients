#!/bin/bash
#SBATCH --array=0-23%24
#SBATCH --partition=main,long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=20GB
#SBATCH --time=5:30:00
#SBATCH --ntasks-per-gpu=1
#SBATCH --cpus-per-task=4
#SBATCH --output=sbatch_out/exp7r.%A.%a.out
#SBATCH --error=sbatch_err/exp7r.%A.%a.err
#SBATCH --job-name=exp7

# was 0-95%32
# Load modules
printenv | grep SLURM*

echo "Loading conda"

# . load_conda.sh
source load_conda.sh

SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:=0}
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

# echo "Copying data to $SLURM_TMPDIR"
#  # Stage dataset into $SLURM_TMPDIR
# mkdir -p $SLURM_TMPDIR/data
# time cp -nr $SCRATCH/eg_paper/mod_cog_data/. $SLURM_TMPDIR/data 
# echo "Copied data to $SLURM_TMPDIR/data"

# original search space (*2 seeds *3 dists)
eg_lr_array=(2.0 3.0 3.5 4.0)
eg_lr_array=(4.5 5.0)
gd_lr_array=(0.01 0.1 0.5 1.0)
gd_lr_array=()

# combined lrs
# eg_lr_array=(4.5 5.0)
# eg_lr_array=(5.5 6.0)
eg_lr_array=(7.0 8.0)
eg_weight_decay_array=(1e-6 1e-7) 

# gd_lr_array=()
gd_weight_decay_array=(1e-5 1e-6)

momentum_array=(0.925)

tag="combined_loss" # we now have noise on inputs, hence exp8
project_dir="$SCRATCH/eg_paper/mod_cog/exp8_sweep" # final /<> becomes the project name on wandb
common_str="n_epochs=6 log_n_mean_std_ratio=1.5 max_grad_norm=2.0 lr_sched=True custom_loss=True weight_init_gain=1.0 tag=$tag project_dir=$project_dir"
run_configs=()
for distr in 'uniform' 'normal' 'log_normal'; do
    for lr in "${eg_lr_array[@]}"; do
        for weight_decay in "${eg_weight_decay_array[@]}"; do
            for momentum in "${momentum_array[@]}"; do
                run_configs+=("lr=$lr update_alg=eg seed=7 weight_decay=$weight_decay weight_distribution=$distr momentum=$momentum $common_str")
                run_configs+=("lr=$lr update_alg=eg seed=8 weight_decay=$weight_decay weight_distribution=$distr momentum=$momentum $common_str")
            done
        done
    done

    for lr in "${gd_lr_array[@]}"; do
        echo "gd_lr: $lr"
        for weight_decay in "${gd_weight_decay_array[@]}"; do
            for momentum in "${momentum_array[@]}"; do
                run_configs+=("lr=$lr update_alg=gd seed=7 weight_decay=$weight_decay weight_distribution=$distr momentum=$momentum $common_str")
                run_configs+=("lr=$lr update_alg=gd seed=8 weight_decay=$weight_decay weight_distribution=$distr momentum=$momentum $common_str")
            done
        done
    done
done

# for config in "${run_configs[@]}"; do
#     echo $config
# done

echo "len(run_configs): ${#run_configs[@]}"

run_config=${run_configs[$SLURM_ARRAY_TASK_ID]}
echo $run_config
python train_mod_cog.py $run_config
