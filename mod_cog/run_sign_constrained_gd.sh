#!/bin/bash
#SBATCH --array=0-4%5
#SBATCH --partition=main,long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=20GB
#SBATCH --time=7:30:00
#SBATCH --ntasks-per-gpu=1
#SBATCH --cpus-per-task=4
#SBATCH --output=sbatch_out/exp_cSGD.%A.%a.out
#SBATCH --error=sbatch_err/exp_cSGD.%A.%a.err
#SBATCH --job-name=cSGD

# cp -rpv /network/projects/_groups/linclab_users/eg/modcog/data_v2 \
#         /network/scratch/c/cornforj/eg_paper/mod_cog_data_v2

echo "Loading conda"
source load_conda.sh

SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:=0}
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

gd_lr_array=(0.1)
gd_weight_decay_array=(1e-6)

momentum_array=(0.925)
seeds=(5 6 7 8 9)

tag="combined_loss_csgd" 
project_dir="$SCRATCH/eg_paper/mod_cog/sign_constrained_gd" # final /<> becomes the project name on wandb
common_str="n_epochs=6 log_n_mean_std_ratio=1.5 max_grad_norm=2.0 lr_sched=True custom_loss=True weight_init_gain=1.0 tag=$tag project_dir=$project_dir"
common_str+=" save_dec_acc_ckpts=True save_checkpoints_per_epoch=True"
common_str+=" freeze_gd_signs=True"
run_configs=()
for distr in 'log_normal'; do
    for lr in "${gd_lr_array[@]}"; do
        for weight_decay in "${gd_weight_decay_array[@]}"; do
            for momentum in "${momentum_array[@]}"; do
                for seed in "${seeds[@]}"; do
                    run_configs+=("lr=$lr update_alg=gd seed=$seed weight_decay=$weight_decay weight_distribution=$distr momentum=$momentum $common_str")
                done
            done
        done
    done
done

echo "len(run_configs): ${#run_configs[@]}"

run_config=${run_configs[$SLURM_ARRAY_TASK_ID]}
echo $run_config
python train_mod_cog.py $run_config
