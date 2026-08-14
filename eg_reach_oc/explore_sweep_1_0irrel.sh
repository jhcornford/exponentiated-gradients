#!/bin/bash
#SBATCH --array=0-7%8
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
#SBATCH --time=12:30:00
#SBATCH --ntasks-per-gpu=3
#SBATCH --cpus-per-task=2
#SBATCH --output=sbatch_out/es1_0iire.%A.%a.out
#SBATCH --error=sbatch_err/es1_0iire.%A.%a.err
#SBATCH --job-name=es1_0iire

source load_conda.sh

SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:=0}
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

eg_lr_array=(1.5 1.25 1 0.75) 
eg_weight_decay_array=(1e-6)

gd_lr_array=(0.3 0.2 0.1 0.05) 
gd_weight_decay_array=(1e-5)

momentum_array=(0.925)

max_grad_norm_array_eg=(4) 
max_grad_norm_array_gd=(4) 

n_irrel_feats_array=(0) 
n_epochs=6000
optimiser="sgd"

tag="es1_2609_0irrel_ln_cosine"

exp_dir="/network/projects/_groups/linclab_users/eg/eg_reach_oc_debug/exp1_sweep"

common_str="n_epochs=$n_epochs opt.lr_scheduler=True opt.optimiser=$optimiser " 
common_str+="opt.action_reg_lambda=0.1 opt.hidden_reg_lambda=0.1 "
common_str+="task_name=reach_exploring exp_dir=$exp_dir "
common_str+="logging.wandb_tag=$tag logging.log_plots=True logging.use_wandb=True "
common_str+="logging.save_checkpoints_every_1k=False "
common_str+="logging.log_local=False "
common_str+="exp.irrel_noise_type=OUmomentum exp.irrel_noise_theta=0.0005 "

seeds_array=(7)

run_configs=()
for seed in "${seeds_array[@]}"; do
for max_grad_norm in "${max_grad_norm_array_eg[@]}"; do
    for n_irrel_feats in "${n_irrel_feats_array[@]}"; do
        for momentum in "${momentum_array[@]}"; do
            for weight_decay in "${eg_weight_decay_array[@]}"; do
                for lr in "${eg_lr_array[@]}"; do
                    config_str="opt.update_algorithm=eg opt.lr=$lr opt.wd=$weight_decay opt.momentum=$momentum seed=$seed "
                    config_str+="opt.max_grad_norm=$max_grad_norm exp.n_irrel_feats=$n_irrel_feats $common_str"
                    config_str+="opt.lr_scheduler=False" # False is cosine decay
                    run_configs+=("$config_str")
                done
            done
        done
    done
done

for max_grad_norm in "${max_grad_norm_array_gd[@]}"; do
    for n_irrel_feats in "${n_irrel_feats_array[@]}"; do
        for momentum in "${momentum_array[@]}"; do
            for weight_decay in "${gd_weight_decay_array[@]}"; do
                for lr in "${gd_lr_array[@]}"; do
                    config_str="opt.update_algorithm=gd opt.lr=$lr opt.wd=$weight_decay opt.momentum=$momentum seed=$seed "
                    config_str+="opt.max_grad_norm=$max_grad_norm exp.n_irrel_feats=$n_irrel_feats $common_str"
                    config_str+="opt.lr_scheduler=False" # False is cosine decay
                    run_configs+=("$config_str")
                done
            done
        done
    done
done
done


echo "len(run_configs): ${#run_configs[@]}"

run_config=${run_configs[$SLURM_ARRAY_TASK_ID]}
echo $run_config
unset SLURM_TRES_PER_TASK
srun python train_controller.py $run_config
#python train_controller.py $run_config