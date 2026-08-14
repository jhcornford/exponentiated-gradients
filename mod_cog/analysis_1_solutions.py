from pathlib import Path
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

import sys
sys.path.append('../')
from src import eg_utils
from src import analysis_utils as au

@dataclass
class DataConfig:
    batch_size: int = 512
    seq_len:int = 350
    copy_data_to_device_mem: bool = False
    num_workers: int = 0
    pin_memory: bool = True

def add_one_hot_encoding(inputs):
    task_rules = inputs[:,:,-1]
    one_hot_rule_inputs = torch.nn.functional.one_hot(task_rules.long(), num_classes=82).float()
    inputs = torch.cat([inputs[:,:,:-1], one_hot_rule_inputs], dim=-1)
    return inputs

@torch.no_grad()
def compute_acc_with_fixation_unit(outputs, labels, dec_mask):
    choices = torch.argmax(outputs, dim=-1)
    labels_for_full_acc = 1 + labels
    labels_for_full_acc[~dec_mask] = 0

    acc = torch.eq(choices, labels_for_full_acc).type(torch.float32).mean().item()
    dec_acc = torch.eq(choices[dec_mask], labels_for_full_acc[dec_mask]).type(torch.float32).mean().item()

    return acc, dec_acc

def get_subtasks_performance(model, loader, device, n_subtasks=82, debug=False):
    """
    If debug = True only iterates over 4 batches
    """
    task_rule_labels = torch.from_numpy(np.arange(n_subtasks)).to(device)
    task_wise_accs = []
    task_wise_proportions = []
    dec_accs = []
    for i, (inputs, labels) in enumerate(loader):
        print(f"{i}/{len(loader)}", end="\r")
        inputs, labels = inputs.to(device), labels.to(device)
        input_subtask_labels = inputs[:,:,-1].clone() # store the tensor of non one hot encoded task labels 
        inputs = add_one_hot_encoding(inputs).to(device)
        labels = labels.to(device).long()

        outputs = model(inputs)
        choices = torch.argmax(outputs, dim=-1)

        dec_mask = inputs[:,:,0] == 0 # fixation/decision period mask, fix=1, dec=0
        labels_for_full_acc = 1 + labels
        labels_for_full_acc[~dec_mask] = 0

        acc = torch.eq(choices, labels_for_full_acc).type(torch.float32).mean().item()
        #dec_acc = torch.eq(choices[dec_mask], labels_for_full_acc[dec_mask]).type(torch.float32).mean().item()

        dec_correct_choices = torch.eq(choices[dec_mask],labels_for_full_acc[dec_mask]).type(torch.float32)
        dec_acc = dec_correct_choices.mean()
        dec_accs.append(dec_acc.item())
        #print(acc, dec_accs, end="\r")

        subtask_labels_dec_period = input_subtask_labels[dec_mask]
        subtask_labels_bool_array = task_rule_labels[:, None] == subtask_labels_dec_period[None,:]
        # this creates a bool array of shape (n_subtasks, batch_size*seq_len) 
        # each row is a bool array of subtask label == task_label (over all time steps and batch examples)

        # broadcast the dec period correct choices array over and mult by subtasks_bool_array
        task_wise_dec_correct_choices = subtask_labels_bool_array*dec_correct_choices[None,:]

        total_taskwise_examples = subtask_labels_bool_array.sum(axis=1)
        total_taskwise_correct = task_wise_dec_correct_choices.sum(axis=1)

        task_wise_acc = total_taskwise_correct/total_taskwise_examples
        task_wise_accs.append(task_wise_acc)

        # also store the relative proportion of each task for taking a weighted average
        # to confirm code results in the same overall accuracy as dec acc
        task_wise_proportion = total_taskwise_examples/total_taskwise_examples.sum()
        task_wise_proportions.append(task_wise_proportion)

        if debug and i ==4: break

    print("\t", np.mean(dec_accs), "is mean dec acc")
    # stack the task_wise lists into n_subtasks x n_batches
    task_wise_accs = torch.stack(task_wise_accs, dim=1)
    task_wise_proportions = torch.stack(task_wise_proportions, dim=1)

    task_wise_accs = task_wise_accs.mean(dim=1) 
    task_wise_proportions = task_wise_proportions.mean(dim=1)
    assert np.allclose((task_wise_accs@task_wise_proportions).detach().cpu().numpy(), np.mean(dec_accs), atol=1e-3)

    return task_wise_accs, task_wise_proportions

def get_taskwise_performance_over_checkpoints(model, dataloader, ckpts, device, debug):
    ckpt_task_wise_accs_dict = {}
    for model_ckpt in ckpts:
        print(model_ckpt.name)
        model = au.load_ckpt(model, model_ckpt)
        model = model.to(device)
        task_wise_accs, task_wise_proportions = get_subtasks_performance(model, dataloader, device, debug=debug)
        #print(task_wise_proportions)
        ckpt_task_wise_accs_dict[model_ckpt.name] = task_wise_accs.detach().cpu().numpy()
    return ckpt_task_wise_accs_dict
    

if __name__ == "__main__":
    device = eg_utils.get_device()

    analysis_dir = Path("/network/projects/_groups/linclab_users/eg/modcog/analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    print(os.listdir(analysis_dir))

    results_dir = Path("/network/projects/_groups/linclab_users/eg/modcog/exp8_final_combined")
    exp_names = []
    for exp_name in os.listdir(results_dir):
        if exp_name.startswith("eg"):
            if "1e-07" in exp_name:exp_names.append(exp_name)
        elif exp_name.startswith("gd"):
            if "1e-06" in exp_name:exp_names.append(exp_name)
    
    # reformat exp_names above for use as a key for results dict
    # for example: eg_log_normal from eg_N2500_lr3.5_m0.925_wd1e-07_mxgn2.0_init_log_normal1.5_a1.0_lrsched_custom_loss
    exp_folders = au.DotDict({au.format_exp_as_key(f):results_dir/f for f in exp_names})
    results_dict = au.load_results(exp_folders, verbose=True)
    # Stack results across all seeds for each experiment
    for exp_name in results_dict.keys():
        results_dict[exp_name] = au.concat_seed_results(results_dict[exp_name])

    # Add colours to exps
    # https://www.datylon.com/blog/data-visualization-for-colorblind-readers
    results_dict.eg_normal.c = "#fec615" # golden yellow
    results_dict.eg_uniform.c = "green" #ff9408" # tangerine
    results_dict.eg_log_normal.c = "#f05039" # orangey red

    results_dict.gd_log_normal.c = "#1f449c" # blue
    results_dict.gd_normal.c = "#8af1fe"
    results_dict.gd_uniform.c = "#9e43a2"
    
    cfg = DataConfig()
    loaders = au.get_dataloaders(cfg)

    eg_exp_results = results_dict.eg_log_normal
    gd_exp_results = results_dict.gd_log_normal

    eg_seed_keys = [k for k in eg_exp_results.keys() if k.startswith("seed")]
    gd_seed_keys = [k for k in gd_exp_results.keys() if k.startswith("seed")]

    print(eg_seed_keys)
    print(gd_seed_keys)

    eg_dfs = []
    gd_dfs = []
    assert eg_seed_keys == gd_seed_keys
    seed_keys = eg_seed_keys

    if "SLURM_ARRAY_TASK_ID" in os.environ.keys():
        slurm_arr_idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
        print(f"SLURM_ARRAY_TASK_ID: {slurm_arr_idx}")
        seed_keys = [seed_keys[slurm_arr_idx]]
    else: 
        slurm_arr_idx = -1
        print("No SLURM_ARRAY_TASK_ID found")
        print("Running all seeds sequentially")
    print("Script slurm_arr_idx", slurm_arr_idx)

    for seed_key in seed_keys:
        print(seed_key)

        gd_model = au.get_model(gd_exp_results[seed_key].cfg)
        eg_model = au.get_model(eg_exp_results[seed_key].cfg)

        gd_dec_acc_ckpts = sorted(gd_exp_results[seed_key].dec_acc_ckpts[:]) + [gd_exp_results[seed_key].epoch_ckpts[-1]]
        eg_dec_acc_ckpts = sorted(eg_exp_results[seed_key].dec_acc_ckpts[:]) + [eg_exp_results[seed_key].epoch_ckpts[-1]]
        
        gd_task_wise_accs = get_taskwise_performance_over_checkpoints(gd_model, loaders["test"], gd_dec_acc_ckpts, device, debug=0)
        gd_df = pd.DataFrame(gd_task_wise_accs)

        eg_task_wise_accs = get_taskwise_performance_over_checkpoints(eg_model, loaders["test"], eg_dec_acc_ckpts, device, debug=0)
        eg_df = pd.DataFrame(eg_task_wise_accs)

        eg_dfs.append(eg_df)
        gd_dfs.append(gd_df)

        eg_save_path = analysis_dir/f"{seed_key}_eg_task_wise_accs.csv"
        gd_save_path = analysis_dir/f"{seed_key}_gd_task_wise_accs.csv"

        eg_df.to_csv(eg_save_path, index=False)
        gd_df.to_csv(gd_save_path, index=False)

        print("Saved results", eg_save_path)



#Paths look like:
# """
# /network/projects/_groups/linclab_users/eg/modcog/exp8_final_combined/gd_N2500_lr0.1_m0.925_wd1e-06_mxgn2.0_init_uniform_a1.0_lrsched_custom_loss/seed_7/dec_acc_ckpts/model_0_0-49.pt
# /network/projects/_groups/linclab_users/eg/modcog/exp8_final_combined/gd_N2500_lr0.1_m0.925_wd1e-06_mxgn2.0_init_uniform_a1.0_lrsched_custom_loss/seed_7/dec_acc_ckpts/model_0_0-99.pt
# /network/projects/_groups/linclab_users/eg/modcog/exp8_final_combined/gd_N2500_lr0.1_m0.925_wd1e-06_mxgn2.0_init_uniform_a1.0_lrsched_custom_loss/seed_7/dec_acc_ckpts/model_25_0-288.pt
# /network/projects/_groups/linclab_users/eg/modcog/exp8_final_combined/gd_N2500_lr0.1_m0.925_wd1e-06_mxgn2.0_init_uniform_a1.0_lrsched_custom_loss/seed_7/dec_acc_ckpts/model_35_0-350.pt
# /network/projects/_groups/linclab_users/eg/modcog/exp8_final_combined/gd_N2500_lr0.1_m0.925_wd1e-06_mxgn2.0_init_uniform_a1.0_lrsched_custom_loss/seed_7/dec_acc_ckpts/model_45_0-403.pt
# /network/projects/_groups/linclab_users/eg/modcog/exp8_final_combined/gd_N2500_lr0.1_m0.925_wd1e-06_mxgn2.0_init_uniform_a1.0_lrsched_custom_loss/seed_7/dec_acc_ckpts/model_55_0-448.pt
# /network/projects/_groups/linclab_users/eg/modcog/exp8_final_combined/gd_N2500_lr0.1_m0.925_wd1e-06_mxgn2.0_init_uniform_a1.0_lrsched_custom_loss/seed_7/dec_acc_ckpts/model_65_0-491.pt
# /network/projects/_groups/linclab_users/eg/modcog/exp8_final_combined/gd_N2500_lr0.1_m0.925_wd1e-06_mxgn2.0_init_uniform_a1.0_lrsched_custom_loss/seed_7/dec_acc_ckpts/model_75_0-540.pt
# /network/projects/_groups/linclab_users/eg/modcog/exp8_final_combined/gd_N2500_lr0.1_m0.925_wd1e-06_mxgn2.0_init_uniform_a1.0_lrsched_custom_loss/seed_7/dec_acc_ckpts/model_85_0-664.pt
# /network/projects/_groups/linclab_users/eg/modcog/exp8_final_combined/gd_N2500_lr0.1_m0.925_wd1e-06_mxgn2.0_init_uniform_a1.0_lrsched_custom_loss/seed_7/dec_acc_ckpts/model_95_1-352.pt
# """