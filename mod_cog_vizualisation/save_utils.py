
from pathlib import Path
import os

from matplotlib import gridspec
import seaborn as sns
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.utils.data import DataLoader

import sys
sys.path.append('../../')
sys.path.append('../')

from src import mod_cog_tasks as mct
from src import mod_cog_data_utils as mcdu
from src import analysis_utils as au
from src import plot_utils as pu
from src.mod_cog_tasks import *
from src import eg_utils

import neurogym as ngym
from neurogym import spaces
from neurogym.wrappers.block import ScheduleEnvs
from neurogym.wrappers import ScheduleEnvs
from neurogym.utils.scheduler import SequentialSchedule
from neurogym.utils.scheduler import RandomSchedule
from neurogym.wrappers.block import MultiEnvs
from neurogym.utils import scheduler
from neurogym.core import TrialWrapper

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def get_results_dict(results_dir):
    exp_names = []
    # these weight decays result in similar size weight mats
    # and random on perf. (i.e. both learn perfectly)
    for exp_name in os.listdir(results_dir):
        if exp_name.startswith("eg"):
            if "1e-07" in exp_name:exp_names.append(exp_name)
        elif exp_name.startswith("gd"):
            if "1e-06" in exp_name:exp_names.append(exp_name)
    exp_folders = au.DotDict({au.format_exp_as_key(f):results_dir/f for f in exp_names})
    results_dict = au.load_results(exp_folders)
    return results_dict

def load_model(run, final_ckpt=True):
    """
    Loads model and optionaly final checkpoint

    Args:
        run : results object
        final_ckpt: bool, load final weights
    """
    model = au.get_modelv2(run.cfg)

    assert str(run.epoch_ckpts[-1]).endswith("6.pt")
    model = au.load_ckpt(model, run.epoch_ckpts[-1])

    device = eg_utils.get_device()
    model = model.to(device)

    return model


@torch.no_grad()
def compute_acc_with_fixation_unit(outputs, labels, dec_mask):
    choices = torch.argmax(outputs, dim=-1)
    labels_for_full_acc = 1 + labels
    labels_for_full_acc[~dec_mask] = 0
    acc = torch.eq(choices, labels_for_full_acc).type(torch.float32).mean().item()
    dec_acc = torch.eq(choices[dec_mask], labels_for_full_acc[dec_mask]).type(torch.float32).mean().item()
    return acc, dec_acc

def add_one_hot_encoding(inputs):
    task_rules = inputs[:,:,-1]
    one_hot_rule_inputs = torch.nn.functional.one_hot(task_rules.long(), num_classes=82).float()
    inputs = torch.cat([inputs[:,:,:-1], one_hot_rule_inputs], dim=-1)
    return inputs

def construct_batch(env, task_id, batch_size=50):
    device = eg_utils.get_device()
    inputs = []
    targets = []
    for i in range(batch_size):
        env.new_trial()
        inputs.append(env.ob[None, ...])
        targets.append(env.gt[None, ...])

    inputs_batch = np.vstack(inputs)
    targets_batch = np.vstack(targets)
    bs, seq_len, _ = inputs_batch.shape

    inputs_batch_with_label = np.concatenate([inputs_batch, np.ones(shape=[bs, seq_len, 1]) * task_id], axis=-1)
    inputs_torch = torch.from_numpy(inputs_batch_with_label.astype(np.float32))
    inputs_onehot = add_one_hot_encoding(inputs_torch).to(device)
    return inputs_onehot, targets_batch

def run_task(env, task_id, model, add_noise=True):
    inputs, targets = construct_batch(env, task_id, batch_size=1) #reduce for stack 50
    device = get_device()
    if add_noise:
        noise = torch.normal(mean=0.0, std=0.1, size=inputs.size(), device=device)
        outputs, activations = model(inputs + noise , return_acts=True)
    else:
        outputs, activations = model(inputs, return_acts=True)

    dec_mask = inputs[:, :, 0] == 0
    acc, dec_acc = compute_acc_with_fixation_unit(outputs, torch.from_numpy(targets).to(device), dec_mask)

    return outputs, activations, acc, dec_acc

def loop_over_tasks(model, add_noise=True, verbose=True):
    env_strs = mct.get_mod_cog_dataset_labels()
    envs = get_mod_cog_dataset_envs()

    for task_id, name in enumerate(env_strs[:]):
        env = envs[task_id]
        try:
            outputs, activations, acc, dec_acc = run_task(env, task_id, model, add_noise=add_noise)
        except ValueError:
            print("Todo: Batching error, different lengths")
            continue

        if verbose:
            print(env_strs[task_id], name, task_id)
            print(f"Acc {acc:.2f}, Dec Acc {dec_acc:.2f}")
            print("-----")
