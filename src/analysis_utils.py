from pathlib import Path
import os
import yaml

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import statsmodels.api as sm

from . import mod_cog_data_utils as mcdu
from . import eg_utils

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def recursive_dot_dict(d):
    if isinstance(d, dict):
        return DotDict({k: recursive_dot_dict(v) for k,v in d.items()})
    else:
        return d
    
def recursive_dict(d):
    if isinstance(d, dict):
        return {k: recursive_dict(v) for k,v in d.items()}
    else:
        return d
    
def concat_dict(acc, new_data, axis=0):
    """
    Dictionary concatenation function.
    """
    def to_array(x):
        if isinstance(x, np.ndarray):
            return x
        else:
            return np.asarray([x])

    for k, v in new_data.items():
        if isinstance(v, dict):
            if k in acc:
                acc[k] = concat_dict(acc[k], v, axis)
            else:
                acc[k] = concat_dict(dict(), v, axis)
        else:
            v = to_array(v)
            if axis ==1 and len(v.shape) == 1:
                v = np.expand_dims(v, 1)
            if k in acc:
                acc[k] = np.concatenate([acc[k], v], axis=axis)
            else:
                acc[k] = np.copy(v)
    return acc

def get_dataloaders(cfg):
    try:
        data_dir = Path(os.environ["SLURM_TMPDIR"])/"data"
        assert data_dir.exists()
        print(f"Using data from {data_dir}")
    except:
        print("Using data from linclab_users directory")
        data_dir = Path('/network/projects/_groups/linclab_users/eg/modcog/data_v2')
        if not data_dir.exists():
            raise FileNotFoundError(
                "Data dir not found: change the hardcoded path in get_dataloaders (src/analysis_utils.py)"
            )

    loaders = {}
    for split in ["train", "val", "test"]:
        dataset = mcdu.ModCogDataset(data_dir, split=split, in_memory=cfg.copy_data_to_device_mem)
        loaders[split] = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=split=="train",
                                    num_workers = 0 if cfg.copy_data_to_device_mem else cfg.num_workers, 
                                    pin_memory=cfg.pin_memory)

    return loaders

class RNNModelV2(torch.nn.Module):
    """
    Same as standard model but optionally returns and logs the activations over the batch.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn = torch.nn.RNNCell(input_size, hidden_size, nonlinearity='relu')
        self.fc  = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, return_acts=False):
        """
        x of shape (batch_size, seq_len, input_size)
        out_tensor of shape (batch_size, seq_len, output_size)
        """
        # activations assigned to CPU. Slice assignment below copies h_t across devices
        activations = torch.zeros(x.size(0), x.size(1), self.hidden_size) # CPU
        out_tensor = torch.zeros(x.size(0), x.size(1), self.output_size).to(x.device)
        h_t = torch.zeros(x.size(0), self.hidden_size).to(x.device)
        for t in range(x.size(1)):
            h_t = self.rnn(x[:, t, :], h_t)
            activations[:,t,:] = h_t.detach()
            out_tensor[:,t, :] = self.fc(h_t)
        if return_acts: return out_tensor, activations
        else: return out_tensor

class RNNModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn = torch.nn.RNNCell(input_size, hidden_size, nonlinearity='relu')
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        x of shape (batch_size, seq_len, input_size)
        out_tensor of shape (batch_size, seq_len, output_size)
        """
        out_tensor = torch.zeros(x.size(0), x.size(1), self.output_size).to(x.device)
        h_t = torch.zeros(x.size(0), self.hidden_size).to(x.device)
        for t in range(x.size(1)):
            h_t = self.rnn(x[:, t, :], h_t)
            out_tensor[:,t, :] = self.fc(h_t)
        return out_tensor

def get_modelv2(cfg, input_size = 115, output_size = 17):
    """ 
    Returns RNN model (v2) that can optionally return activations in the forward pass.

    Input and output sizes are default for the mod cog task
    """
    model = RNNModelV2(input_size, cfg.model.hidden_size, output_size)
    if cfg.update_alg == "eg":
        eg_utils.set_split_bias(model)
        print("Setting split bias")
    return model

def get_model(cfg, input_size = 115, output_size = 17):
    """
    Input and output sizes are default for the mod cog task
    """
    model = RNNModel(input_size, cfg.model.hidden_size, output_size)
    if cfg.update_alg == "eg":
        eg_utils.set_split_bias(model)
        print("Setting split bias")
    return model

def load_ckpt(model, ckpt_path):
    device = eg_utils.get_device()
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    return model

def format_exp_as_key(exp_name):
    s = exp_name[:2] # optimiser prefix, e.g. "eg" or "gd"
    if "log_normal" in exp_name:
        s += "_log_normal"
    elif "normal" in exp_name:
        s += "_normal"
    elif "uniform" in exp_name:
        s += "_uniform"
    return s

def sort_epoch_checkpoints(ckpt_list):
    """
    eg, ending with ...seed_5/epoch_ckpts/model_0.pt'),
    """
    return sorted(ckpt_list, key=lambda x: int(x.name.split("_")[-1].split(".")[0]))

def sort_dec_acc_checkpoints(ckpt_list):
    """
    eg, ending with .../seed_5/dec_acc_ckpts/model_35_0-426.pt)
    """
    return sorted(ckpt_list, key=lambda x: int(x.name.split("_")[1]))

def load_results(exp_folders, verbose=False):
    # could have just used exp_folders, but making an explicit results_dict
    results_dict = DotDict({})
    for exp_name, exp_folder in exp_folders.items():
        results_dict[exp_name] = DotDict({"exp_path": exp_folder})

    # load the results 
    for exp_name in results_dict.keys():
        #print(exp_name, end="")
        found, not_found = 0, 0
        seed_dirs= sorted(os.listdir(exp_folders[exp_name]), key=lambda x: int(x.split("_")[-1]))
        for seed_dir in seed_dirs[:]:
            results_dict[exp_name][seed_dir] = DotDict({"path": results_dict[exp_name].exp_path/seed_dir})

            # grab checkpoint filepaths
            epoch_ckpts_dir = results_dict[exp_name][seed_dir].path/"epoch_ckpts"
            dec_acc_ckpts_dir = results_dict[exp_name][seed_dir].path/"dec_acc_ckpts"
            results_dict[exp_name][seed_dir].epoch_ckpts = list(epoch_ckpts_dir.glob("*.pt"))
            results_dict[exp_name][seed_dir].dec_acc_ckpts = list(dec_acc_ckpts_dir.glob("*.pt"))

            if len(results_dict[exp_name][seed_dir].dec_acc_ckpts) == 0:
                print("No dec_acc_ckpts found")
            else:
                results_dict[exp_name][seed_dir].dec_acc_ckpts = sort_dec_acc_checkpoints(results_dict[exp_name][seed_dir].dec_acc_ckpts)
            
            if len(results_dict[exp_name][seed_dir].epoch_ckpts) == 0:
                print("No epoch_ckpts found")
            else:
                results_dict[exp_name][seed_dir].epoch_ckpts = sort_epoch_checkpoints(results_dict[exp_name][seed_dir].epoch_ckpts)

            # load the train and eval results
            try:
                results_dict[exp_name][seed_dir].train_results = np.load(results_dict[exp_name][seed_dir].path/"train_results.npy", allow_pickle=True)[()]
                results_dict[exp_name][seed_dir].eval_results  = np.load(results_dict[exp_name][seed_dir].path/"eval_results.npy", allow_pickle=True)[()]
                found += 1
            except:
                not_found +=1

            # load the config into a dict
            with open(results_dict[exp_name][seed_dir].path/"config.yaml", 'r') as file:
                config = DotDict(yaml.safe_load(file))
                results_dict[exp_name][seed_dir].cfg = recursive_dot_dict(config)

        if verbose: print(f" : loaded {found} results from {found+not_found}")
    return results_dict

def concat_seed_results(exp_d):
    # todo: this is not handling the r2 values correctly, concatenating them
    exp_train_results = DotDict({})
    exp_eval_results = DotDict({})
    for seed_key in exp_d.keys():
        # check is a seed_key, if not e.g. exp_path, skip
        if not seed_key.startswith("seed"): continue
        assert isinstance(exp_d[seed_key], dict)
        try:
            assert "train_results" in exp_d[seed_key]
            assert "eval_results" in exp_d[seed_key]
        except:
            print(f"Skipping {seed_key}")
            continue

        exp_train_results = concat_dict(exp_train_results, exp_d[seed_key].train_results, axis=1)
        exp_eval_results  = concat_dict(exp_eval_results, exp_d[seed_key].eval_results, axis=1)
    
    exp_d.train_results = exp_train_results
    exp_d.eval_results  = exp_eval_results
    #print(exp_d.train_results.keys())
    return DotDict(exp_d)


def find_all_matching_subfolders(path, folder:str):
    """
    path: top level folder to search through
    folder: folder name to search for
    """
    matching_folders = []
    for p in path.iterdir():
        if p.is_dir():
            if p.name == folder:
                matching_folders.append(p)
            matching_folders.extend(find_all_matching_subfolders(p, folder))
    return matching_folders

def fit_ols(x,y, standardise_by_stddev=True, verbose=False):
    assert x.ndim == 1
    assert y.ndim == 1
    assert type(x) == np.ndarray
    assert type(y) == np.ndarray

    if standardise_by_stddev:
        x = x / (np.std(x) + 1e-7)
        y = y / (np.std(y) + 1e-7)
    
    x_c = sm.add_constant(x)
    model = sm.OLS(y, x_c)
    results = model.fit()
    if verbose:
        print(results.summary())
    
    results_dict = {"const":results.params[0], "slope":results.params[1],
                    "pconst":results.pvalues[0], "pslope":results.pvalues[1],
                    "r2":results.rsquared}
    return results_dict


def analyse_ckpt_diffs(ckpt1, ckpt2, standardise_by_stddev=True):
    results_dict = {}
    
    for k in ckpt1.keys(): # todo, replace with a func that groups e.g. biases and has the set keys
        if "bias" in k: continue
        p1 = ckpt1[k].detach().cpu().numpy().ravel()
        p2 = ckpt2[k].detach().cpu().numpy().ravel()

        p_sign_flips = np.sign(p1) != np.sign(p2)
        print(f"{k} Sign flip: {p_sign_flips.sum()}/{len(p_sign_flips)}. So {p_sign_flips.sum()/len(p_sign_flips)*100:.2f}% of the weights changed sign.")
        p_abs_diff = np.abs(p2-p1) # magnitude of the weight change
        print(f"{k} L2 norm of weight change: {np.linalg.norm(p_abs_diff):.4f}")
        p1_abs = np.abs(p1)
        if standardise_by_stddev:
            p_abs_diff = p_abs_diff / np.std(p_abs_diff)
            p1_abs = p1_abs / np.std(p1_abs)
        p1_abs_c = sm.add_constant(p1_abs)
        model = sm.OLS(p_abs_diff, p1_abs_c)
        results = model.fit()
        results_dict[k] = {"const":results.params[0], "slope":results.params[1],
                           "pconst":results.pvalues[0], "pslope":results.pvalues[1],
                           "r2":results.rsquared}
        
    results_df = pd.DataFrame(results_dict)
    return results_df