from pathlib import Path
import json
from typing import Mapping

from omegaconf import OmegaConf
import wandb

import torch
from torch.utils.data import Dataset

def format_exp_name(cfg):
    name = f'{cfg.model.name}_{cfg.opt.update_algorithm}_lr_{cfg.opt.lr}'
    name += f'_k_{cfg.dataset.k}_n_{cfg.dataset.n}'
    # wd=0 runs carry no suffix, which is how the shipped results/ dirs are
    # named. Keep this conditional: it makes re-runs land on the same names as
    # the provided data, and avoids `_wd0.0` (0 is coerced to float by the
    # config), which analysis/sparse_inputs_rofk.ipynb filters out as if it
    # were a nonzero-wd run.
    if cfg.opt.wd != 0:
        name += f'_wd{cfg.opt.wd}'
    return name

def format_exp_output_dir(cfg):
    project_dir = Path(cfg.project_results_dir)
    exp_dir = f"n{cfg.dataset.n}_k{cfg.dataset.k}"
    exp_savename =  format_exp_name(cfg)
    seed_str = f'seed_{cfg.seed}'
    output_parent = project_dir/exp_dir/exp_savename/seed_str
    return output_parent

def save_config(cfg, output_parent:Path=None, 
                output_name="config.yaml",
                verbose=True, 
                dry_run=False, 
                return_path=False):
    
    if output_parent is None:
        output_parent = format_exp_output_dir(cfg)
    output_parent.mkdir(parents=True, exist_ok=True)
    save_path = output_parent/"config.yaml"
    
    if not dry_run:
        OmegaConf.save(cfg,save_path )
        if verbose:
            print("Saved config:    ")
            print(output_parent/"config.yaml")
    
    if return_path:
        return save_path

def config_exists(cfg: Mapping,
                  output_parent:Path=None, 
                  output_name="config.yaml",
                  report_exists=True)->bool:

    cfg_savepath = save_config(cfg, output_parent, output_name,
                               dry_run=True, return_path=True)
    if report_exists:
        if cfg_savepath.exists():
            print(f"Config already exists at {cfg_savepath}")
    return cfg_savepath.exists()

def save_results_dict(cfg, results_dict, 
                      output_parent:Path=None,
                      output_name="results_dict.json",
                      verbose=True):
    
    if output_parent is None:
        output_parent = format_exp_output_dir(cfg)
    output_path = output_parent/output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as fout:
        json.dump(results_dict, fout)

    if verbose:
        print("Saved result_dict:    ")
        print(output_path)



def init_wandb(cfg):
    # https://docs.wandb.ai/ref/python/init
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    run = wandb.init(project=cfg.exp.wandb_project, #entity=cfg.exp.wandb_entity, 
                     config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                     reinit=True, tags=[cfg.exp.wandb_tag])
    #os.environ['WANDB_DIR'] = str(Path.home()/ "scratch/")
    run.name = format_exp_name(cfg)
    return run

class Dataset(Dataset):
    """
    Wraps X, y numpy arrays as a torch Dataset (float32 inputs, long labels).
    """
    def __init__(self,X, y):
        super().__init__()
        self.X = X.astype("float32")
        self.y =y.astype("long")

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

def binary_acc(yhat, y, reduction="mean"):
    yhat = yhat.squeeze()
    n_correct = torch.sum(yhat==y)
    if reduction=="mean":
        return n_correct*100.0/y.size(0)
    elif reduction=="sum":
        return n_correct*100.0