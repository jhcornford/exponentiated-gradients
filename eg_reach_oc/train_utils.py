from pathlib import Path
import random
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
import json
import hashlib


def set_seed_all(seed):
    """
    Sets all random states
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def hash_config(cfg):
    """
    Hashes the config file to use as a unique identifier for the experiment.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    cfg_json = json.dumps(cfg_dict, sort_keys=True)
    cfg_hash = hashlib.sha3_224(cfg_json.encode()).hexdigest()
    return cfg_hash
 
def format_exp_name(cfg):
    """Run dir name. Omits reg lambdas, noise params, and weight init - not varied in the sweeps."""
    name = "rnn_"
    #name += "-ln_" if cfg.model.use_layernorm else "_"
    name += str(cfg.model.hidden_size)
    #name += f"{cfg.model.weight_distribution}_g{cfg.model.weight_init_gain}"

    if cfg.opt.optimiser == "adamw":
        name += f'_adamw-{cfg.opt.update_algorithm}_lr{cfg.opt.lr}'
    elif cfg.opt.optimiser == "sgd":
        name += f'_s{cfg.opt.update_algorithm}_lr{cfg.opt.lr}_m{cfg.opt.momentum}'
    else:
        raise NotImplementedError
    name += f'_wd{cfg.opt.wd}'
    #name += f'_ar{cfg.opt.action_reg_lambda}_hr{cfg.opt.hidden_reg_lambda}'
    name += f'_mxn{cfg.opt.max_grad_norm}'
    name += f'_bs{cfg.batch_size}'
    name += f'_linearcooldown' if cfg.opt.lr_scheduler else '_cosinecooldown'
    name += f'_nirrel{cfg.exp.n_irrel_feats}'
    name += f'_updates{cfg.n_epochs}'
    if cfg.exp.relative_distance_loss:
        name += f'_rdl'
    if cfg.exp.time_weighted_err:
        name += f'_twe'
    if cfg.model.layernorm:
        name += f'_ln'
    if cfg.opt.truncation:
        name += f"_trunc{cfg.opt.truncation}"
        if cfg.opt.truncation_updates:
            name += f"u"
        
    #if _m:{cfg.opt.momentum}'
    #name += f"_k{cfg.model.lognormal_k}"
    return name

def init_wandb(cfg):
    # https://docs.wandb.ai/ref/python/init
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    run = wandb.init(project=cfg.task_name, #entity=cfg.exp.wandb_entity, 
                     config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                     reinit=True, tags=[cfg.logging.wandb_tag])
    #os.environ['WANDB_DIR'] = str(Path.home()/ "scratch/")
    run.name = format_exp_name(cfg)
    return run

def format_exp_output_dir(cfg):
    exp_dir = Path(cfg.exp_dir)
    if "hash_exp_name" in cfg.logging.keys() and cfg.logging.hash_exp_name:
        exp_savename = hash_config(cfg)
        print("Train utils - Hashed config: ", exp_savename)
    else:
        exp_savename =  format_exp_name(cfg)
    seed_str = f'seed_{cfg.seed}'
    output_parent = exp_dir/exp_savename/seed_str
    return output_parent

def save_config(cfg, output_parent:Path=None, 
                output_name="config.yaml",
                verbose=True, 
                dry_run=False, 
                return_path=False):
    
    if output_parent is None:
        output_parent = format_exp_output_dir(cfg)
    output_parent.mkdir(parents=True, exist_ok=True)
    save_path = output_parent/output_name
    
    if not dry_run:
        OmegaConf.save(cfg,save_path )
        if verbose:
            print("Saved config:    ")
            print(output_parent/output_name)
    
    if return_path:
        return save_path
