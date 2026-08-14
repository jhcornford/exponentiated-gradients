from pathlib import Path
from tqdm import tqdm
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector
from omegaconf import OmegaConf

import sys
sys.path.append('../')
from src import eg_utils
from src import mod_cog_data_utils as mcdu
from src import analysis_utils as au
from src import prune

import wandb

from train_mod_cog import RNNModel, check_and_name_config
from train_mod_cog import ModCogLossMSE, ModCogLossCombined, compute_acc_with_fixation_unit
from train_mod_cog import get_wcc_lr_schedule, evaluate_model


def get_model(cfg):
    input_size = 115 # hardcoded for now, but should make responsive to data
    output_size = 17
    model = RNNModel(input_size, cfg.model.hidden_size, output_size)
    if cfg.update_alg == "eg":
        eg_utils.set_split_bias(model)
        print("Setting split bias")
    return model

def get_results_dict(results_dir):
    # drop larger weights for now, comparing 1e-7 eg, 1e-6 gd
    exp_names = []
    for exp_name in os.listdir(results_dir):
        if exp_name.startswith("eg"):
            if "1e-07" in exp_name:exp_names.append(exp_name)
        elif exp_name.startswith("gd"):
            if "1e-06" in exp_name:exp_names.append(exp_name)
    #print(exp_names)
    # reformat exp_name for use as a key
    exp_folders = au.DotDict({au.format_exp_as_key(f):results_dir/f for f in exp_names})
    results_dict = au.load_results(exp_folders)
    return results_dict

def get_dataloaders(cfg):
    try:
        data_dir = Path(os.environ["SLURM_TMPDIR"])/"data"
        assert data_dir.exists()
        print(f"Using data from {data_dir}")
    except:
        print("Using data from scratch directory")
        data_dir = Path(os.path.join(os.environ['SCRATCH'], "eg_paper/mod_cog_data_v2"))

    loaders = {}
    for split in ["train", "val", "test"]:
        dataset = mcdu.ModCogDataset(data_dir, split=split, in_memory=cfg.copy_data_to_device_mem)
        loaders[split] = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=split=="train",
                                    num_workers = 0 if cfg.copy_data_to_device_mem else cfg.num_workers, 
                                    pin_memory=cfg.pin_memory)

    assert cfg.steps_per_epoch == len(loaders["train"])
    input_size = dataset.input_size -1 + 82 # -1 for labels, 82 is the default (hardcoded) number of tasks in the mod cog dataset
    output_size = dataset.output_size

    print(input_size, output_size)

    return loaders

def get_non_bias_params(model):
    """returns a list of parameters that are not biases"""
    non_bias_params = []
    for n, p in model.named_parameters():
        if "bias" in n: continue
        else : non_bias_params.append(p)
    return non_bias_params

def calc_cosine_similarity(vec1, vec2):
    with torch.no_grad():
        return vec1@vec2 / (vec1.norm() * vec2.norm())
    
def get_non_bias_pruneparams(model, verbose=False):
    """
    Note torch.nn.utils.prune expects iterables of (module, parameter_name) pairs 
    """
    parameters = []
    for mod_name, mod in model.named_modules():
        #print(mod_name)
        for n, p in mod.named_parameters(recurse=False):
            #print(n)
            if "bias" in n: continue
            else : 
                parameters.append((mod, n))
                if verbose: print(f"appended {mod_name}:{n}, {p.shape} to params")

    return parameters

def prune_and_clean_prune_param(module_pnames_iterable, amount):
    prune.global_unstructured(module_pnames_iterable, pruning_method=prune.L1Unstructured, amount=amount)
    for mod, pname in module_pnames_iterable:
        prune.remove(mod, pname)

def train_model(model, dataset, opt, n_steps, lr_scheduler=None, 
                use_wandb=True, custom_loss=False, max_grad_norm=None, 
                results_dict=None, cfg=None, log_parameter_stats=False):
    
    # Save a copy of model params to see how different the solution is after learning
    initial_params = parameters_to_vector(model.parameters()).detach()

    if results_dict is None: results_dict = {}
    device = eg_utils.get_device()
    model.train()

    if cfg.mse_only_custom_loss:
        full_loss = ModCogLossMSE()
    else:
        full_loss = ModCogLossCombined(label_smoothing=0.1)
    ce_criterion = torch.nn.CrossEntropyLoss()

    if type(dataset) == DataLoader: pbar = tqdm(dataset) # using cached data
    else: pbar = tqdm(range(n_steps)) # using neurogym data obj

    for update_idx, obj in enumerate(pbar):
        # Forward pass
        if type(obj) == list: # from pytorch dataloader
            inputs, labels = obj
            inputs = inputs.to(device)
            # in dataset will need to add the labels uncoding
            task_rules = inputs[:,:,-1]
            one_hot_rule_inputs = torch.nn.functional.one_hot(task_rules.long(), num_classes=82).float()
            inputs = torch.cat([inputs[:,:,:-1], one_hot_rule_inputs], dim=-1)
            inputs = inputs.to(device)
            labels = labels.to(device).long()
        else:
            inputs, labels = dataset()
            inputs = torch.from_numpy(inputs).float().to(device)
            labels = torch.from_numpy(labels).to(device)
        opt.zero_grad(set_to_none=True)

        outputs = model(inputs + torch.normal(mean=0.0, std=0.1, size=inputs.size(), device=device))
        dec_mask = inputs[:, :, 0] == 0

        acc, dec_acc = compute_acc_with_fixation_unit(outputs, labels, dec_mask)
        
        # Backward pass
        if custom_loss:
            loss, loss_fix_no_grad, loss_dec_no_grad = full_loss(outputs, labels, dec_mask)
        else:
            loss = ce_criterion(input=outputs.permute(0,2,1), target=labels)
        
        loss.backward()
        # Gradient clipping
        with torch.no_grad():
            grads_norm_before_clip = 0
            for p in model.parameters():
                if p.grad is not None:
                    grads_norm_before_clip += p.grad.flatten().norm().item() ** 2

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, norm_type=2)
                grads_norm_after_clip = 0
                for p in model.parameters():
                    if p.grad is not None:
                        grads_norm_after_clip += p.grad.flatten().norm().item() ** 2
            else:
                grads_norm_after_clip = grads_norm_before_clip
            
            grads_norm_before_clip = np.sqrt(grads_norm_before_clip)
            grads_norm_after_clip = np.sqrt(grads_norm_after_clip)

        # Update params and get new learning rate
        opt.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
            lr = lr_scheduler.get_last_lr()[0]
        else:
            lr = opt.param_groups[0]['lr']
                    
        # Logging logic
        pbar.update(1)
        pbar.set_description(f"Training... lr:{lr:.2f} gn{grads_norm_before_clip:.2f}")
        pbar.set_postfix({"Loss": loss.item(), "mse_fix": loss_fix_no_grad,
                           "Acc": acc, "Dex Acc": dec_acc})
        
        if "Loss_auc" not in results_dict.keys(): loss_auc = 0
        else: loss_auc = results_dict["Loss_auc"][-1]
        if "Dec_Acc_auc" not in results_dict.keys(): dec_acc_auc = 0
        else: dec_acc_auc = results_dict["Dec_Acc_auc"][-1]
        
        with torch.no_grad():
            updated_params = parameters_to_vector(model.parameters())
            cos_sim_init_params = calc_cosine_similarity(initial_params, updated_params)

        batch_results_dict = {"Loss": loss.item(), 
                              "Loss_auc": loss.item()+loss_auc,
                              "Mse_fixation": loss_fix_no_grad,
                              "Acc": acc, 
                              "Dec Acc": dec_acc,
                              "Dec_Acc_auc": dec_acc+dec_acc_auc,
                              "lr": lr,
                              "grads_norm_before_clip":grads_norm_before_clip,
                              "grads_norm_after_clip":grads_norm_after_clip,
                              "cos_sim_init_params": cos_sim_init_params.item()}
        
        results_dict = au.concat_dict(results_dict, batch_results_dict)
        if use_wandb: 
            wandb.log(batch_results_dict)
    return results_dict

    
if __name__ == '__main__':
    if "SLURM_ARRAY_TASK_ID" in os.environ.keys():
        slurm_arr_idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
        print(f"SLURM_ARRAY_TASK_ID: {slurm_arr_idx}")
    else: 
        slurm_arr_idx = -1
    print("Script slurm_arr_idx", slurm_arr_idx)
    
    # Grab pretrained model checkpoints
    scratch = Path(os.environ['SCRATCH'])
    results_dir = scratch/"eg_paper/mod_cog/sign_constrained_gd"
    results_dict = get_results_dict(results_dir)
    print("Results loaded", results_dict.keys())
    
    gd_exps_dict = results_dict.gd_log_normal # can pass in as an arg

    n_seeds = 5 # n experiments out of 5 total to run
    gd_exp_list = [d for k,d in gd_exps_dict.items() if k.startswith("seed")]
    all_exps_list = gd_exp_list[:n_seeds]
    
    # Set up experiment
    prune_props = [0.9, 0.925, 0.95, 0.975, 0.99] # 5 x 5 seeds = 25 experiments
    gd_lrs = [0.01, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75] # 7 
    momentum_arr = [0.9, 0.925] # x 2 

    settings = []
    for m in momentum_arr:
        for pp in prune_props:
            for lr in gd_lrs:
                for exp in gd_exp_list[:n_seeds]:
                    settings.append((exp, pp, lr, m))

    print(f"N experiments: {len(settings)}")  
    # n_seeds * len(prune_props) * (len(gd_lrs) + len(eg_lrs)) * len(momentum_arr) = (25 * 5 * 7 * 2) + (25 * 5 * 8 * 2)
    
    exp, prune_prop, lr, m = settings[slurm_arr_idx]

    cfg = exp.cfg
    cfg.prune_prop = prune_prop
    cfg.lr = lr
    cfg.momentum = m
    cfg.n_epochs = 1
    cfg.use_wandb = True
    cfg.project_dir = Path("/network/scratch/p/pingsheng.li/eg_paper/mod_cog/exp8_cSGD_pruning_cooldown_1123")
    cfg.tag = "pruning_cooldown_1123"
    exp_name = check_and_name_config(cfg)
    exp_name += f"_ff{prune_prop}"
    exp_dir = cfg.project_dir / exp_name / f"seed_{cfg.seed}"
    print(exp_dir)
    os.makedirs(exp_dir, exist_ok=True)

    overwrite = True # Set False to prevent re-running previous experiments 
    # e.g change this if adding runs to a sweep shell file.
    if not overwrite:
        if (exp_dir/"train_results.npy").exists():
            print("Overwrite is False and training results already exists")
            raise ValueError(f"Training results already exists: {exp_dir}")
    OmegaConf.save(au.recursive_dict(cfg),exp_dir/"config.yaml")


    if cfg.use_wandb:
        os.environ['WANDB_DIR'] = str(Path.home()/ "scratch/")
        run = wandb.init(reinit=False, 
                        name=exp_name,
                        project=cfg.project_dir.name,
                        tags = [cfg.tag],
                        config=cfg)

    # Load model and prune
    ckpt = exp.epoch_ckpts[-2] # -2 for the model before lr cooldown
    model = get_model(cfg).to(eg_utils.get_device())
    au.load_ckpt(model, ckpt)
    to_prune = get_non_bias_pruneparams(model)
    prune.global_unstructured(to_prune, 
                              pruning_method=prune.L1Unstructured,
                              amount=prune_prop)

    # Set up training
    loaders = get_dataloaders(cfg)
    opt = eg_utils.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                       weight_decay=cfg.weight_decay, update_alg=cfg.update_alg,
                       freeze_gd_signs=True,) # clamp_sum=False)
    
    scheduler = get_wcc_lr_schedule(opt,cfg, lr_peak_iter=0, lr_cooldown_niters=1000)

    # Evaluate 
    run_eval = True # set false to speed up debugging
    eval_results_dict = {} # only used if using cached data
    if run_eval:
        eval_dict = evaluate_model(0, model, loaders,  use_wandb=cfg.use_wandb,
                                    custom_loss=cfg.custom_loss, cfg=cfg)
        eval_results_dict = au.concat_dict(eval_results_dict, eval_dict)

    # Train model
    train_results_dict = {}
    train_results_dict = train_model(model, loaders["train"], opt, n_steps=cfg.steps_per_epoch,
                                     lr_scheduler=scheduler, max_grad_norm=cfg.max_grad_norm,
                                     use_wandb=cfg.use_wandb, custom_loss=cfg.custom_loss,
                                     results_dict=train_results_dict, cfg=cfg)
    

    # Save model and run eval again
    ckpt_path = exp_dir /"epoch_ckpts"/  f"model_{1}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    print(f"saved model to {ckpt_path}")
            
    if run_eval:
        eval_dict = evaluate_model(1, model, loaders,  use_wandb=cfg.use_wandb,
                                    custom_loss=cfg.custom_loss, cfg=cfg)
        eval_results_dict = au.concat_dict(eval_results_dict, eval_dict)

    if cfg.use_wandb:
        wandb.finish()
    np.save(exp_dir/"train_results.npy", train_results_dict)
    np.save(exp_dir/"eval_results.npy", eval_results_dict)
