from dataclasses import dataclass
from typing import Optional
import time
from pathlib import Path
from tqdm import tqdm
import os

import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import hydra 
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore

import sys
sys.path.append('../')
from src import mod_cog_tasks as mct, eg_utils
from src import mod_cog_data_utils as mcdu
from src import analysis_utils as au
from src import initializations

import wandb

@dataclass
class ModelConfig:
    hidden_size: int = 2500

@dataclass
class Config:
    seed: int = 7
    batch_size: int = 512
    seq_len:int = 350
    model: ModelConfig = ModelConfig()
    steps_per_epoch: int = 1000
    n_epochs: int = 2
    lr: float = 5.0
    lr_sched: bool = True
    momentum: float = 0.9
    weight_decay: float = 1e-6 
    max_grad_norm: float = 2
    update_alg: str = "eg"
    custom_loss: bool = True
    mse_only_custom_loss: bool = False # if false loss is combined mse and ce
    copy_data_to_device_mem: bool = False # to put on gpua
    num_workers: int = 4
    pin_memory: bool = True
    tag : str = "mod_cog_debug3rdjune"
    use_wandb: bool = True
    save_checkpoints_per_epoch: bool = False
    save_dec_acc_ckpts: bool = False
    weight_distribution: str = "log_normal" 
    log_n_mean_std_ratio: Optional[float] = 1.5
    weight_init_gain: Optional[float] = 1.0 # 0.75 #float(1/np.sqrt(3))
    project_dir: str = os.path.join(os.environ['SCRATCH'], "eg_paper/mod_cog_debug")
    freeze_gd_signs:bool =False

cs = ConfigStore.instance()
cs.store(name="config", node=Config)

def check_and_name_config(cfg):
    exp_name = f"{cfg.update_alg}_N{cfg.model.hidden_size}_lr{cfg.lr}"
    exp_name += f"_m{cfg.momentum}_wd{cfg.weight_decay}_mxgn{cfg.max_grad_norm}"
    exp_name += f"_init_{cfg.weight_distribution}"
    if cfg.weight_distribution == "log_normal": 
        exp_name += str(cfg.log_n_mean_std_ratio)
    exp_name += f"_a{cfg.weight_init_gain}"
    if cfg.lr_sched: exp_name += "_lrsched"
    if cfg.custom_loss: 
        if cfg.mse_only_custom_loss: 
            exp_name += "_mse_only_custom_loss"
        else:
            exp_name += "_custom_loss"
    
    return exp_name

def get_wcc_lr_schedule(opt, cfg, lr_peak_iter, lr_cooldown_niters):
    """
    Warmup constant cooldown schedule
    https://arxiv.org/abs/2405.18392
    https://arxiv.org/abs/2404.06395v2

    Note cfg.steps_per_epoch must be correct for dataset and bs
    """
    epochs = cfg.n_epochs
    iters_per_epoch = cfg.steps_per_epoch
    if lr_cooldown_niters == 0:
        lr_schedule = np.interp(np.arange((epochs+1) * iters_per_epoch),
                        [0, lr_peak_iter, epochs * iters_per_epoch],
                        [0, 1, 1])
    else:
        lr_schedule = np.interp(np.arange((epochs+1) * iters_per_epoch),
                        [0, lr_peak_iter, (epochs*iters_per_epoch)-lr_cooldown_niters, epochs * iters_per_epoch],
                        [0, 1, 1, 0])   
    return lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
     
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
        out_tensor of shape (batch_size, output_size)
        """
        out_tensor = torch.zeros(x.size(0), x.size(1), self.output_size).to(x.device)
        h_t = torch.zeros(x.size(0), self.hidden_size).to(x.device)
        for t in range(x.size(1)):
            h_t = self.rnn(x[ :,t,: :], h_t)
            out_tensor[:,t, :] = self.fc(h_t)
        return out_tensor

@torch.no_grad()
def compute_acc_with_fixation_unit(outputs, labels, dec_mask):
    choices = torch.argmax(outputs, dim=-1)
    labels_for_full_acc = 1 + labels
    labels_for_full_acc[~dec_mask] = 0

    acc = torch.eq(choices, labels_for_full_acc).type(torch.float32).mean().item()
    dec_acc = torch.eq(choices[dec_mask], labels_for_full_acc[dec_mask]).type(torch.float32).mean().item()

    return acc, dec_acc

class ModCogLossCombined(torch.nn.Module):
    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.mse_loss = torch.nn.MSELoss()
        self.ce_loss = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def _compute_fixation_loss(self, outputs, dec_mask):
        bs, seq_len, out_dim = outputs.size()
        # only the fixation unit should be active
        target_mse = torch.concat([torch.ones(bs, seq_len, 1, device=outputs.device),
                                   torch.zeros(bs, seq_len, out_dim - 1, device=outputs.device)], dim=-1)
        return self.mse_loss(input=outputs[~dec_mask], target=target_mse[~dec_mask])

    def _compute_decoder_loss(self, outputs, labels, dec_mask):
        # first output is the fixation unit, which should not be active. Labels are shifted by one bc of it
        return self.ce_loss(input=outputs[dec_mask], target=1 + labels[dec_mask])

    def forward(self, outputs, labels, dec_mask):
        loss_fix = self._compute_fixation_loss(outputs, dec_mask)
        loss_dec = self._compute_decoder_loss(outputs, labels, dec_mask)
        return loss_fix + loss_dec, loss_fix.item(), loss_dec.item()


class ModCogLossMSE(ModCogLossCombined):
    def __init__(self, **ignored):
        super().__init__()

    def _compute_decoder_loss(self, outputs, labels, dec_mask):
        # first output is the fixation unit, which should not be active. Labels are shifted by one bc of it
        mse_labels = torch.nn.functional.one_hot(1 + labels[dec_mask], num_classes=outputs.shape[-1])
        return self.mse_loss(input=outputs[dec_mask], target=mse_labels.float())


def train_model(epoch_idx, model, dataset, opt, n_steps, lr_scheduler=None, use_wandb=True, custom_loss=False,
                max_grad_norm=None, save_dec_acc_ckpts=True, dec_acc_ckpt_thresholds=[], exp_dir=None,
                results_dict=None, cfg=None, log_parameter_stats=False): # log_parameter_stats=True 
    
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
        if log_parameter_stats:
            # first save a copy of params before update to calculate weight change proportionality etc
            initial_params = {p_name: p.detach().cpu().numpy() for p_name, p in model.named_parameters()}

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
        
        # Save a model checkpoint if the dec_acc is above a certain threshold
        if save_dec_acc_ckpts: 
            # note also save after first 50 and 100 epochs
            if exp_dir is None: raise ValueError("exp_dir must be provided if saving dec acc ckpts")
            if epoch_idx == 0 and update_idx == 99 or epoch_idx == 0 and update_idx == 49: 
                ckpt_path = exp_dir /"dec_acc_ckpts"/  f"model_0_{epoch_idx}-{update_idx}.pt"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt_path)
                print(f"saved model to {ckpt_path}")
            if len(dec_acc_ckpt_thresholds) > 0: 
                if dec_acc*100 >= dec_acc_ckpt_thresholds[0]:
                    acc_threshold = dec_acc_ckpt_thresholds.pop(0)
                    print(f"Dec Acc reached {acc_threshold} at epoch {epoch_idx} step {update_idx}")
                    ckpt_path = exp_dir /"dec_acc_ckpts"/  f"model_{acc_threshold}_{epoch_idx}-{update_idx}.pt"
                    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), ckpt_path)
                    print(f"saved model to {ckpt_path}")
            
        # Logging logic
        pbar.update(1)
        pbar.set_description(f"Training... lr:{lr:.2f} gn{grads_norm_before_clip:.2f}")
        pbar.set_postfix({"Loss": loss.item(), "mse_fix": loss_fix_no_grad,
                           "Acc": acc, "Dex Acc": dec_acc})
        
        if "Loss_auc" not in results_dict.keys(): loss_auc = 0
        else: loss_auc = results_dict["Loss_auc"][-1]
        
        batch_results_dict = {"Loss": loss.item(), 
                              "Loss_auc": loss.item()+loss_auc,
                              "Mse_fixation": loss_fix_no_grad,
                              "Acc": acc, 
                              "Dec Acc": dec_acc,
                              "lr": lr,
                              "grads_norm_before_clip":grads_norm_before_clip,
                              "grads_norm_after_clip":grads_norm_after_clip}
        if log_parameter_stats: # turn this off to avoid error raising, pass from main
            # note expensive to do this every step
            if save_dec_acc_ckpts:
                param_diffs = {p_name: (p.detach().cpu().numpy() - initial_params[p_name]) for p_name, p in model.named_parameters()}
                param_ols = {}
                for  p_name, _ in model.named_parameters():
                    if "weight" not in p_name: continue
                    if "base_layer" in p_name: p_name_logging = p_name.replace("base_layer.", "")
                    else: p_name_logging = p_name
                    # handle nan, turn off logging
                    try:
                        param_ols[p_name_logging]= au.fit_ols(np.abs(initial_params[p_name].ravel()),
                                                              np.abs(param_diffs[p_name].ravel()))
                    except Exception as e:
                        param_ols[p_name_logging] = {"const": np.nan, "slope":np.nan,
                                                    "pconst":np.nan, "pslope":np.nan,
                                                    "r2":np.nan}
                batch_results_dict.update(param_ols)

                sign_flips = {}
                for  p_name, p in model.named_parameters():
                    if "weight" not in p_name: continue
                    if "base_layer" in p_name: p_name_logging = p_name.replace("base_layer.", "")
                    else: p_name_logging = p_name
                    try:
                        sign_flips[p_name+"_sflips"] = (np.sign(initial_params[p_name]) != np.sign(p.detach().cpu().numpy())).sum()
                    except Exception as e:
                        sign_flips[p_name+"_sflips"] = np.nan
                batch_results_dict.update(sign_flips)

            with torch.no_grad():
                if hasattr(model.rnn, "base_layer"):
                    w_norm = torch.linalg.norm(model.rnn.base_layer.weight_hh.flatten()).item()
                else:
                    w_norm = torch.linalg.norm(model.rnn.weight_hh.flatten()).item()
            weight_metrics = {
                # "hh_mean": np.mean(w), 
                # "hh_std": np.std(w), 
                # "h_skew": stats.skew(w.flatten()),
                # "h_kurtosis": stats.kurtosis(w.flatten()),
                "h_norm": w_norm
                }

            # if hasattr(model.rnn, "base_layer"):
            #     w = model.rnn.base_layer.weight_hh.detach().cpu().numpy()
            # else:
            #     w = model.rnn.weight_hh.detach().cpu().numpy()
            # # if cfg.weight_distribution == "log_normal":
            # #     w = np.abs(w)
            # weight_metrics = {
            #     "hh_mean": np.mean(w),
            #     "hh_std": np.std(w),
            #     "hh_skew": stats.skew(w.flatten()),
            #     "hh_kurtosis": stats.kurtosis(w.flatten()),
            #     "hh_norm": np.linalg.norm(w)
            #     }

            batch_results_dict.update(weight_metrics)        
        results_dict = au.concat_dict(results_dict, batch_results_dict)
        if use_wandb: 
            wandb.log(batch_results_dict)
    return results_dict

def evaluate_model(epoch_idx, model, loaders, use_wandb=True, custom_loss=False, cfg=None):

    """This is only used when using cached data."""
    results_dict = {}
    device = eg_utils.get_device()
    model.eval()

    if cfg.mse_only_custom_loss:
        full_loss = ModCogLossMSE()
    else:
        full_loss = ModCogLossCombined(label_smoothing=0.1)

    ce_criterion = torch.nn.CrossEntropyLoss()

    for split in ["val", "test"]:
        loader = loaders[split]
        accs = []
        dec_accs = []
        losses = []
        for inputs, labels in loader:
            inputs = inputs.to(device)
            task_rules = inputs[:,:,-1]
            one_hot_rule_inputs = torch.nn.functional.one_hot(task_rules.long(), num_classes=82).float()
            inputs = torch.cat([inputs[:,:,:-1], one_hot_rule_inputs], dim=-1)
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            outputs = model(inputs)
            dec_mask = inputs[:, :, 0] == 0

            acc, dec_acc = compute_acc_with_fixation_unit(outputs, labels, dec_mask)

            if custom_loss:
                loss = full_loss(outputs, labels, dec_mask)[0]  # [0] ignores individial .item()-ized losses
            else:
                loss = ce_criterion(input=outputs.permute(0,2,1), target=labels)
            accs.append(acc)
            dec_accs.append(dec_acc)
            losses.append(loss.item())
        mean_acc = np.mean(accs)
        mean_dec_acc = np.mean(dec_accs)
        mean_loss = np.mean(losses)
        print(f"{epoch_idx}:{split} Acc: {mean_acc:.2f}, Dec Acc: {mean_dec_acc:.2f}, Loss: {mean_loss:.2f}")
        results_dict[f"{split}_Acc"] = mean_acc
        results_dict[f"{split}_Dec_Acc"] = mean_dec_acc
        results_dict[f"{split}_Loss"] = mean_loss
    if use_wandb:
        wandb.log(results_dict)
    return results_dict

@hydra.main(config_name="config", version_base=None)
def main(cfg):
    start = time.time()
    device = eg_utils.get_device()

    print("\nRunning with following Config:")
    print(" --------------------------- ")
    print(OmegaConf.to_yaml(cfg, resolve=True, sort_keys =True), end="")
    print("--------------------------- ")
    
    # Seed logic (use SLURM_PROCID to set seed if multiple tasks are running per gpu node)
    # note we are maxing out the gpu usage though
    if "SLURM_PROCID" in os.environ.keys():
        seed = int(os.environ["SLURM_PROCID"]) + cfg.seed
        print(f"SLURM_PROCID found, setting seed to {os.environ['SLURM_PROCID']} + {cfg.seed}")
        cfg.seed = seed
    print("Setting seed", cfg.seed)
    eg_utils.set_seed_all(cfg.seed)

    exp_name = check_and_name_config(cfg)
    
    # Data logic
    use_cached_data = True # note hardcoded 82 dim tasks for now, 
    if not use_cached_data:
        print("Constructing mod cog neurogym object...")
        dataset = mct.construct_mod_cog_dataset(cfg.batch_size, cfg.seq_len)
        input_size = dataset.env.observation_space.shape[0]
        output_size = dataset.env.action_space.n
        print(f"Input size: {input_size}, Output size: {output_size}")
    else:
        try:
            data_dir = Path(os.environ["SLURM_TMPDIR"])/"data"
            assert data_dir.exists()
            print(f"Using data from {data_dir}")
        except:
            print("Using data from scratch directory")
            data_dir = Path(os.path.join(os.environ['SCRATCH'], "eg_paper/mod_cog_data_v2"))

        print("Using cached data")
        loaders = {}
        for split in ["train", "val", "test"]:
            dataset = mcdu.ModCogDataset(data_dir, split=split, in_memory=cfg.copy_data_to_device_mem)
            loaders[split] = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=split=="train",
                                        num_workers = 0 if cfg.copy_data_to_device_mem else cfg.num_workers, 
                                        pin_memory=cfg.pin_memory)

        assert cfg.steps_per_epoch == len(loaders["train"])
        input_size = dataset.input_size -1 + 82 # -1 for labels, 82 is the default (hardcoded) number of tasks in the mod cog dataset
        output_size = dataset.output_size

    print(f"Data loading took {time.time()-start:.2f} seconds")


    # Model logic
    model = RNNModel(input_size, cfg.model.hidden_size, output_size).to(device)
    if cfg.update_alg == "eg":
        eg_utils.set_split_bias(model)

    initializations.re_init_network(model, all_modules=True, init=cfg.weight_distribution,
                                    gain=cfg.weight_init_gain, mean_std_ratio=cfg.log_n_mean_std_ratio)

    # Optimizer logic
    opt = eg_utils.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                       weight_decay=cfg.weight_decay, update_alg=cfg.update_alg,
                       freeze_gd_signs=cfg.freeze_gd_signs)

    if cfg.lr_sched:
        scheduler = get_wcc_lr_schedule(opt,cfg, lr_peak_iter = 500, lr_cooldown_niters=1000)
    else:
        scheduler = None
        
    # Logging logic 
    dec_acc_ckpt_thresholds = [25, 35, 45, 55, 65, 75, 85, 95] # this var gets get used in the train_model function if cfg.save_dec_acc_ckpts is True
    
    project_dir = Path(cfg.project_dir)
    exp_dir = project_dir / exp_name / f"seed_{cfg.seed}"
    os.makedirs(exp_dir, exist_ok=True)
    overwrite = True # Set False to prevent re-running previous experiments 
    # e.g change this if adding runs to a sweep shell file.
    if not overwrite:
        if (exp_dir/"train_results.npy").exists():
            print("Overwrite is False and training results already exists")
            raise ValueError(f"Training results already exists: {exp_dir}")
    OmegaConf.save(cfg,exp_dir/"config.yaml")

    if cfg.use_wandb:
        os.environ['WANDB_DIR'] = str(Path.home()/ "scratch/")
        wandb_project  = "mod_cog_debug"
        run = wandb.init(reinit=False, 
                        name=exp_name,
                        project=project_dir.name,
                        tags = [cfg.tag],
                        config=OmegaConf.to_container(cfg, resolve=True))
        
    train_results_dict = {}
    eval_results_dict = {} # only used if using cached data
    # Training loop logic
    for epoch_idx in range(cfg.n_epochs):
        if cfg.save_checkpoints_per_epoch:
            ckpt_path = exp_dir /"epoch_ckpts"/  f"model_{epoch_idx}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"saved model to {ckpt_path}")
        
        # Evaluate model (if cached data, else online)
        run_eval = True # set false to speed up debugging
        if run_eval:
            if use_cached_data:
                eval_dict = evaluate_model(epoch_idx, model, loaders,  use_wandb=cfg.use_wandb,
                                        custom_loss=cfg.custom_loss, cfg=cfg)
                eval_results_dict = au.concat_dict(eval_results_dict, eval_dict)
        
        # Train model 
        if use_cached_data:
            train_results_dict = train_model(epoch_idx, model, loaders["train"], opt, n_steps=cfg.steps_per_epoch, 
                            max_grad_norm=cfg.max_grad_norm,
                            lr_scheduler=scheduler, use_wandb=cfg.use_wandb, 
                            custom_loss=cfg.custom_loss, save_dec_acc_ckpts=cfg.save_dec_acc_ckpts,
                            dec_acc_ckpt_thresholds=dec_acc_ckpt_thresholds, exp_dir=exp_dir,
                            results_dict=train_results_dict, cfg=cfg)
        else:
            train_results_dict = train_model(epoch_idx, model, dataset, opt, n_steps=cfg.steps_per_epoch, 
                            max_grad_norm=cfg.max_grad_norm,
                            lr_scheduler=scheduler, use_wandb=cfg.use_wandb, 
                            custom_loss=cfg.custom_loss, save_dec_acc_ckpts=cfg.save_dec_acc_ckpts,
                            dec_acc_ckpt_thresholds=dec_acc_ckpt_thresholds, exp_dir=exp_dir,
                            results_dict=train_results_dict, cfg=cfg)

    #########################
    # final evaluation
    epoch_idx = cfg.n_epochs
    save_final = True
    if cfg.save_checkpoints_per_epoch or save_final:
        ckpt_path = exp_dir / "epoch_ckpts" / f"model_{epoch_idx}.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        print(f"saved model to {ckpt_path}")

    # Evaluate model (if cached data, else online)
    run_eval = True  # set false to speed up debugging
    if run_eval:
        if use_cached_data:
            eval_dict = evaluate_model(epoch_idx, model, loaders, use_wandb=cfg.use_wandb,
                                       custom_loss=cfg.custom_loss, cfg=cfg)
            eval_results_dict = au.concat_dict(eval_results_dict, eval_dict)
    ##########################

    if cfg.use_wandb:
        wandb.finish()
    print(f"Total time: {time.time()-start:.2f} seconds")
    np.save(exp_dir/"train_results.npy", train_results_dict)
    np.save(exp_dir/"eval_results.npy", eval_results_dict)

        
if __name__ == "__main__":
    main()
