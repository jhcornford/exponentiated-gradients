from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import math 
import wandb

from datetime import datetime
from motornet.effector import RigidTendonArm26
from motornet.muscle import MujocoHillMuscle

import train_utils
from random_reach_env import CustomRandomTargetReach

import hydra
from omegaconf import OmegaConf

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from reach_plot_utils import eval_and_plot, plot_losses

import sys
sys.path.append('../')
from src import eg_utils
from src import initializations
#from src import adamw_eg
from src import analysis_utils as au

def get_wcc_lr_schedule(opt, cfg, lr_peak_iter, lr_cooldown_niters):
    """
    Warmup constant cooldown schedule
    https://arxiv.org/abs/2405.18392
    https://arxiv.org/abs/2404.06395v2

    Note cfg.steps_per_epoch must be correct for dataset and bs
    Refactor to use cfg.n_iters instead of cfg.n_epochs * cfg.steps_per_epoch
    """
    epochs = cfg.n_epochs
    iters_per_epoch = cfg.steps_per_epoch
    # Calculate the total number of iterations
    total_iters = (epochs + 1) * iters_per_epoch
    if lr_cooldown_niters == 0:
        # Linear interpolation for learning rate schedule without cooldown
        lr_schedule = np.interp(np.arange(total_iters),
                        [0, lr_peak_iter, epochs * iters_per_epoch],
                        [0, 1, 1])
    else:
        lr_schedule = np.interp(np.arange(total_iters),
                        [0, lr_peak_iter, (epochs*iters_per_epoch)-lr_cooldown_niters, epochs * iters_per_epoch],
                        [0, 1, 1, 0])   
    return lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)

def get_cosine_lr_schedule(opt, cfg, warmup_iters=0, flat_iters=0):
    """
    Returns a LambdaLR scheduler that implements cosine learning rate decay with optional warm-up and flat period.

    Args:
        opt: The optimizer instance (e.g., torch.optim.SGD, torch.optim.Adam).
        total_iters: Total number of iterations for training (e.g., epochs * iterations per epoch).
        warmup_iters: Number of iterations to linearly increase the learning rate (warm-up phase).
        flat_iters: Number of iterations to keep the learning rate constant after warm-up.

    Returns:
        A LambdaLR scheduler with the specified learning rate schedule.
    """
    epochs = cfg.n_epochs
    iters_per_epoch = cfg.steps_per_epoch

    # Calculate the total number of iterations
    total_iters = (epochs + 1) * iters_per_epoch
    def lr_lambda(current_iter):
        if current_iter < warmup_iters:
            # Warm-up phase: linear increase from 0 to 1
            return float(current_iter) / float(max(1, warmup_iters))
        elif current_iter < warmup_iters + flat_iters:
            # Flat phase: keep learning rate at 1
            return 1.0
        else:
            # Cosine decay phase
            decay_iter = current_iter - warmup_iters - flat_iters
            decay_total = max(1, total_iters - warmup_iters - flat_iters)
            return 0.5 * (1 + math.cos(math.pi * decay_iter / decay_total))
    
    return lr_scheduler.LambdaLR(opt, lr_lambda)

def get_optimizer(cfg, params):
        # Optimizer logic
    if cfg.opt.optimiser == "adamw":
        if cfg.opt.update_algorithm == "eg":
            optimizer = adamw_eg.AdamWEG(params, 
                                         lr=cfg.opt.lr, 
                                         betas=(0.9, 0.999), 
                                         eps=1e-8, 
                                         weight_decay=cfg.opt.wd)
        else:
            optimizer = torch.optim.AdamW(params, 
                                          lr=cfg.opt.lr, 
                                          betas=(0.9, 0.999), 
                                          eps=1e-8,
                                          weight_decay=cfg.opt.wd)
    elif cfg.opt.optimiser == "sgd":
        optimizer = eg_utils.SGD(params,
                                 lr=cfg.opt.lr,
                                 momentum=cfg.opt.momentum,
                                 weight_decay=cfg.opt.wd,
                                 update_alg=cfg.opt.update_algorithm)
    return optimizer

class RNNModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layernorm=False):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn = torch.nn.RNNCell(input_size, hidden_size, nonlinearity='relu')
        self.fc  = torch.nn.Linear(hidden_size, output_size)
        self.hh_nonlinearity = nn.ReLU()  # redundant with RNNCell's relu when layernorm off; with LN gives relu(LN(relu(x)))
        self.fc_nonlinearity = nn.Sigmoid()
        self.layernorm = nn.LayerNorm(hidden_size) if layernorm else None

    def forward(self, x, h):
        """
        x of shape (batch_size, seq_len, input_size)
        out_tensor of shape (batch_size, output_size)
        """
        x = x.squeeze(1)
        h_t = self.rnn(x, h)
        if self.layernorm is not None:
            h_t = self.layernorm(h_t)
        h_t = self.hh_nonlinearity(h_t)
        out_tensor = self.fc(h_t)
        return self.fc_nonlinearity(out_tensor), h_t

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size, device=device)

def build_controller(cfg):
    obs_noise    = cfg.exp.obs_noise if "obs_noise" in cfg.exp.keys() else 0.0
    action_noise = cfg.exp.action_noise if "action_noise" in cfg.exp.keys() else 0.0  # not currently forwarded to env

    env = CustomRandomTargetReach(
            effector=RigidTendonArm26(muscle=MujocoHillMuscle()).to(device),
            proprioception_delay=0.02,
            vision_delay=0.05,
            action_frame_stacking=0, # pass the last action to the next timestep as input
            obs_noise=obs_noise,
            device=device,
            irrel_noise_dims=cfg.exp.n_irrel_feats,
            irrel_noise_type=cfg.exp.irrel_noise_type,
            irrel_noise_theta=cfg.exp.irrel_noise_theta,
        )

    policy = RNNModel(input_size=env.observation_space.shape[0],
                      hidden_size=cfg.model.hidden_size,
                      output_size=env.n_muscles,
                      layernorm=cfg.model.layernorm)
    policy.to(device)

    controller = Controller(env=env, policy=policy)
    controller.to(device)
    return controller

class Controller(nn.Module):
    def __init__(self, policy, env):
        super().__init__()
        self.env = env
        self.policy = policy

    def forward(self, x, states):
        """
        x is observations (batch_size, 1, -1)
        states is last hidden (batch_size, hidden_size)
        """
        u, h = self.policy(x, states)
        obs, reward, _, _, info = self.env.step(u)
        output = {
            "actions": u,
            "policy_states": h,
            "observations": obs,
            "reward_states": info,
            "reward": reward,
            }
        output.update({"env_" + key: val for key, val in info["states"].items()})
        return output

def forward_pass(controller, batch_size, n_timesteps, 
                 eval_mode: bool = False, use_perturbations: bool = False):
    h = controller.policy.init_hidden(batch_size=batch_size) # (batch, hidden)
    obs, info = controller.env.reset(options=dict(batch_size=batch_size, eval_mode=eval_mode))
    outputs = {
        "actions": [],
        "policy_states": [h], 
        "observations": [obs],
        "reward_states": [],
        "reward": [],
        }
    outputs.update({"env_" + key: [val] for key, val in info["states"].items()})
    outputs.update({"env_" + key: [val] for key, val in info.items() if key != "states"})
    
    if use_perturbations: 
        raise NotImplementedError
        # joint loads will be of shape [1,2]? 
    else:
        joint_load = None 
        endpoint_load = None

    for t in range(n_timesteps):
        h = outputs["policy_states"][-1]
        obs = outputs["observations"][-1].reshape(batch_size, 1, -1)        
        output = controller(obs, h) #endpoint_load=endpoint_load, joint_load=joint_load)
        for key, val in output.items():
            outputs[key].append(val)
    
    return outputs

@hydra.main(config_path = "conf", config_name = "reach", version_base=None)
def main(cfg):
    for k in os.environ.keys():
        if k.startswith("SLURM"): print(k, os.environ[k], flush=True)

    print("\nRunning with following Config:\n")
    print(" --------------------------- ")
    print(OmegaConf.to_yaml(cfg, resolve=True, sort_keys =True))
    print(" --------------------------- ")

    # Seed logic (use SLURM_PROCID to set seed if multiple tasks are running per gpu node)
    if "SLURM_PROCID" in os.environ.keys():
        seed = int(os.environ.get("SLURM_PROCID", 0)) + cfg.seed
        print(f"SLURM_PROCID found, setting seed to {os.environ['SLURM_PROCID']} + {cfg.seed}")
        cfg.seed = seed
    print("Setting seed", cfg.seed)
    cfg.seed = int(cfg.seed)
    train_utils.set_seed_all(cfg.seed)

    # 1. Set up output dir
    if cfg.logging.log_local:
        exp_output_dir = train_utils.format_exp_output_dir(cfg)
        train_utils.save_config(cfg)
        
    # 2. Build controller and optimizer etc
    controller = build_controller(cfg)

    if cfg.opt.update_algorithm == "eg":
        eg_utils.set_split_bias(controller.policy)

    initializations.re_init_network(controller.policy, 
                                    all_modules=True, 
                                    init=cfg.model.weight_distribution,
                                    gain=cfg.model.weight_init_gain, 
                                    mean_std_ratio=cfg.model.log_n_mean_std_ratio)

    optimizer = get_optimizer(cfg, params=controller.policy.parameters())

    if cfg.opt.lr_scheduler:
        lr_scheduler = get_wcc_lr_schedule(optimizer, cfg, lr_peak_iter = 100, lr_cooldown_niters=500)
    else: # refactor, false is no lr scheduler
        lr_scheduler = get_cosine_lr_schedule(optimizer, cfg, warmup_iters=100, flat_iters=0)

    if cfg.logging.use_wandb:
        wandb_logger = train_utils.init_wandb(cfg)
    
    if cfg.logging.log_local:
        torch.save(controller.state_dict(), os.path.join(exp_output_dir, "init_model.pth"))
    
    # 3. Train loop 
    losses, l1_losses, rel_l1_losses = [], [], []
    weight_norms, gradients = [], []
    results_dict = {}
    mean_iter_time = [0, 0, 0, 0] # profile the loop with a list of timepoints.
    info_interval = 50

    for iter_idx in range(0, cfg.n_epochs):
        log_this_iter = (iter_idx % info_interval == 0) or (iter_idx == cfg.n_epochs - 1)

        if log_this_iter:
            start_time = datetime.now()

        optimizer.zero_grad()
        outputs = forward_pass(controller, cfg.batch_size, n_timesteps=100, 
                                use_perturbations=cfg.exp.use_perturbations)

        if log_this_iter:
            time_elapsed = (datetime.now() - start_time).total_seconds()
            mean_iter_time[1] += (time_elapsed - mean_iter_time[1]) / (iter_idx + 1)
        
        # todo - what are the dimensions of these, why are we stacking them?
        y = torch.stack(outputs["env_fingertip"][1:])
        actions = torch.stack(outputs["actions"])[:, :, :]
        label = torch.stack(outputs["env_goal"])
        
        # is _reg a bad var naming choie? Yes
        action_reg = torch.mean(actions.pow(2.))
        hidden_reg = torch.mean(torch.stack(outputs["policy_states"]).pow(2.))

        action_reg_loss = cfg.opt.action_reg_lambda * action_reg
        hidden_reg_loss = cfg.opt.hidden_reg_lambda * hidden_reg
        total_loss = action_reg_loss + hidden_reg_loss

        if cfg.exp.time_weighted_err:
            err = y-label # (seq_len, batch_size, 2 ie. (x,y))
            seq_len = err.size(0)
            err_mask = torch.arange(1, seq_len+1, device=device)*2/(seq_len+1)
            time_weighted_err = err * err_mask.view(-1, 1, 1)
            loss_unreduced = torch.abs(time_weighted_err).sum(dim=-1) 
        else:
            loss_unreduced =  torch.abs(y-label).sum(dim=-1) # this is just l1 loss not reduced to mean
        
        # calculate l1 and relative l1 loss_unreduced for logging
        with torch.no_grad():
            l1_loss_unreduced = torch.abs(y-label).sum(dim=-1) 
            l1_loss = l1_loss_unreduced.mean()

        start_position = y[0, :, :]    # (batch_size, xy)
        end_position = label[-1, :, :] # (batch_size, xy)
        reach_dist  = torch.norm(end_position - start_position, dim=-1, p=1, keepdim=True) 
        rel_l1_loss = torch.mean(l1_loss_unreduced / reach_dist.view(1, -1)) 
        if cfg.exp.relative_distance_loss:
            total_loss += torch.mean(loss_unreduced/reach_dist.view(1, -1)) * reach_dist.mean() # normalise by reach distance to rescale to be similar to l1 loss
        else:
            total_loss += torch.mean(loss_unreduced) 

        if log_this_iter:
            time_elapsed = (datetime.now() - start_time).total_seconds()
            mean_iter_time[2] += (time_elapsed - mean_iter_time[2]) / (iter_idx + 1)

        total_loss.backward()


        # Gradient clipping
        max_grad_norm=cfg.opt.max_grad_norm
        with torch.no_grad():
            grads_norm_before_clip = 0
            for p in controller.policy.parameters():
                if p.grad is not None:
                    grads_norm_before_clip += p.grad.flatten().norm().item() ** 2

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(controller.policy.parameters(), max_grad_norm, norm_type=2)
                grads_norm_after_clip = 0
                for p in controller.policy.parameters():
                    if p.grad is not None:
                        grads_norm_after_clip += p.grad.flatten().norm().item() ** 2
            else:
                grads_norm_after_clip = grads_norm_before_clip
            
            grads_norm_before_clip = np.sqrt(grads_norm_before_clip)
            grads_norm_after_clip = np.sqrt(grads_norm_after_clip)


        # Update params and get new learning rate
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
            lr = lr_scheduler.get_last_lr()[0]
        else:
            lr = optimizer.param_groups[0]['lr']
 
        # Logging logic 
        if log_this_iter:
            time_elapsed = (datetime.now() - start_time).total_seconds()
            mean_iter_time[3] += (time_elapsed - mean_iter_time[3]) / (iter_idx + 1)
        
        with torch.no_grad():
            if hasattr(controller.policy.rnn, "base_layer"):
                w_norm = torch.linalg.norm(controller.policy.rnn.base_layer.weight_hh.flatten()).item()
                sigma_max = eg_utils.max_singular_value(controller.policy.rnn.base_layer.weight_hh).item()
            else:
                w_norm = torch.linalg.norm(controller.policy.rnn.weight_hh.flatten()).item()
                sigma_max = eg_utils.max_singular_value(controller.policy.rnn.weight_hh).item()

        if log_this_iter:
            time_elapsed = (datetime.now() - start_time).total_seconds()
            mean_iter_time[0] += (time_elapsed - mean_iter_time[0]) / (iter_idx + 1)
            print(f"{time_elapsed:.2f}s ({' '.join([f'{mu:.2f}' for mu in mean_iter_time])})")
        
        # note these are effectively being logged x2, as also in results_dict
        # refactor the log_ths_iter if change
        weight_norms.append(w_norm)
        losses.append(total_loss.item())
        l1_losses.append(l1_loss.item())
        rel_l1_losses.append(rel_l1_loss.item())

        batch_results_dict = {"loss": total_loss.item(),
                              "rel_l1_loss":rel_l1_loss.item(),
                              "l1_loss":l1_loss.item(),
                              "lr": lr,
                              "grads_norm_before_clip":grads_norm_before_clip,
                              "grads_norm_after_clip":grads_norm_after_clip,
                              "hidden_reg":hidden_reg.item(),
                              "action_reg":action_reg.item(),
                              "Whh_norm":w_norm,
                              "loss_auc":np.sum(losses),
                              "sigma_max":sigma_max,}
        if cfg.logging.log_local:
            results_dict = au.concat_dict(results_dict, batch_results_dict)

        if cfg.logging.use_wandb:
            wandb_logger.log(batch_results_dict)

        if log_this_iter:
            av_loss = sum(losses[-info_interval:])/info_interval
            av_l1_loss = sum(l1_losses[-info_interval:])/info_interval
            av_rel_l1_loss = sum(rel_l1_losses[-info_interval:])/info_interval
            print(f"{iter_idx}/{cfg.n_epochs}, Loss Policy: {av_loss:.3f}, L1: {av_l1_loss:.3f}, rL1: {av_rel_l1_loss:.3f} ", end="")
            # End of run
        
        if (iter_idx % 1000 == 0) and cfg.logging.save_checkpoints_every_1k:
            if cfg.logging.log_local:
                torch.save(controller.state_dict(), os.path.join(exp_output_dir, f"controller_{iter_idx}.pth"))

        if ((iter_idx % 500 == 0) or (iter_idx == cfg.n_epochs - 1)) and cfg.logging.log_plots:
            print("logging plot")
            fig = eval_and_plot(controller)
            if cfg.logging.use_wandb:
                wandb_logger.log({"eval_chart": wandb.Image(fig)})
            if cfg.logging.log_local:
                fig.savefig(f"{exp_output_dir}/eval_{iter_idx}.png")
                fig.savefig(f"{exp_output_dir}/eval_{iter_idx}.pdf")
            plt.close(fig)
    
    # End of run
    if cfg.logging.log_local:
        torch.save(controller.state_dict(), os.path.join(exp_output_dir, f"controller_{iter_idx}.pth"))
        fig = eval_and_plot(controller)
        fig.savefig(f"{exp_output_dir}/eval_{iter_idx}.png")
        fig.savefig(f"{exp_output_dir}/eval_{iter_idx}.pdf")
        np.save(exp_output_dir/"results.npy", results_dict)
        plt.close(fig)
    
    if cfg.logging.use_wandb:
        fig = eval_and_plot(controller)
        wandb_logger.log({"eval_chart": wandb.Image(fig)})
        plt.close(fig)
   

if __name__ == "__main__":
    main()
