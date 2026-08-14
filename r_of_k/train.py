from tqdm import tqdm
from typing import Mapping

import numpy as np
from omegaconf import II, MISSING, OmegaConf, DictConfig
import hydra
from hydra.core.config_store import ConfigStore
import wandb

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import r_of_k_data as r_of_k

import sys
sys.path.append('../')
from src import eg_utils
# import cloudpickle

import train_utils
# cloudpickle.register_pickle_by_value(train_utils)


def build_dataloaders(cfg: Mapping, verbose=True):
    print("-----")
    X,y, w_star = r_of_k.generate_noisy_r_of_k_data(
                                            n_datapoints=cfg.dataset.n_train_datapoints,
                                            n=cfg.dataset.n, 
                                            k=cfg.dataset.k, 
                                            r=cfg.dataset.r,
                                            p_rel=cfg.dataset.p,
                                            neg_wstar=False,
                                            verbose=verbose,
                                            noise_probs=(0.05, 0.9, 0.05))
    print("-----")
    X_val, y_val, _ = r_of_k.generate_noisy_r_of_k_data(
                                            n_datapoints=cfg.dataset.n_val_datapoints,
                                            n=cfg.dataset.n,
                                            k=cfg.dataset.k,
                                            r=cfg.dataset.r,
                                            p_rel=cfg.dataset.p,
                                            w_star=w_star,
                                            verbose=verbose,
                                            noise_probs=None)
    print("-----")
    X_test, y_test, _ = r_of_k.generate_noisy_r_of_k_data(
                                            n_datapoints=cfg.dataset.n_test_datapoints,
                                            n=cfg.dataset.n,
                                            k=cfg.dataset.k,
                                            r=cfg.dataset.r,
                                            p_rel=cfg.dataset.p,
                                            w_star=w_star,
                                            verbose=verbose,
                                            noise_probs=None)
    
    loaders = {}
    for split in ['train', 'val', 'test']:
        if split == 'train':
            rofk_dataset = train_utils.Dataset(X, y) 
            
        elif split == 'val':
            rofk_dataset = train_utils.Dataset(X_val, y_val) 
            
        elif split == 'test':
            rofk_dataset = train_utils.Dataset(X_test, y_test) 

        loaders[split] = DataLoader(rofk_dataset,
                                    batch_size=cfg.dataset.batch_size,
                                    shuffle=True if 'train' in split else False)
    return loaders, w_star


class RofKLinear(nn.Module):
    def __init__(self, in_feats, out_feats, bias, n, r, p):
        super(RofKLinear, self).__init__()
        self.linear = nn.Linear(in_features=in_feats, out_features=out_feats, bias=bias)
        self.r = r
        # init such that the expected activation is r. Note this assumes every
        # input dim is drawn Bernoulli(p), i.e. the data was generated with
        # p_irrel == p_rel == p. If p_irrel were ever made independent, the
        # denominator would need the mixture mean, k*p_rel + (n-k)*p_irrel.
        self.linear.weight.data = torch.ones_like(self.linear.weight.data)*(r/(p*n))

    def forward(self, x):
        # r is subtracted to init logits at 0 given the init
        return self.linear(x) - self.r

def build_model(cfg):
    """
    For r-of-k we use a single-neuron linear layer with no bias and identity
    act func (the binary CE loss takes logit inputs). Weights are init to
    r/(p*n) so the expected pre-activation is r (see RofKLinear).
    """
    model = RofKLinear(in_feats=cfg.dataset.n, out_feats=1, bias=False, 
                       n=cfg.dataset.n, r=cfg.dataset.r, p=cfg.dataset.p)
    return model

def get_optimiser(model: torch.nn.Module, cfg: Mapping, verbose=False):
    """
    Returns eg_utils.SGD over model.parameters(), configured from cfg.opt
    (lr, weight decay, momentum, update algorithm).
    """
    params = model.parameters()
    opt = eg_utils.SGD(params, lr=cfg.opt.lr,
                       weight_decay=cfg.opt.wd,
                       momentum=cfg.opt.momentum,
                       update_alg=cfg.opt.update_algorithm)
    return opt

def train_epoch(cfg, opt, model, loaders, epoch_i, scaler, use_wandb, use_tqdm=False): 
    """
    Returns batch_accs, batch_losses lists
    """
    
    loss_func = torch.nn.functional.binary_cross_entropy_with_logits
    
    batch_accs, batch_losses = [], []
    if use_tqdm: progress_bar = tqdm(loaders['train'], desc='Train')
    else: progress_bar = loaders["train"]
    for batch_i, (X, y) in enumerate(progress_bar):
        X = X.to(device)
        y = y.to(device)

        model.train()
        opt.zero_grad(set_to_none=True)
        with autocast():
            logits = model(X) 
            loss = loss_func(logits.squeeze(), y.float())
        
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            y_hat = (probs>=0.5).long() # labels are long (int 32)
            batch_acc = train_utils.binary_acc(y_hat, y)
        
        batch_accs.append(batch_acc.item())
        batch_losses.append(loss.item())

        if use_tqdm:
            progress_bar.set_description(
            f'Epoch {epoch_i}, batch {batch_i+1}: Acc {np.mean(batch_accs):.2f}% '
            )

        if use_wandb:
            wandb.log({"batch_acc" : batch_acc.item(),
                       "batch_loss" : loss.item()})

    # print("Accs:", batch_accs)
    # print("Losses:", batch_losses)
    return batch_losses, batch_accs

def eval_model(cfg, results_dict, model, loaders, epoch_i, use_wandb, print_output=True):
    """
    """
    loss_func = torch.nn.functional.binary_cross_entropy_with_logits
    output_str = f"Epoch {epoch_i}: "
    model.eval()
    #with autocast:
    with torch.no_grad():
        for key in ["train", "val", "test"]:
            loss, acc, n = 0,0,0
            average_logits = 0
            average_yhat = 0
            for X, y in loaders[key]:
                X = X.to(device)
                y = y.to(device)
                logits = model(X)
                probs = torch.sigmoid(logits)
                y_hat = (probs>=0.5).long() # labels are long (int 32)
                loss += loss_func(logits.squeeze(), y.float(), reduction="sum").item()
                acc += train_utils.binary_acc(y_hat, y, reduction="sum").item()
                n += y.size(0)
                average_logits += logits.sum().item()
                average_yhat += y_hat.sum().item()
            results_dict[key+"_losses"].append(loss/n)
            results_dict[key+"_accs"].append(acc/n)
            output_str += f"{key} acc {acc/n:.2f} loss {loss/n:.2f}, "
            if key == "train":
                output_str += f"av. logits: {average_logits/n:.2f} "
                output_str += f"av. y_hat: {average_yhat/n:.2f} "
            
            if use_wandb:
                wandb.log({key+"_loss" : loss/n,
                           key+"_acc" : acc/n,
                           "epoch_i": epoch_i})
        if print_output:
            print(output_str)

    results_dict["epoch_idxs"].append(epoch_i)


@hydra.main(config_path = "conf", config_name = "main_config", version_base=None)
def main(cfg, verbose = False, overwrite=True):
    print("\nRunning with following Config:\n")
    print(" --------------------------- ")
    print(OmegaConf.to_yaml(cfg, resolve=True, sort_keys =True))
    print(" --------------------------- ")

    if cfg.exp.use_wandb:
        run = train_utils.init_wandb(cfg)
        print("Logging with wandb, run.name:", run.name)

    # Do not re-run existing sweep. 
    if train_utils.config_exists(cfg, report_exists=True):
        if not overwrite: return None
    else:
        train_utils.save_config(cfg)

    eg_utils.set_seed_all(cfg.seed)

    model = build_model(cfg)
    model.to(device)
        
    # save model
    opt = get_optimiser(model, cfg)

    loaders, w_star = build_dataloaders(cfg)
    # with torch.no_grad():
    #     model.weight.data = torch.from_numpy(w_star.copy()).float().to(device)
    scaler = GradScaler() 

    if verbose:
        print(model)
        print("--------")
        print(opt)
        print("--------")
        print(loaders)
        print("--------")

    results_dict = {
        'train_losses': [], 'train_accs': [],
        'val_losses': [], 'val_accs': [],
        'test_losses': [], 'test_accs': [],
        'epoch_idxs':[]
    }
    for epoch_i in range(cfg.n_epochs):
        eval_model(cfg, results_dict=results_dict, model=model, 
                   loaders=loaders, epoch_i=epoch_i, 
                   use_wandb=cfg.exp.use_wandb)
        train_epoch(cfg, opt=opt, model=model, loaders=loaders, 
                    epoch_i=epoch_i, scaler=scaler, 
                    use_wandb=cfg.exp.use_wandb)
        # possibly save model here
    
    train_utils.save_results_dict(cfg, results_dict)
    
if __name__ == "__main__":
    device = eg_utils.get_device()
    main()
