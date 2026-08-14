from pathlib import Path
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import seaborn as sns
import pickle

import sys
sys.path.append('../../')
sys.path.append('../')
# from src import mod_cog_tasks as mct 
from src import mod_cog_data_utils as mcdu
from src import eg_utils
from src import analysis_utils as au
from src import plot_utils as pu
from src import prune 

import networkx as nx
from copy import deepcopy

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

def get_rnn_module(model):
    try:
        return model.rnn.base_layer
    except:
        return model.rnn

def load_rnn(run):
    device = eg_utils.get_device()
    model = load_model(run)
    model = model.to(device)
    rnn = get_rnn_module(model)
    return rnn

def prune_wrec(rnn_module, prune_prop = 0.9, inverse_magnitude=False):
    prune.global_unstructured([(rnn_module, "weight_hh")], 
                          pruning_method=prune.L1Unstructured,
                          amount=prune_prop,inverse_magnitude=inverse_magnitude)
    prune.remove(rnn_module, "weight_hh")


def clean_weights(w, abs_weights=True):
    # S1: Remove self connections, abs(W)
    w_no_self = w.copy()
    if abs_weights: 
        w_no_self = np.abs(w_no_self)
    np.fill_diagonal(w_no_self, 0.0)
    
    # S2: Create directed graph
    G = nx.from_numpy_array(w_no_self, create_using=nx.DiGraph)

    return G 

def get_weight_hh(module, detach=True):
    try: # module is the model
        try:
            weight_hh = module.rnn.weight_hh
        except:
            weight_hh = module.rnn.base_layer.weight_hh
    except: # module is the rnn
        try:
            weight_hh = module.weight_hh
        except:
            weight_hh = module.base_layer.weight_hh
        
    if detach:
        return weight_hh.detach().cpu().numpy()
    else: 
        return weight_hh

def shortest_path_length(G, weight=None):
    """
    G can be either undirected or directed
    """
    if weight is not None:
        raise # delete when you've implemented an inverse function.
    if not nx.is_connected():
        raise
    else:
        L = nx.average_shortest_path_length(G, weight=weight)
    return L

def calculate_node_clustering_coefs(G):
    # Undirected clustering (ignores direction and weight)    
    C_undir = nx.clustering(G.to_undirected())
    # Directed clustering (Fagiolo)
    C_dir = nx.clustering(G)
    return C_undir, C_dir

if __name__ == "__main__":
    eg_c = "#f05039" # orangey red
    gd_c = "#1f449c" # blue
    silver_c = "#c0c0c0"
    blue_silver = "#a8b8d0"
    gold_c = "#d4af37"
    soft_gold = "#f1c40f"

    #  Get all final experiments 
    linclab_drive  = Path("/network/projects/_groups/linclab_users/eg/")
    results_dir = linclab_drive/"modcog/exp8_final_combined"
    results_dict = get_results_dict(results_dir)
    print("Results loaded", results_dict.keys())

    # Restrict to lognormal experiments
    gd_exps_dict = results_dict.gd_log_normal 
    eg_exps_dict = results_dict.eg_log_normal
    gd_exp_list = [d for k,d in gd_exps_dict.items() if k.startswith("seed")]
    eg_exp_list = [d for k,d in eg_exps_dict.items() if k.startswith("seed")]
    seeds_keylist = [k for k in eg_exps_dict.keys() if k.startswith("seed")]
    all_exps_list = gd_exp_list[:] + eg_exp_list[:]

    n_seeds = len(gd_exp_list) # should be 5
    prune_proportions = [0.99, 0.98, 0.97, 0.96, 0.95]
    print(n_seeds)

    fig_path = Path("../figs/clustering/")
    for pp in prune_proportions:
        for seed_idx in range(n_seeds):
            print(seeds_keylist[seed_idx], pp)
            print("--------- ")
            eg_run = eg_exp_list[seed_idx]
            gd_run = gd_exp_list[seed_idx]
            eg_rnn = load_rnn(eg_run)
            gd_rnn = load_rnn(gd_run)
            prune_wrec(eg_rnn,prune_prop = pp, inverse_magnitude=False)
            prune_wrec(gd_rnn,prune_prop = pp, inverse_magnitude=False)
            # Herem for analysing slivers of weights, option for something like
            # prune_wrec(get_rnn_module(gd_model),prune_prop = 0.01, inverse_magnitude=True)
            gd_wrec = get_weight_hh(gd_rnn)
            eg_wrec = get_weight_hh(eg_rnn)

            #print("Processing weights")
            gEG = clean_weights(eg_wrec, abs_weights=True)
            gGD = clean_weights(gd_wrec, abs_weights=True)   
            
            #print("Calculating node clustering coefs")
            eg_clusterings = calculate_node_clustering_coefs(gEG)
            gd_clusterings = calculate_node_clustering_coefs(gGD)

            # save the clusterings for later analysis
            savedir = Path("/network/projects/_groups/linclab_users/eg/modcog/analysis/clustering")
            
            fname = f"eg_{seeds_keylist[seed_idx]}_{pp}_node_ccs.pkl"
            with open(savedir/fname, "wb") as f:
                pickle.dump(eg_clusterings, f)

            fname = f"gd_{seeds_keylist[seed_idx]}_{pp}_node_ccs.pkl"
            with open(savedir/fname, "wb") as f:
                pickle.dump(gd_clusterings, f)
            
            # plot histograms of clustering coefs
            clustering_types = ["Undirected", "Directed"]
            for idx, clustering_title in enumerate(clustering_types):
                print(f"{clustering_title} {seeds_keylist[idx]}")
                gd_ccs = list(gd_clusterings[idx].values())
                eg_ccs = list(eg_clusterings[idx].values()) 
                max_x = max(max(gd_ccs), max(eg_ccs))
                print("\t", max_x)
                print(" ")
                bin_linspace = np.linspace(0, max_x, 50)

                gd_counts, gd_bin_edges = np.histogram(gd_ccs, bin_linspace)
                eg_counts, eg_bin_edges = np.histogram(eg_ccs, bin_linspace)
                fig = plt.figure(figsize=(5,1.25), dpi=150)
                plt.stairs(gd_counts, gd_bin_edges, fill=True, alpha=0.7, color=gd_c, edgecolor='black', linewidth=0.7, label="GD")
                plt.stairs(eg_counts, eg_bin_edges, fill=True, alpha=0.7, color=eg_c, edgecolor='black', linewidth=0.7, label="EG")
                plt.xlabel("Clustering coefficient")
                plt.ylabel("Numbers of nodes")
                plt.title(f"{clustering_title} Clustering\n {seeds_keylist[seed_idx]}")
                plt.legend()
                sns.despine()
                plt.savefig(fig_path/f"{seeds_keylist[seed_idx]} Pruned to {pp} {clustering_title}.png", bbox_inches='tight')
    
