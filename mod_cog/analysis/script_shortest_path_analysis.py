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
from src import mod_cog_tasks as mct 
from src import mod_cog_data_utils as mcdu
from src import eg_utils
from src import analysis_utils as au
from src import plot_utils as pu
from src import prune 

import networkx as nx
from copy import deepcopy

from mod_cog.analysis.script_clustering_analysis import (
    load_model,
    get_results_dict,
    load_rnn,
    prune_wrec,
    get_weight_hh,
    clean_weights,
)


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

    shortest_path_results_dict = {}
    eg_shortest_path_results_dict = {}
    gd_shortest_path_results_dict = {}

    fig_path = Path("../figs/clustering/")
    for pp in prune_proportions:
        eg_shortest_path_results_dict[str(pp)] = []
        gd_shortest_path_results_dict[str(pp)] = []
        for seed_idx in range(n_seeds):
            print(seeds_keylist[seed_idx], pp)
            print("---- ---- ")
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
            
            print("Calculating average shortest path")
            #gEG_average_path_len_undirected = nx.average_shortest_path_length(gEG.to_undirected(), weight=None)
            gEG_average_path_len_directed   = nx.average_shortest_path_length(gEG, weight=None)

            #gGD_average_path_len_undirected = nx.average_shortest_path_length(gGD.to_undirected(), weight=None)
            gGD_average_path_len_directed   = nx.average_shortest_path_length(gGD, weight=None)

            print(gEG_average_path_len_directed)
            print(eg_shortest_path_results_dict)
        
            eg_shortest_path_results_dict[str(pp)].append(gEG_average_path_len_directed)
            gd_shortest_path_results_dict[str(pp)].append(gGD_average_path_len_directed)

        savedir = Path("/network/projects/_groups/linclab_users/eg/modcog/analysis/clustering")
        eg_df = pd.DataFrame(eg_shortest_path_results_dict)
        eg_df.to_csv(Path(savedir/"eg_shortest_path_results.csv"))
        
        gd_df = pd.DataFrame(gd_shortest_path_results_dict)
        gd_df.to_csv(Path(savedir/"gd_shortest_path_results.csv"))

        print(gd_df)
        print(eg_df)

            # # save the clusterings for later analysis
            # savedir = Path("/network/projects/_groups/linclab_users/eg/modcog/analysis/clustering")
            
            # fname = f"eg_{seeds_keylist[seed_idx]}_{pp}_av_shortest_path.pkl"
            # with open(savedir/fname, "wb") as f:
            #     pickle.dump(gEG_average_path_len_directed, f)

            # fname = f"gd_{seeds_keylist[seed_idx]}_{pp}_av_shortest_path.pkl"
            # with open(savedir/fname, "wb") as f:
            #     pickle.dump(gGD_average_path_len_directed, f)
            
            # plot histograms of clustering coefs
            # clustering_types = ["Undirected", "Directed"]
            # for idx, clustering_title in enumerate(clustering_types):
            #     print(f"{clustering_title} {seeds_keylist[idx]}")
            #     gd_ccs = list(gd_clusterings[idx].values())
            #     eg_ccs = list(eg_clusterings[idx].values()) 
            #     max_x = max(max(gd_ccs), max(eg_ccs))
            #     print("\t", max_x)
            #     print(" ")
            #     bin_linspace = np.linspace(0, max_x, 50)

            #     gd_counts, gd_bin_edges = np.histogram(gd_ccs, bin_linspace)
            #     eg_counts, eg_bin_edges = np.histogram(eg_ccs, bin_linspace)
            #     fig = plt.figure(figsize=(5,1.25), dpi=150)
            #     plt.stairs(gd_counts, gd_bin_edges, fill=True, alpha=0.7, color=gd_c, edgecolor='black', linewidth=0.7, label="GD")
            #     plt.stairs(eg_counts, eg_bin_edges, fill=True, alpha=0.7, color=eg_c, edgecolor='black', linewidth=0.7, label="EG")
            #     plt.xlabel("Clustering coefficient")
            #     plt.ylabel("Numbers of nodes")
            #     plt.title(f"{clustering_title} Clustering\n {seeds_keylist[seed_idx]}")
            #     plt.legend()
            #     sns.despine()
            #     plt.savefig(fig_path/f"{seeds_keylist[seed_idx]} Pruned to {pp} {clustering_title}.png", bbox_inches='tight')
    
