from pathlib import Path
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import seaborn as sns
import pickle
import json

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

from mod_cog.analysis.script_clustering_analysis import (
    load_model,
    get_results_dict,
    load_rnn,
    prune_wrec,
    get_weight_hh,
    clean_weights,
    calculate_node_clustering_coefs
)

if __name__ == "__main__":
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
    
    small_world_directed = {}
    for alg in ["eg", "gd"]:
        small_world_directed[alg]   = {pp: [] for pp in prune_proportions }
    print(small_world_directed)

    for pp in prune_proportions:
        for seed_idx in range(n_seeds):
            print(f"{pp}, {seeds_keylist[seed_idx]}")
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

            print("Processing weights")
            gEG = clean_weights(eg_wrec, abs_weights=True)
            gGD = clean_weights(gd_wrec, abs_weights=True)   
            
            print("Calculating small world coefs")
            for alg, G in [("eg",gEG),("gd",gGD)]:
                average_path_len_directed = nx.average_shortest_path_length(G, weight=None)
                average_clustering_directed = nx.average_clustering(G, weight=None)
                print("Av clustering, path:", alg,average_clustering_directed, average_path_len_directed)
                
                # make a random graph 
                n = G.number_of_nodes()
                p = nx.density(G)
                G_rand = nx.erdos_renyi_graph(n, p, directed=True)
                print(alg, "densities", nx.density(G_rand), p)

                average_clustering_rand_dir =  nx.average_clustering(G_rand, weight=None)
                average_path_length_rand_dir =  nx.average_shortest_path_length(G_rand, weight=None)
    
                sw_directed = (average_clustering_directed/average_clustering_rand_dir) / (average_path_len_directed/average_path_length_rand_dir)

                small_world_directed[alg][pp].append(sw_directed)
                
            print(f"EG|GD Small world directed", small_world_directed["eg"][pp], small_world_directed["gd"][pp])

    savedir = Path("/network/projects/_groups/linclab_users/eg/modcog/analysis/clustering")
    with open(savedir/"small_world_directed.json", "w") as f:
        json.dump(small_world_directed, f)
    