# ExponentiatedGradients

Code associated with the paper "Exponentiated gradients support effective learning in biologically relevant scenarios with brain-like synaptic distributions" 

Jonathan Cornford, Roman Pogodin, Arna Ghosh, Kaiwen Sheng, Brendan A. Bicknell,
Olivier Codol, Pingsheng Li, Beverley A. Clark, Guillaume Lajoie, Blake A. Richards

2026

## Installation

```bash
source make_conda.sh
```

This creates a conda environment, and installs two external libraries:

- **neurogym** (for Mod-Cog tasks) — cloned into `mod_cog/neurogym/` and pip installed
  editable so `import neurogym` resolves via the environment not the path. From https://github.com/gyyang/neurogym at commit
  `89e7188`.  Required to generate the Mod-Cog data cache (see below). Note **gym** version `0.23.1` is used for neurogym compatibility.
- **MotorNet** — the biomechanical arm/muscle simulator used by the reach (continous control) experiments
  (`eg_reach_oc/`). Cloned into `eg_reach_oc/MotorNet/`, from
  https://github.com/OlivierCodol/MotorNet at v0.2.0 (`47b33df`),
  then a local patch (`eg_reach_oc/motornet_v0.2.0.patch`) is applied: GPU/device fixes plus a gradient-reset helper.

Requires `pip>=21.3`. Python 3.9. PyTorch 2.3.x (see `make_conda.sh`).

## Repository structure

- `mod_cog` — Mod-Cog training and pruning experiments
- `eg_reach_oc` — motor-learning reach (continous control) experiments (uses MotorNet)
- `r_of_k` — single, point neuron learning experiments
- `src` — shared utilities, including the EG optimizer implementations

The `.sh` job scripts target SLURM and will need adapting to your setup. Each script requires
`sbatch_out/` and `sbatch_err/` to exist in the directory you submit from.
Training scripts import `src` via a relative path, so run each from within its own experiment
directory.

## Mod-Cog data caching

The Mod-Cog tasks come from neurogym. Rather than sampling task data live during training (slow), the code samples fixed train/val/test splits once and caches them to disk as NumPy memmap
files. Generate the cache with:

```bash
cd src
python mod_cog_data_utils.py     # writes memmap files to $SCRATCH/eg_paper/mod_cog_data_v2/
```

`train_mod_cog.py` then loads from that cache (or from `$SLURM_TMPDIR/data` if present). 

## Analysis and figures

Figure-producing notebooks are in each experiment's `analysis/` folder (plus `mod_cog_dsa/`,
`mod_cog_vizualisation/`, and `imagenet_plotting/`). The paper figure each one is related to is detailed below. 

Many analysis notebooks save and load intermediate results to/from the original cluster paths (e.g. `/network/projects/.../linclab_users/eg/...`) and will need adjusting for your set-up. Subdirectories that notebooks save to (e.g. `figs/`, `plot_data/`) will also need to be created. For the Imagenet experiments `imagenet_plotting/` contains the wandb exports from the simulation runs.  

### Main figures

| Figure | Content | Scripts and notebooks |
|---|---|---|
| **Fig 1** | Mod-Cog training | In `mod_cog` first run `exp8_combined_loss_final.sh`, then `analysis_1_solutions.sh` (per-task accuracy CSVs). Figure panels produced by `weights_and_accuracy.ipynb` and `solution_order.ipynb`, both in `mod_cog/analysis/` |
| **Fig 2** | Dynamical similarity analysis (DSA) | First run `mod_cog_vizualisation/save_activity.ipynb` (saves per-task activity), then `mod_cog_dsa/batch-compare.py` (pairwise DSA distances). Panels 2A, 2C, 2D produced by `mod_cog_dsa/dsa_compare_eg_vs_gd.ipynb`; panel 2B by `mod_cog_vizualisation/dsa_dynamics_viz.ipynb` |
| **Fig 3** | Pruning experiments | `mod_cog/analysis/pruning.ipynb` (Panels B, C). Prune and retrain `mod_cog/prune_experiments.py`, `mod_cog/analysis/pruning_relearn.ipynb` (Panels D-F) |
| **Fig 4** | Point neuron experiments | In `r_of_k` first run `run_sweeps.sh` then plots produced by `r_of_k/analysis/sparse_inputs_rofk.ipynb` |
| **Fig 5** | Continuous control with noise | In `eg_reach_oc` first run `es1_final_cosdecay_0irrel.sh` and `es1_final_cosdecay_500irrel.sh`. Figure panels produced by `eg_reach_oc/analysis/reach_control.ipynb` (also produces S9) |

### Supplementary figures

| Figure | Content | Scripts and notebooks |
|---|---|---|
| **S1** | Mod-Cog task examples | `mod_cog/analysis/task_examples.ipynb` |
| **S2** | Sign-function-free EG | Train via `mod_cog/exp8_combined_loss_final_eg_unsigned.sh`. Figure panel is the `eg_unsigned` curve of the accuracy cell in `mod_cog/analysis/cSGD_weights_accuracy.ipynb`, plotted alone|
| **S3** | GD/EG activity histograms + spectral analysis | `mod_cog/analysis/activity_spectral.ipynb` |
| **S4** | Dual-space changes across weight initialisations | `mod_cog/analysis/dual_space_changes.ipynb` |
| **S5** | Sign-constrained (projected) GD | In `mod_cog` first run `run_sign_constrained_gd.sh`. Figure panels are produced by `cSGD_figure_plotting.ipynb`, which loads intermediate data saved by `cSGD_weights_accuracy.ipynb` and `cSGD_weight_hists.ipynb`, all in `mod_cog/analysis/` |
| **S6** | Sign-constrained GD pruning | In `mod_cog` first run `run_sign_constrained_gd.sh` then `prune_experiments_cSGD.py`. Figure panels produced by `cSGD_pruning.ipynb`, `cSGD_pruning_relearn.ipynb`, `cSGD_figure_plotting.ipynb` in `mod_cog/analysis/`|
| **S7** | Reverse (largest-first) magnitude pruning | `mod_cog/analysis/reverse_pruning.ipynb` |
| **S8** | Small-world analysis of large recurrent weights | In `mod_cog/analysis` run `script_clustering_analysis.py`, `script_shortest_path_analysis.py`, and `script_small_world_analysis.py`. Figure panels generated by `small_world_hubs.ipynb`, `small_world_plotting.ipynb` |
| **S9** | Continuous-control learning curves | `eg_reach_oc/analysis/reach_control.ipynb` (same notebook as Fig 5) |
| **S10** | Continuous-control weight distributions | `eg_reach_oc/analysis/reach_weight_dists.ipynb` |
| **S11** | ResNet-50 ImageNet results | `imagenet_plotting/plot_training_curves.ipynb` |

## Multicompartment (NEURON) simulations

The multicompartment neuron simulations are not yet included here. That code was adapted from Bicknell & Häusser 2021, *A synaptic learning rule for exploiting nonlinear dendritic computation* (https://github.com/babicknell/Dendrites); an updated package is in preparation for publication.

## License

MIT — see `LICENSE`.

## Third-party code and licenses

- **MotorNet** is cloned at setup and `eg_reach_oc/motornet_v0.2.0.patch` applies modifications. MotorNet and the patch is subject to to GPLv3, not MIT.
- `src/prune.py` (PyTorch v2.1.2) and `src/adamw_eg.py` (PyTorch v2.3.0) are modified copies of
  PyTorch code (BSD-3-Clause); 
- `src/yang19.py` is a copy of a file from neurogym at commit `946162e` (2020-11-19), when
  neurogym was MIT-licensed (it relicensed to Apache-2.0 in December 2024).
- `src/mod_cog_tasks.py` is modified from [Mod_Cog](https://github.com/mikailkhona/Mod_Cog) (CC0) at commit `5dd30a1`.

MotorNet and neurogym are cloned at setup, not redistributed here, only the MotorNet patch
and the copied files above carry third-party licenses.

See `THIRD_PARTY_LICENSES` for full license texts