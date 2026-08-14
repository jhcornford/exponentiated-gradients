# mod_cog

ModCog RNN training and pruning experiments. Run scripts from inside `mod_cog/`. Network training uses cached task data, which relies on neurogym being cloned and installed (see `make_conda.sh` and the top-level README).

Training:

- `train_mod_cog.py`: main training loop
- `exp8_combined_loss.sh`: hyperparameter search
- `exp8_combined_loss_final.sh`: final EG/GD runs, saves to `exp8_final_combined` (Fig 1)
- `exp8_combined_loss_final_eg_unsigned.sh`: sign-function-free EG ablation (Fig S2)
- `run_sign_constrained_gd.sh`: sign-constrained (projected) GD (Figs S5, S6)

Pruning re-training:

- `exp9_pruning.sh`, `prune_experiments.py`: prune final EG/GD models and re-train (Fig 3)
- `exp9_pruning_cSGD.sh`, `prune_experiments_cSGD.py`: prune final cSGD models and re-train (Fig S6)

Analysis:

- `analysis/`: figure notebooks (see top-level README for relation to figures)
- `analysis_1_solutions.sh`, `analysis_1_solutions.py`: per-task accuracy CSVs feeding
  `analysis/solution_order.ipynb` (Fig 1)
- `alpha/`: eigenspectrum / power-law-fit utilities used by `analysis/activity_spectral.ipynb` (Fig S3)