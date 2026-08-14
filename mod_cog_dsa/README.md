# mod_cog_dsa — Dynamical Similarity Analysis (Fig 2)

1. **Activity capture**: `mod_cog_vizualisation/save_activity.ipynb` runs each trained
   network (EG/GD × seeds 5–9 × 82 tasks, from `mod_cog/exp8_final_combined`) on 200
   trials and saves hidden-unit activity. It uses
   `mod_cog_vizualisation/mod_cog_tasks_no_randomness.py`, which fixes the tasks'
   random timing draws so all trials of a task have equal length.
2. **Pairwise DSA**: `batch-compare.py` reduces each network's activity to 25 PCs and
   computes the DSA distance for every pair of networks. Hyperparameters (rank 25,
   10 delays) are set in `cfg/cfg.json`. Results are written to `stats/dsa.h5`.
3. **Hyperparameter sweep**: `batch-sweep.py` runs the DSA fit across a grid of rank
   (2–40) and delay (1–13) settings, set in `cfg/sweep.json`. Results are written to
   `stats/sweep.h5` and plotted by `dsa_hyperparameter_sweep.ipynb`.
4. **Figures**:
   - `dsa_compare_eg_vs_gd.ipynb` produces Fig 2A, 2C, 2D (Fig 2D group comparisons
     use two-sided Mann-Whitney U tests).
   - Fig 2B is produced by `mod_cog_vizualisation/dsa_dynamics_viz.ipynb`.

## Notes

- The batch scripts require the `DSA` package and PyTables, see the commented optional
  install lines in `make_conda.sh`.
