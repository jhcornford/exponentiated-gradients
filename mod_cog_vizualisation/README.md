# mod_cog_vizualisation

- `save_activity.ipynb` — runs the trained ModCog models (`exp8_final_combined`
  checkpoints) over all 82 tasks using the fixed-duration task variants in
  `mod_cog_tasks_no_randomness.py`, and saves per-task activity
  (`{id}_{task}_data.npz`) to (`modcog/analysis_dsa_fixlen/`). These
  files are the input to the DSA pipeline in `mod_cog_dsa/` (Fig 2A/C/D).
- `dsa_dynamics_viz.ipynb` — loads the files written by
  `save_activity.ipynb` (`modcog/analysis_dsa_fixlen/`) and plots per-period PCA
  trajectories 

Helpers: `viz_utils.py` (loading / period slicing / PCA / plotting);
`save_utils.py` (model loading / evaluation utilities the save notebook
imports). 
