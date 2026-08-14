
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA

class TaskAnalysis():
    """
    Class to hold data for an algorithm and task across seeds.
    """
    def __init__(self, root_dir, filename, seed_strs):
        self.root_dir = root_dir
        self.filename = filename
        self.seed_strs = seed_strs
        
        # we assume the filename is of form e.g: 0_go_data.npz
        self.env_idx = filename.split("_")[0]
        self.env_str = filename.split("_")[1]
        
        # init datastructures to hold analyses
        self.seed_data = dict() # of dicts
        self.seed_period_slices = dict()
        self.seed_pca_objs = dict() # todo possibly seperated into the individual sections.        

def load_seeds(task_analysis_obj):
    for seed in task_analysis_obj.seed_strs:
        seed_acts_path = task_analysis_obj.root_dir/seed/task_analysis_obj.filename
        seed_data = np.load(seed_acts_path, allow_pickle = True)
        task_analysis_obj.seed_data[seed] = seed_data


def process_timings(timings):
    """
    timings is of the form:
    array([{'fixation': 5, 'stimulus': 10, 'delay': 10, 'decision': 15},
           {'fixation': 5, 'stimulus': 10, 'delay': 10, 'decision': 15},
           :
           :
    the trial_timing are an end index, hence -1 for a slice
    see timings.append(env.end_ind.copy()) in save_activity.ipynb

    Returns a dict: period_name -> list of slices across trials.
    """   
    # check non decreasing & have all the same keys
    for d in timings:
        lst = list(d.values())
        assert all(x <= y for x, y in zip(lst, lst[1:])), "Timing entries are not non-decreasing"
        assert d.keys() == timings[0].keys(), "Dictionaries have different keys"
        
    # extract period_slices by period
    period_names = list(timings[0].keys())
    period_slices = {p: [] for p in period_names}

    for trial_timing in timings:
        start = 0
        for period in period_names:
            end = trial_timing[period]
            #start = np.max(0, start-1)
            s = slice(start, end) if start <= end else slice(0, 0)
            period_slices[period].append(s)
            start = end 

    return period_slices


def get_trial_targets(period_slices, targets_arr):
    slice_list = period_slices["decision"]
    trial_targets = []
    for trial_idx, slice_ in enumerate(slice_list):
        trial_targets.append(int(targets_arr[trial_idx][slice_][0]))
    return trial_targets


def stack_acts(acts, period_slices):
    period_acts = {  }
    for period, slice_list in period_slices.items():
        arr = np.empty(acts.shape[0], dtype=object)
        for trial_idx, period_slice in enumerate(slice_list):
            arr[trial_idx] = acts[trial_idx][period_slice]
        period_acts[period] = arr
    return period_acts


def run_pca(arr, n_components=5):
    """
    arr is object array n_trial, seq_len, n_feats
    seq_len can vary
    """
    original_lengths = [a.shape[0] for a in arr]
    
    pca_obj = PCA(n_components=n_components)
    stacked_arr = np.vstack(arr)
    if stacked_arr.shape[0] ==0:
        print("Warning: No data!")
        return None, None
    pca_transformed = pca_obj.fit_transform(stacked_arr)
    
    # Reconstruct arr with original sequence lengths
    projected_arr = np.empty(arr.shape[0], dtype=object)
    start = 0
    for idx, length in enumerate(original_lengths):
        end = start + length
        projected_arr[idx] = pca_transformed[start:end]
        start = end
    
    return projected_arr, pca_obj


def plot_pca_trajectories(pca_proj, dim=2, trial_targets=None, show=True, tight=True):
    """
    pca_proj: dict[str, list of np.ndarray], where each array is [T, dim]
              (T = timepoints or trial length, dim = PCA dimensions)
    dim: 2 or 3 — number of dimensions to plot
    """
    assert dim in (2, 3), "Only 2D or 3D plotting supported"
    if trial_targets is not None:
        cmap = plt.cm.hsv(np.linspace(0, 1, 16, endpoint=False)) # 16 classes
 
    periods = list(pca_proj.keys())
    n_periods = len(periods)

    fig = plt.figure(figsize=(2 * n_periods, 2), dpi=300)
    axs = []
    for j, period in enumerate(periods):
        if dim == 3:
            ax = fig.add_subplot(1, n_periods, j + 1, projection='3d')
        else:
            ax = fig.add_subplot(1, n_periods, j + 1)
        axs.append(ax)
        period_pca = pca_proj[period]

        for trial_idx, proj in enumerate(period_pca):
            if trial_targets is not None:
                c = cmap[trial_targets[trial_idx]]
            else: c=None
            if dim == 3:
                ax.plot(proj[:, 0], proj[:, 1], proj[:, 2], alpha=0.6, color=c)
            else:
                ax.plot(proj[:, 0], proj[:, 1], alpha=0.6, color=c)

        ax.set_title(period)
        ax.set_xticks([])
        ax.set_yticks([])
        if dim == 3:
            ax.set_zticks([])

    if tight:
        plt.tight_layout()
    if show:
        plt.show()

    return fig, axs
