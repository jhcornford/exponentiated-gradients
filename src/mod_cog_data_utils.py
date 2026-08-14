import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

try:
    from . import eg_utils
except:
    import eg_utils

import os

class ModCogDataset(Dataset):
    def __init__(self, data_dir,split="train", in_memory=False):
        # Todo: take in a dict for changing a default dict of shape values
        split_sizes = {"train": 512*1000, "val": 512*100, "test": 512*100}
        self.device = eg_utils.get_device()
        #print(f"Using device: {self.device}")
        assert data_dir.exists()

        if (Path(data_dir)/f"mod_cog_{split}_x.memmap").exists():
            print(f"Loading {split} memmap files", data_dir)
            self.x = np.memmap(Path(data_dir)/f"mod_cog_{split}_x.memmap", dtype=np.float32, mode='r',
                                   shape=(split_sizes[split], 350, 34))
            self.y = np.memmap(Path(data_dir)/f"mod_cog_{split}_y.memmap", dtype=np.int8, mode='r',
                                      shape=(split_sizes[split], 350))
            if in_memory:
                self.x = torch.from_numpy(self.x[:].copy()).to(self.device)
                self.y = torch.from_numpy(self.y[:].copy()).to(self.device)
                self.memmap = False
            else:
                self.memmap = True
        else:
            self.memmap = False
            self.x = torch.from_numpy(np.load(Path(data_dir)/f"mod_cog_{split}.npz")['x'])
            self.y = torch.from_numpy(np.load(Path(data_dir)/f"mod_cog_{split}.npz")['y'])
            if in_memory:
                self.x = self.x.numpy()  # Convert x to a NumPy array
                self.x = torch.from_numpy(self.x).to(self.device)
                self.y = torch.from_numpy(self.y[:]).to(self.device)

    def __len__(self):
        return self.x.shape[0]
    
    @property
    def input_size(self):
        return self.x.shape[2]

    @property
    def seq_len(self):
        return self.x.shape[1]
    
    @property
    def output_size(self):
        # y: 0 outside the decision period, 0-15 during decision. Training adds +1 so
        # class 0 = fixation and classes 1-16 are the ring choices (hence 17 outputs,
        # one more than max(y)+1). NB the decision-period values are not plain ring
        # indices — see the Mod-Cog wrap quirk documented in my_set_groundtruth
        # (mod_cog_tasks.py).
        return 17
    
    def __getitem__(self, idx):
        if self.memmap:
            x_data = self.x[idx].copy()
            y_data = self.y[idx].copy()
            x = torch.from_numpy(x_data)
            y = torch.from_numpy(y_data)
            return x, y
        else:
            x = self.x[idx]
            y = self.y[idx]
            return x, y
        
if __name__ == '__main__':

    import mod_cog_tasks as mct

    def sizeof_fmt(num, suffix="B"):
        for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
            if abs(num) < 1024.0:
                return f"{num:3.1f}{unit}{suffix}"
            num /= 1024.0
        return f"{num:.1f}Yi{suffix}"

    # Quick code to write a cached dataset for the mod cog tasks 
    # Could be re-written, after deciding data format (e.g. compressed.npz that gets 
    # expanded to memmap files once, or just memmap files, or just npz files, etc.) 
    seed = 7
    batch_size = 512 # just used in conj. with split_sizes for setting size
    seq_len = 350

    split_sizes = {"train": 1000, "val": 100, "test": 100}
    #split_sizes = {"val": 100, "test": 100}

    overwrite = True # overwrite existing cached data
    data_dir = Path(os.environ['SCRATCH'])/"eg_paper/mod_cog_data_v2"
    #data_dir = Path("/network/scratch/c/cornforj/eg_paper/mod_cog_data_v2")
    print(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    print("---------------------------" )
    print("Writing a cached dataset for the mod cog tasks")

    # Reproducibility: the task code draws some per-trial values (e.g. integrate delays)
    # from global np.random, so this seed call — not the envs' own seeds — determines the
    # generated data. Also note training reads the cached files written here, so changes
    # to mod_cog_tasks.py have no effect until the dataset is regenerated.
    eg_utils.set_seed_all(seed)

    # get mod_cog_data object
    print("Constructing the mod cog dataset object...")
    dataset = mct.construct_mod_cog_dataset(batch_size, seq_len)
    input_size = dataset.env.observation_space.shape[0]
    output_size = dataset.env.action_space.n
    print(f"Input size: {input_size}, Output size: {output_size}")
    num_non_rule_inputs = 33 # this is for seperating out the one hot encoding of task rule encoding
    print(f"Assuming input features after {num_non_rule_inputs} are task rule encodings")

    for split in split_sizes.keys():
        if (data_dir/f"mod_cog_{split}_x.memmap").exists() and not overwrite:
            print(f"{split} data already exists, skipping")
        else:
            print(f"Sampling {split} data...")
            # draw samples and store in memory, make sure have a enough memory for this 
            x_data = np.empty((batch_size*split_sizes[split], seq_len, num_non_rule_inputs + 1), dtype=np.float32)
            y_data = np.empty((batch_size*split_sizes[split], seq_len), dtype=np.int8)

            print(f"X data shape,{x_data.shape}, size {sizeof_fmt(x_data.__sizeof__())}")
            print(f"Y data shape,{y_data.shape}, size {sizeof_fmt(y_data.__sizeof__())}")

            memmap_x_fname = data_dir / f"mod_cog_{split}_x.memmap"
            memmap_y_fname = data_dir / f"mod_cog_{split}_y.memmap"

            memmap_x_array = np.memmap(memmap_x_fname, dtype=np.float32, mode='w+', 
                                   shape=(batch_size*split_sizes[split], seq_len, num_non_rule_inputs + 1))
            memmap_y_array = np.memmap(memmap_y_fname, dtype=np.int8, mode='w+',
                                        shape=(batch_size*split_sizes[split], seq_len))
            
            pbar = tqdm(range(split_sizes[split]))
            for i in pbar:
                inputs, labels = dataset()
                taskrule_one_hot = inputs[:,:,33:]
                taskrule_labels  = np.argmax(taskrule_one_hot, axis=-1,keepdims=True)    
                x = np.concatenate([inputs[:,:,:33], taskrule_labels], axis=-1, dtype=np.float32)
                x_data[i*batch_size:(i+1)*batch_size] = x
                y_data[i*batch_size:(i+1)*batch_size] = labels.astype(np.int8)

                memmap_x_array[i*batch_size:(i+1)*batch_size] = x
                memmap_y_array[i*batch_size:(i+1)*batch_size] = labels.astype(np.int8)
                memmap_x_array.flush()
                memmap_y_array.flush()

                pbar.set_description(f"Drawing sample {i}: {sizeof_fmt(x_data.__sizeof__())}, {sizeof_fmt(y_data.__sizeof__())}")

            print("Writing compressed, x, y to disk...")
            np.savez_compressed(data_dir / f"mod_cog_{split}.npz", x=x_data, y=y_data)
            print("Done")
