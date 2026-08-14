import os
import sys
import time
import json
import DSA
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from argparse import ArgumentParser
from joblib import Parallel, delayed
import shutil


def make_database(job_id: int, a: dict, n_components: int, n_ranks: int, n_delays: int):

  # to throttle the speed of the parallel process if too fast
  # this is counter-intuitive but if too many processes finish fast the delayer will internally throttle the 
  # parallelization even more to avoid overhead issues, which ends up being even more detrimental in our (specific) 
  # case.
  # this is because the distribution of process speeds is pretty wide for us.
  start_time = time.time()

  hyperparameters = {
    "n_components": n_components,
    "n_ranks": n_ranks,
    "n_delays": n_delays,
  }
  df = pd.DataFrame({
    **{k: [v] for k, v in a.items()},
    **{k: [v] for k, v in hyperparameters.items()}
    })
  df["n_components"] = df["n_components"].astype("int16")
  df["n_ranks"] = df["n_ranks"].astype("int16")
  df["n_delays"] = df["n_delays"].astype("int16")

  query = query_from_jobs(job=a, hyperparameters=hyperparameters)

  # check if comparison was already logged in the main table
  filepath_main = os.path.join(OUTPUT_DIR, "sweep.h5")
  if os.path.exists(filepath_main):
    entry_already_exists = not pd.read_hdf(filepath_main, key="pairs", where=query).empty
    if entry_already_exists:
      if VERBOSE > 10:
        print(f"comparison already exists in {os.path.relpath(filepath_main, OUTPUT_DIR)}, skipping...")
      time.sleep(max(0, 0.1 - (time.time() - start_time)))
      return None

  # else check if temp file already exists and contains the comparison
  filepath_temp = os.path.join(TMP_PATH, os.path.splitext(os.path.basename(filepath_main))[0] + "_" + str(job_id) + ".h5")
  if os.path.exists(filepath_temp):
    entry_already_exists = not pd.read_hdf(filepath_temp, key="pairs", where=query).empty
    if entry_already_exists:
      if VERBOSE > 10:
        print(f"temp file already exists, skipping: {filepath_temp}")
      time.sleep(max(0, 0.1 - (time.time() - start_time)))
      return None

  # retrieve model data      
  with open(job2path(a), "rb") as f:
    X1 = np.load(f, allow_pickle=True)["acts"]
    # if len(np.unique([x.shape[0] for x in X1])) > 1:
    #   return None
    # print(a, X1.shape, np.unique([x.shape[0] for x in X1]))
    X1 = np.stack(X1).astype(np.float32)
    if X1.shape[1] <= n_delays:
      return None
  Y1 = PCA(n_components=n_components).fit_transform(X1.reshape(-1, X1.shape[-1]))
  Y1 = Y1.reshape(*X1.shape[:-1], n_components)
  N = Y1.shape[0] // 10  # 10-fold cross-validation
  X_train = Y1[N:, ...]  # train on folds 2-10
  X_test = Y1[:N, ...]  # test on fold 1

  dmd = DSA.DMD(X_train, n_delays=n_delays, verbose=False)
  dmd.fit(rank=n_ranks)
  Y_test = dmd.predict(X_test)

  pred_start = (n_delays - 1) * dmd.delay_interval + 1  # I don't get this but that's what the original example does
  stats = DSA.stats.compute_all_stats(X_test[pred_start:, :], Y_test[pred_start:, :], rank=n_ranks, norm=True)
  
  df["mse"] = [stats['MSE']]
  df["r2"] = [stats['R2']]
  df["correl"] = [stats['Correl']]

  if DRY_RUN:
    # abort before we actually start saving things but after fancy computations are done
    time.sleep(max(0, 0.1 - (time.time() - start_time)))
    return None
  
  if os.path.exists(filepath_temp):
    # temp file exists, but does not contain the comparison, adding to df
    df.to_hdf(filepath_temp, key="pairs", mode="a", format="table", data_columns=True, append=True)
    time.sleep(max(0, 0.1 - (time.time() - start_time)))
    return filepath_temp

  # temp file does not exist, create it
  overrides = {"task": 20, "model": 2}
  string_cols = df.select_dtypes(include="object").columns  # get all object (string) columns
  min_itemsize = {col: overrides.get(col, 5) for col in string_cols}  # defaults to 5, overrides if specified
  df.to_hdf(filepath_temp, key="pairs", mode="w", format="table", data_columns=True, min_itemsize=min_itemsize)

  time.sleep(max(0, 0.1 - (time.time() - start_time)))
  return filepath_temp


def query_from_jobs(job, hyperparameters):
  job = " & ".join([k + " == '" + str(v) + "'" for k, v in job.items()])
  hprm = " & ".join([k + " == '" + str(v) + "'" for k, v in hyperparameters.items()])
  return job + " & " + hprm


def job2path(job):
  basepath = os.path.join(DATA_PATH, job["model"], "seed_" + str(job["seed"]))
  filenames = [file for file in os.listdir(basepath) if job["task"] in file.split("_") and not file.startswith(".")]
  if len(filenames) != 1:
    raise ValueError(f"Expected exactly one file for job {job}, found {len(filenames)}: {filenames}")
  filename = filenames[0]
  return os.path.join(basepath, filename)


def parse_range_string(range_str):
  if isinstance(range_str, int):
    return [range_str]
  result = []
  parts = range_str.split(",")
  for part in parts:
    part = part.strip()
    if "-" in part:
      start, end = map(int, part.split("-"))
      result.extend(range(start, end + 1))
    else:
      result.append(int(part))
  return result


#----------------------
# MAIN CODE
#----------------------
if __name__ == "__main__":
  # -------------------------------
  # Sort out arguments
  # -------------------------------
  helper = {
    "cfg": "The config file to use (default: sweep.json)",
    "v": "Print progress of parallel workers or not, passed to joblib as-is so more in joblib docs (default: 0)",
    "c": "Number of cores for processing (default: 1)",
    "dir": "Output directory in which the output file is located (default ./stats/)",
    "f": "Force overwrite the output file entry if it already exists (default: False)",
  }

  parser = ArgumentParser()
  parser.add_argument("config", help=helper["cfg"], type=str, default="sweep.json")
  parser.add_argument("-c", "--n_cores", dest="n_cores", help=helper["c"], type=int, default=1)
  parser.add_argument("-v", "--verbose", dest="verbose", help=helper["v"], type=int, default=0)
  parser.add_argument("-f", dest="force_overwrite", help=helper["f"], action='store_true')
  parser.add_argument("-o", "--output_dir", dest="output_dir", help=helper["dir"], type=str, default="")
  parser.add_argument("--dry-run", dest="dry_run", help="run this script as a dry run.", action='store_true')
  args = parser.parse_args()

  N_CORES = args.n_cores
  VERBOSE = args.verbose
  DRY_RUN = args.dry_run
  FORCE_OVERWRITE = args.force_overwrite
  CONFIG_FILE = args.config
  OUTPUT_DIR = args.output_dir

  ROOT = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
  OUTPUT_DIR = OUTPUT_DIR if OUTPUT_DIR != "" else os.path.join(ROOT, "stats")
  TMP_PATH = os.path.join(OUTPUT_DIR, "temp")
  CONFIG_FILE = os.path.join(ROOT, "cfg", CONFIG_FILE)

  os.makedirs(TMP_PATH, exist_ok=True)

  if DRY_RUN:
    print("\nRUNNING THIS IN DRY RUN MODE.\n")

  print(f"processing from config file --> {CONFIG_FILE}")
  with open(CONFIG_FILE, "r") as f:
    CFG = json.load(f)

  DATA_PATH = Path("/network/projects/_groups/linclab_users/eg/modcog/analysis_dsa_fixlen")
  if not os.path.isdir(DATA_PATH):
    DATA_PATH = os.path.abspath(os.path.join(".", "data"))
  CFG["seed"] = parse_range_string(CFG["seed"])
  CFG["n_ranks"] = parse_range_string(CFG["n_ranks"])
  CFG["n_components"] = parse_range_string(CFG["n_components"])
  CFG["n_delays"] = parse_range_string(CFG["n_delays"])

  # create all combinations of model-model comparisons
  keys = ["model", "seed", "task"]
  combinations_as_lists = list(itertools.product(*[CFG[key] for key in keys]))
  combinations = [{key: prm for key, prm in zip(keys, comb)} for comb in combinations_as_lists]

  # create all combinations of hyperparameters for each comparisons
  keys = ["n_components", "n_ranks", "n_delays"]
  combinations_as_lists = list(itertools.product(*[CFG[key] for key in keys]))
  hyperparameter_sets = [{key: prm for key, prm in zip(keys, comb)} for comb in combinations_as_lists]

  # combine the above into all the jobs we want
  jobs = [{"a": combination, **hprm_set} for hprm_set in hyperparameter_sets for combination  in combinations]

  # send the jobs
  n_jobs = len(jobs)
  print("processing {} jobs...".format(n_jobs))
  parallel = Parallel(n_jobs=min(n_jobs, N_CORES), verbose=VERBOSE)
  delayer = delayed(make_database)
  temp_files = parallel(delayer(job_id=job_id, **job) for job_id, job in enumerate(jobs))

  # temp files consolidation logic
  # temp_files = [os.path.join(TMP_PATH, f) for f in os.listdir(TMP_PATH) if f.endswith(".h5")]
  temp_files = [f for f in temp_files if f is not None]
  if len(temp_files) == 0:
    print("No temp. files were successfully created. Exiting.")
    shutil.rmtree(TMP_PATH)
    sys.exit(0)

  df = pd.DataFrame()
  for f in temp_files:
    df_tmp = pd.read_hdf(f, key="pairs")
    df = pd.concat([df, df_tmp])
    
  master_file = os.path.join(OUTPUT_DIR, "_".join(os.path.basename(f).split("_")[:-1]) + ".h5")
  
  if os.path.exists(master_file) and not DRY_RUN:
    # check how many indexes already exist
    with pd.HDFStore(master_file) as store:
      last_idx = store.get_storer("pairs").nrows
    # change the current data's indexing accordingly, then append to existing file
    df.index = range(last_idx, last_idx + len(df))
    df.to_hdf(master_file, key="pairs", mode="a", format="table", data_columns=True, append=True)
  elif not DRY_RUN:
    overrides = {"task": 20, "model": 2}
    string_cols = df.select_dtypes(include="object").columns  # get all object (string) columns
    min_itemsize = {col: overrides.get(col, 5) for col in string_cols}  # defaults to 5, overrides if specified
    df.reset_index(drop=True, inplace=True)
    df.to_hdf(master_file, key="pairs", mode="w", format="table", data_columns=True, min_itemsize=min_itemsize)

  shutil.rmtree(TMP_PATH)
