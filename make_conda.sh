#!/bin/bash
module load anaconda/3

# torch version 2.3, numpy<2 
# conda create -n eg_rnns 
conda create --prefix $SCRATCH/.conda/eg_rnns conda conda-libmamba-solver python=3.9 pytorch=2.3 pytorch-cuda=11.8 matplotlib scipy seaborn "numpy<2" tqdm scikit-learn ipykernel cloudpickle pillow -c pytorch -c nvidia

# conda activate eg_rnns
conda activate $SCRATCH/.conda/eg_rnns

# install gym and neurogym
pip install gym==0.23.1  # compatible with neurogym (0.24.1 is the latest compatible but it recommends downgrading)
cd mod_cog

# neurogym 
git clone https://github.com/gyyang/neurogym.git
cd neurogym/
git checkout 89e71881426e547c10cffc96205aecda158a64b7
pip install -e .
cd ../..

# install other packages
pip install wandb
pip install hydra-core --upgrade
pip install tqdm
pip install scikit-learn
pip install statsmodels

# Optional: DSA analysis (mod_cog_dsa/). Not needed for training.
# pip install tables
# pip install git+https://github.com/mitchellostrow/DSA@9bcb1d66

cd eg_reach_oc
# MotorNet: clone, then apply local changes
# (GPU/device fixes + effector.grad_reset, which the reach env requires).
git clone https://github.com/OlivierCodol/MotorNet.git
cd MotorNet
git checkout 47b33dff4a7cd93904ee5dcc34db76708a31a900
git apply ../motornet_v0.2.0.patch
pip install -e .
cd ..

conda install -c anaconda ipykernel
python -m ipykernel install --user --name=eg_rnns
