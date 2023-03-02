wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh
chmod -v +x Miniconda3-py310_23.1.0-1-Linux-x86_64.sh

# check the shasum
echo "32d73e1bc33fda089d7cd9ef4c1be542616bd8e437d1f77afeeaf7afdb019787 *Miniconda3-py310_23.1.0-1-Linux-x86_64.sh" | shasum --check

./Miniconda3-py310_23.1.0-1-Linux-x86_64.sh

conda create --name tf_env python=3.10

conda activate tf_env

conda install -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0

mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

pip install --upgrade setuptools pip

pip install nvidia-pyindex

pip install nvidia-tensorrt

echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/python3.10/site-packages/tensorrt/' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

pip install tensorflow==2.11.0

