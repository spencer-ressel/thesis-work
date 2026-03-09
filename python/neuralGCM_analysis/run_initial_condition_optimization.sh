#!/bin/bash
#PBS -N ngcm-sressel
#PBS -A uwas0172
#PBS -q main
#PBS -l select=1:ncpus=1:ngpus=1:gpu_type=a100
#PBS -l walltime=00:45:00
#PBS -o initial_condition_optimization.out
#PBS -j oe

module purge

module load conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neural_gcm

module load cuda/12.9.86
module load cudnn/9.19.0.56

python -c "import jax; print(jax.__version__); print(jax.lib.xla_bridge.get_backend().platform)"

echo "LD_LIBRARY_PATH:"
echo $LD_LIBRARY_PATH

echo "CUDA_VISIBLE_DEVICES:"
echo $CUDA_VISIBLE_DEVICES

python3 ngcm_optimal.py
