#!/bin/bash
#SBATCH --job-name=ngcm-sressel
#SBATCH --output=multi_day_mean_state_log.out
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --account=bewd-delta-gpu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate neural_gcm

module purge
module load cuda/11.8
module load nvidia/24.5
module load cudnn/8.9.0.131

python -c "import jax; print(jax.__version__); print(jax.lib.xla_bridge.get_backend().platform)"

export LD_LIBRARY_PATH=/sw/external/libraries/cudnn-linux-x86_64-8.9.0.131_cuda11-archive/lib:/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/cuda-11.8.0-vfixfmc/lib64:$LD_LIBRARY_PATH

correction_interval=1
time python multi_day_mean_state_test.py "$correction_interval"
