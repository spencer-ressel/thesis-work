#!/bin/bash -l

#PBS -N get_data_for_graphcast
#PBS -A uwas0152
#PBS -l select=1:ncpus=4:ngpus=0:mem=256GB
#PBS -l walltime=04:00:00
#PBS -q main
#PBS -j oe

CONDA_ENV="modified-npl"

module load conda 2>/dev/null || true
conda activate "${CONDA_ENV}"

# REPO_ROOT="${PBS_O_WORKDIR}"

# Run IC optimization from repo root.
# cd "${REPO_ROOT}"
start_time=$(date +%s)

python3 -u get_data_for_graphcast.py

end_time=$(date +%s)
time_diff=$((end_time - start_time))
hours=$((time_diff / 3600))
minutes=$(( (time_diff % 3600) / 60 ))
seconds=$((time_diff % 60))
echo "Total time: ${hours}h ${minutes}m ${seconds}s"
