#!/bin/bash -l

#PBS -N cape_calculation_p4k
#PBS -A uwas0152
#PBS -l select=1:ncpus=4:mem=512GB
#PBS -l walltime=24:00:00
#PBS -q preempt
#PBS -o aquaplanet_logs
#PBS -j oe

CONDA_ENV="modified-npl"

module load conda 2>/dev/null || true
conda activate "${CONDA_ENV}"

start_time=$(date +%s)
python3  cape_calculation.py

end_time=$(date +%s)
time_diff=$((end_time - start_time))
hours=$((time_diff / 3600))
minutes=$(( (time_diff % 3600) / 60 ))
seconds=$((time_diff % 60))
echo "Total time: ${hours}h ${minutes}m ${seconds}s"