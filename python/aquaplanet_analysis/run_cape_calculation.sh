#!/bin/bash -l

#PBS -N cape_calculation_m4k_9
#PBS -A uwas0152
#PBS -l select=1:ncpus=1:mem=256GB
#PBS -l walltime=01:30:00
#PBS -q casper
#PBS -o aquaplanet_logs
#PBS -j oe

CONDA_ENV="modified-npl"
ilat=9

module load conda
conda activate "${CONDA_ENV}"

python3  cape_calculation.py "$ilat"
