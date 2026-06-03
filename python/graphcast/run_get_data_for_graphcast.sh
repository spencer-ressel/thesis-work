#!/bin/bash -l

CONDA_ENV="modified-npl"

module load conda 2>/dev/null || true
conda activate "${CONDA_ENV}"

START_DATE="2020-12-29T00:00:00.000000000"
END_DATE="2021-02-27T00:00:00.000000000"

start_time=$(date +%s)

export PYTHONPATH=/glade/u/home/sressel/thesis-work/python/auxiliary_functions/src:$PYTHONPATH
python3 get_data_for_graphcast.py "$START_DATE" "$END_DATE"

end_time=$(date +%s)
time_diff=$((end_time - start_time))
hours=$((time_diff / 3600))
minutes=$(( (time_diff % 3600) / 60 ))
seconds=$((time_diff % 60))
echo "Total time: ${hours}h ${minutes}m ${seconds}s"
