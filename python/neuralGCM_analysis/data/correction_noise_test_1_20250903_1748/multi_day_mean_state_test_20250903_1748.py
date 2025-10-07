print(f"{'Loading imports...':<30}", end="")
import gcsfs
import jax
import numpy as np
import pickle
import xarray
import json
import os
import pandas as pd
from datetime import datetime
from pathlib import Path

from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils
import neuralgcm
from neuralgcm import PressureLevelModel
base_directory = Path("/u/sressel/neuralgcm/model_outputs")
print('\u2713')

print(f"{'Specifying model...':<30}", end="")
# model_name = '../neuralgcm/neuralgcm/checkpoints/deterministic_1_4_deg.pkl'
model_name = '../neuralgcm/neuralgcm/checkpoints/deterministic_2_8_deg.pkl'
with open(f"{model_name}", "rb") as f:
    ckpt = pickle.load(f)

model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)
print('\u2713')

print(f"{'Loading initial conditions...':<30}", end="")

demo_start_time = '2000-01-15'

data_inner_steps = 24  # process every 24th hour


eval_era5 = xarray.load_dataset(
    f"../neuralgcm/neuralgcm/input/2_8_thirty_year_DJF_climatology.nc"
).isel(time=0, drop=True).expand_dims(time=[pd.Timestamp('2000-01-15')])
print('\u2713')

#############  MODEL
print(f"{'Configuring model...':<30}", end="")
inner_steps = 24  # save model outputs at 'inner steps' intervals (hr)
outer_steps = 1 * 24 // inner_steps  # total of 'outer steps' (days)
timedelta = np.timedelta64(1, 'h') * inner_steps
times = (np.arange(outer_steps) * inner_steps)  # time axis in hours

import pandas as pd
# initialize model state (bring ERA5 snapshot --> encoding into the model's latent state)
inputs = model.inputs_from_xarray(eval_era5.isel(time=0))
input_forcings = model.forcings_from_xarray(eval_era5.isel(time=0))
random_key = 37
rng_key = jax.random.key(random_key)  # optional for deterministic models
#rng_key = jax.random.key(42)  # optional for deterministic models
initial_state = model.encode(inputs, input_forcings, rng_key)

# use persistence for forcing variables (SST and sea ice cover is fixed to t=0)
all_forcings = model.forcings_from_xarray(eval_era5.head(time=1))
print('\u2713')

print(f"{'Calculating mean-state correction...':<30}", end="")
# make forecast
final_state, predictions = model.unroll(
    initial_state,
    all_forcings,
    steps=outer_steps,
    timedelta=timedelta,
    start_with_input=True,
)
predictions_ds = model.data_to_xarray(predictions, times=eval_era5.isel(time=[0]).time)
print('\u2713')

# Calculate the mean state correction term
mean_state_correction = final_state.state - initial_state.state

print(f"{'Running model...':<30}", end="")
new_predictions_list = []
n_days = int(30*(24/inner_steps))
noise_amplitude = 10e-6
for day in np.arange(0, n_days, 1):

    noise = jax.random.normal(rng_key, shape=mean_state_correction.vorticity.shape) * noise_amplitude
    corrected_initial_state = initial_state.replace(state = final_state.state - mean_state_correction)
    corrected_initial_state = corrected_initial_state.replace(
        state = corrected_initial_state.state.replace(
            **{
                'vorticity': corrected_initial_state.state.vorticity + noise,
                'divergence': corrected_initial_state.state.divergence + noise,
                'temperature_variation': corrected_initial_state.state.temperature_variation + noise,
                'sim_time': initial_state.state.sim_time
            }
        )
    )
    
#    corrected_initial_state = corrected_initial_state.replace(memory = final_state.memory - mean_state_correction_memory)

    # make forecast
    final_state, new_predictions = model.unroll(
        corrected_initial_state,
        all_forcings,
        steps=outer_steps,
        timedelta=timedelta,
        start_with_input=True,
    )
    new_predictions_list.append(model.data_to_xarray(new_predictions, times=predictions_ds.isel(time=[0]).time))

concatenated_predictions_ds = xarray.concat(
    xarray.align(*[prediction for prediction in new_predictions_list]),
    dim='new_dim'
)
concatenated_predictions_ds = concatenated_predictions_ds.isel(time=0, drop=True).rename({'new_dim':'time'})
print('\u2713')

print(f"{'Saving output...':<30}", end="")
# Log model/experiment parameters
experiment_parameters = {
    'experiment_name': 'correction_noise_test_1',
    'description': 'This test uses the 2.8 deg model and 24-hr timesteps with random noise added to the initial condition at each time step',
    'model_name': model_name,
    'demo_start_time': demo_start_time,
    'inner_steps': inner_steps,
    'outer_steps': outer_steps,
    'timedelta': str(timedelta),
    'n_days': n_days,
    'random_key': random_key,
    'noise_amplitude': str(noise_amplitude)
}

# Create a unique identifier for each experimental run
file_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
#output_filename = f"multi_day_{demo_start_time}_deterministic_cf_corrected_30_minute_timesteps.nc"

# Save each experiments data to a new folder
output_folder_name = f"{experiment_parameters['experiment_name']}_{file_timestamp}"
output_folder = base_directory / output_folder_name
output_folder.mkdir(parents=True, exist_ok=False)

# Save the experiment parameters
json_path = output_folder / "experiment_parameters.json"
with open(json_path, "w") as f:
    json.dump(experiment_parameters, f, indent=4)

# Save the experiment data
concatenated_predictions_ds.to_netcdf(output_folder / "model_output.nc")

# Create running experiment log
experiments_log_file = base_directory / "experiment_log.json"
if experiments_log_file.exists():
    with open(experiments_log_file, "r") as f:
        log_data = json.load(f)

else:
    log_data = []

log_data.insert(
    0,
    {
        'experiment_name': experiment_parameters['experiment_name'],
        'timestamp': file_timestamp,
        'description': experiment_parameters['description']
    }
)

# Save updated log
with open(experiments_log_file, "w") as f:
    json.dump(log_data, f, indent=4)

# Path to the running script
script_path = os.path.abspath(__file__)

# Read the source code
with open(script_path, 'r') as f:
    script_contents = f.read()

filename = f"multi_day_mean_state_test_{file_timestamp}.py"

# Save in the experiment folder
save_path = os.path.join(output_folder, filename)
with open(save_path, 'w') as f:
    f.write(script_contents)

print('\u2713')

print("Finished")
