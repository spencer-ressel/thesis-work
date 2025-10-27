#################### DOCUMENTATION #########################
'''
Author: Spencer Ressel
Date: 2025-10-27

This code runs the NeuralGCM model using ERA5 data as an initial condition. It is designed to run the model in a quasi-linear way, with a mean-state term that does not evolve in time and a small MJO perturbation that evolves relative to the mean-state. 


'''
############################################################

######################### IMPORTS ##########################

print(f"{'Loading imports...':<30}", end="")
import gcsfs
import jax
from jax import config
import jax.numpy as jnp
import copy
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
from jax_helper_functions import *
base_directory = Path("/u/sressel/neuralgcm/model_outputs")
print('\u2713')
###############################################################

#################### SIMULATION PARAMETERS ####################
experiment_name = 'code_formatting_changes_test_5'
experiment_description = 'This is a 10-day test of the 2.8deg deterministic model with 1-hr timesteps. There is a perturbation and the mean-state is corrected.'
simulation_parameters = dict(
    add_perturbation = True,
    correct_mean_state = True,
    smooth_initial_conditions = False
)
#### Specify the number of timesteps
n_timesteps = int(10*24)
random_key = 37
rng_key = jax.random.key(random_key)  # optional for deterministic models

print(f"{'Specifying model...':<30}", end="")
model_name = '../neuralgcm/neuralgcm/checkpoints/deterministic_2_8_deg.pkl'
# model_name = '../neuralgcm/neuralgcm/checkpoints/models_v1_precip_stochastic_precip_2_8_deg.pkl'
with open(f"{model_name}", "rb") as f:
    ckpt = pickle.load(f)

model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)
print('\u2713')
###############################################################


################### LOAD INITIAL CONDITIONS ###################
print(f"{'Loading initial conditions...':<30}", end="")
demo_start_time = '2000-01-15'
eval_era5 = xarray.load_dataset(
    f"../neuralgcm/neuralgcm/input/2_8_thirty_year_DJF_climatology.nc"
).isel(time=0, drop=True).expand_dims(time=[pd.Timestamp('2000-01-15')])
print('\u2713')

#### Add a perturbation
mjo_composites = xarray.load_dataset(
    f"../neuralgcm/neuralgcm/input/MJO_phase_composites.nc"
)

print(f"{'Configuring model...':<30}", end="")
# initialize model state (bring ERA5 snapshot --> encoding into the model's latent state)
inputs_dict = model.inputs_from_xarray(eval_era5.isel(time=0))
perturbed_inputs_dict = model.inputs_from_xarray((eval_era5 + mjo_composites.sel(phase=5, drop=True)).isel(time=0))
input_forcings = model.forcings_from_xarray(eval_era5.isel(time=0))

# Encode initial data into model state
mean_state = model.encode(inputs_dict, input_forcings, rng_key)
perturbed_initial_state = model.encode(perturbed_inputs_dict, input_forcings, rng_key)

# Create an initial condition based on whether a perturbation is included
if simulation_parameters['add_perturbation']:
    initial_state = copy.deepcopy(perturbed_initial_state)
else:
    initial_state = copy.deepcopy(mean_state)
print('\u2713')
#################################################################

################## CALCULATE MEAN-STATE CORRECTION ##############
print(f"{'Calculating mean-state correction...':<30}", end="")
evolved_mean_state = model.advance(mean_state, input_forcings)

mean_state_correction = copy.deepcopy(mean_state)
config.update("jax_enable_x64", True)
mean_state_correction.state = tree_subtraction(
        tree_to_float64(evolved_mean_state.state),
        tree_to_float64(mean_state.state)
)
config.update("jax_enable_x64", False)
mean_state_correction_float32 = mean_state_correction.replace(state = tree_to_float32(mean_state_correction.state))
decoded_mean_state_correction = model.data_to_xarray(model.decode(mean_state_correction_float32, input_forcings), times=None)
print('\u2713')
#################################################################

######################### TIMESTEPPING ##########################
print(f"{'Running model...':<30}", end="")
predictions_list = []
predictions_list.append(model.data_to_xarray(model.decode(initial_state, input_forcings), times=None))
for timestep in np.arange(0, n_timesteps, 1):

    # Advance the model state forward by one timestep
    if timestep == 0:
        config.update("jax_enable_x64", False)
        advanced_state = model.advance(initial_state, input_forcings)
      
    else:
        config.update("jax_enable_x64", False)
        advanced_state = model.advance(corrected_advanced_state, input_forcings)

    if simulation_parameters['correct_mean_state']:
        # Apply the mean-state correction
        corrected_advanced_state = copy.deepcopy(initial_state)
        config.update("jax_enable_x64", True)
        corrected_advanced_state.state = tree_to_float32(
            tree_subtraction(
                tree_to_float64(advanced_state.state),
                mean_state_correction.state
            )
        )
        config.update("jax_enable_x64", False)
    else:
        # Advance the uncorrected state
        corrected_advanced_state = copy.deepcopy(advanced_state)

    # Save the advanced states to an xarray DataArray
    predictions_list.append(model.data_to_xarray(model.decode(corrected_advanced_state, input_forcings), times=None))

# Concatenate the states into a single DataArray
concatenated_predictions_ds = xarray.concat(
    xarray.align(*[prediction for prediction in predictions_list]),
    dim='time'
)
print('\u2713')
#################################################################


############################## OUTPUT ###########################
print(f"{'Saving output...':<30}", end="")
# Log model/experiment parameters
experiment_parameters = {
    'experiment_name': experiment_name,
    'description': experiment_description,
    'model_name': model_name,
    'demo_start_time': demo_start_time,
    'n_timesteps': n_timesteps,
    'random_key': random_key,
    'simulation_parameters': simulation_parameters,
}

# Create a unique identifier for each experimental run
file_timestamp = datetime.now().strftime("%Y%m%d_%H%M")

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
decoded_mean_state_correction.to_netcdf(output_folder / "mean_state_correction.nc")

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
####################################################################

print("Finished")
