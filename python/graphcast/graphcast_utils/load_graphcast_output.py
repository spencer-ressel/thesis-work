import config
from auxiliary_functions.time_utils import convert_ns_to_datetime
import xarray as xr

def load_predictions(prediction_initial_date, prediction_timesteps, prediction_type, run_notes, vars_to_load, decode_time=True):
    vars_to_drop = [var for var in config.PREDICTED_VARIABLES if var not in vars_to_load]

    prediction_filepath = (
        f"{config.GRAPHCAST_DATA_DIRECTORY}/{prediction_type}_model_forecasts/{prediction_initial_date}/{prediction_initial_date}_{prediction_timesteps}{run_notes}.zarr"
    )

    predictions = xr.open_zarr(prediction_filepath, drop_variables=vars_to_drop).load().sel(batch=0, drop=True)

    if decode_time:
        datetimes = convert_ns_to_datetime(predictions.time, prediction_initial_date)
        predictions = predictions.assign_coords({'time':datetimes.datetime}).drop_vars(["datetime"])

    return predictions