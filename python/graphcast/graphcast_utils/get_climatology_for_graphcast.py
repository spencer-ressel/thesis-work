# ------------------------------------------------------------
# Standard library imports
# ------------------------------------------------------------
import gc
import glob
import logging
import os
import sys
from datetime import datetime
import shutil
import subprocess
from functools import partial

# ------------------------------------------------------------
# Third‑party scientific stack
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
from auxiliary_functions.time_utils import datetime64_to_yyyymmdd, string_to_yyyymm, convert_time_to_ns, convert_ns_to_datetime, extract_years_months

# ------------------------------------------------------------
# External APIs / data access
# ------------------------------------------------------------
import cdsapi

# ------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------
pbs_job_id = os.environ.get('PBS_JOBID')

if pbs_job_id:
    # PBS_JOBID often looks like '123456.desktop1' — strip the host part if you just want the number
    job_id_short = pbs_job_id.split('.')[0]
    log_filename = f"data_download_logs/data_download_{job_id_short}.log"
else:
    log_filename = "data_download_logs/data_download.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename, mode="a"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# Log uncaught exceptions to file
def log_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        # Let Ctrl+C behave normally
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = log_exception

start_date = sys.argv[1]
end_date = sys.argv[2]
logger.info(f"Start date: {start_date}")
logger.info(f"End date: {end_date}")

datetime_values = np.arange(
    np.datetime64(start_date),
    np.datetime64(end_date) + np.timedelta64(1, 'D'),
    np.timedelta64(6, 'h')
)
time_ns = np.arange(
    0,
    len(datetime_values) * 6 * 3600 * 10**9,
    6 * 3600 * 10**9,
    dtype='timedelta64[ns]'
)
datetime_array = xr.DataArray(
    data=datetime_values,
    dims=['time'],
    coords={'time':time_ns}
)
datetime_array = datetime_array.assign_coords(
    datetime=("time", datetime_values),   # secondary coordinate
    time=("time", time_ns)                # replace primary coordinate
)
datetime_array_with_batch = datetime_array.expand_dims('batch')

logger.info("Starting data download script")

def preprocess(ds: xr.Dataset, pressure_levels: np.ndarray | None = None) -> xr.Dataset:
    sel_kwargs = {"time": ds.time.where(ds['time'].dt.hour.isin([0, 6, 12, 18]), drop=True)}

    if pressure_levels is not None:
        sel_kwargs["level"] = pressure_levels

    subset_data = ds.sel(**sel_kwargs)

    if pressure_levels is not None:
        subset_data = subset_data.assign_coords({'level': pressure_levels.astype(np.int32)})

    target_grid = xr.Dataset(
        {
            "lat": (["lat"], np.arange(-90, 91, 1.0)),
            "lon": (["lon"], np.arange(0, 360, 1.0)),
        }
    )
    regridder = xe.Regridder(subset_data, target_grid, "bilinear", reuse_weights=False)

    return regridder(subset_data)  # type: ignore

yyyymm_strings = pd.date_range(
    pd.to_datetime(start_date).to_period("M").to_timestamp(),
    pd.to_datetime(end_date).to_period("M").to_timestamp(),
    freq="MS"
).strftime("%Y%m")

logger.info(f"Dates: {np.datetime64(start_date).astype('datetime64[h]')} : {np.datetime64(end_date).astype('datetime64[h]')}")

graphcast_data_directory = f"/glade/u/home/sressel/spencer-scratch/graphcast_input_data/6hr_climatology"
if not os.path.exists(graphcast_data_directory):
    logger.info("Creating output directory...")
    os.makedirs(graphcast_data_directory, exist_ok=True)
logger.info(f"Graphcast data directory: {graphcast_data_directory}")

years, months = extract_years_months(start_date, end_date)

for year in years:
    if not os.path.exists(f"{graphcast_data_directory}/{year}"):
        os.makedirs(f"{graphcast_data_directory}/{year}", exist_ok=True)

# Pressure levle variables
pressure_level_base = "/gdex/data/d633000/e5.oper.an.pl"

pressure_levels = xr.DataArray(
    data = [200, 850],
    dims=['level'],
    coords={'level': [200, 850]}
)

pressure_level_variables = {
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
}

pressure_level_variables_old_names = {
    "u_component_of_wind": "U",
    "v_component_of_wind": "V",
}

for variable in pressure_level_variables.keys():

    files_list = []

    logger.info(f"-- {variable}")

    for year in years:
        logger.info(f"---- {year}")
        
        if os.path.isfile(f"{graphcast_data_directory}/{year}/{variable}.nc"):
            logger.info(f"-- {variable} exists for {year}, skipping...")
            continue

        pattern = f"{pressure_level_base}/{year}*/e5.oper.an.pl.*_{pressure_level_variables[variable]}.*.nc"
        files_list.extend(glob.glob(pattern))

        if not files_list:
            logger.info(f"---- No files found for variable {variable} in month {ym}")
            continue

        else:
            logger.info(f"---- Loading files...")

            # Load and concatenate the data, preprocess to subset to 6-hourly, regrid
            if variable == 'v_component_of_wind':
                pressure_levels = pressure_levels.sel(level=200)
            pressure_level_data = xr.open_mfdataset(sorted(files_list), preprocess=partial(preprocess, pressure_levels=pressure_levels)).load()
            pressure_level_data = pressure_level_data.rename({pressure_level_variables_old_names[variable]: variable})

            # Add a batch dimension to the data
            pressure_level_data_with_batch = pressure_level_data.expand_dims('batch')

            # Re-write the time coordinates to have time in [ns] and a datetime with batch in datetime64[ns]
            pressure_level_data_retimed = pressure_level_data_with_batch.assign_coords(
                datetime=datetime_array_with_batch,
                time=("time", time_ns)
            )

        # logger.info(pressure_level_data)
        logger.info(f"---- Output directory: {graphcast_data_directory}")
        logger.info(f"---- Saving data...")
        pressure_level_data_retimed.to_netcdf(f"{graphcast_data_directory}/{year}/{variable}.nc")
        pressure_level_data_retimed.close()
        del pressure_level_data_retimed
        gc.collect()

logger.info("All Finished")