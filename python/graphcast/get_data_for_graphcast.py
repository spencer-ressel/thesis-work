# ------------------------------------------------------------
# Standard library imports
# ------------------------------------------------------------
import gc
import glob
import logging
import os
import sys
from datetime import datetime

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
from dateutil.relativedelta import relativedelta

# ------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("data_download_logs/data_download.log", mode="a"),
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

logger.info("Starting data download script")
def surface_level_preprocess(ds: xr.Dataset) -> xr.Dataset:
    # Subset to a specific region and keep only one variable
    subset_data = ds.sel(
        # level=pressure_levels,
        time=ds.time.where(ds['time'].dt.hour.isin([0, 6, 12, 18]), drop=True)
    )

    target_grid = xr.Dataset(
        {
            "lat": (["lat"], np.arange(-90, 91, 1.0)),
            "lon": (["lon"], np.arange(0, 360, 1.0)),
        }
    )
    regridder = xe.Regridder(subset_data, target_grid, "bilinear", reuse_weights=False) 

    return regridder(subset_data) # type: ignore

def pressure_level_preprocess(ds: xr.Dataset) -> xr.Dataset:
    # Subset to a specific region and keep only one variable
    subset_data = ds.sel(
        level=pressure_levels,
        time=ds.time.where(ds['time'].dt.hour.isin([0, 6, 12, 18]), drop=True)
    ).assign_coords({'level': pressure_levels.astype(np.int32)})

    target_grid = xr.Dataset(
        {
            "lat": (["lat"], np.arange(-90, 91, 1.0)),
            "lon": (["lon"], np.arange(0, 360, 1.0)),
        }
    )
    regridder = xe.Regridder(subset_data, target_grid, "bilinear", reuse_weights=False) 

    return regridder(subset_data) # type: ignore

yyyymm_strings = pd.date_range(
    pd.to_datetime(start_date).to_period("M").to_timestamp(),
    pd.to_datetime(end_date).to_period("M").to_timestamp(),
    freq="MS"
).strftime("%Y-%m")

logger.info(f"Dates: {np.datetime64(start_date).astype('datetime64[h]')} : {np.datetime64(end_date).astype('datetime64[h]')}")

graphcast_data_directory = f"/glade/u/home/sressel/spencer-scratch/graphcast_input_data/{string_to_yyyymm(start_date)}_{string_to_yyyymm(end_date)}"
if not os.path.exists(graphcast_data_directory):
    logger.info("Creating output directory...")
    os.makedirs(graphcast_data_directory, exist_ok=True)
logger.info(f"Graphcast data directory: {graphcast_data_directory}")

years, months = extract_years_months(start_date, end_date)

# Surface level variables
logger.info("Surface level variables")
surface_base = "/gdex/data/d633000/e5.oper.an.sfc"

surface_variables = {
    "2m_temperature": "2t",
    "mean_sea_level_pressure": "msl",
    "10m_u_component_of_wind": "10u",
    "10m_v_component_of_wind": "10v"
}

surface_variables_old_names = {
    "2m_temperature": "VAR_2T",
    "mean_sea_level_pressure": "MSL",
    "10m_u_component_of_wind": "VAR_10U",
    "10m_v_component_of_wind": "VAR_10V"
}

for variable in surface_variables.keys():
    files_list = []

    logger.info(f"  {variable}")
    for ym in yyyymm_strings:
        pattern = f"{surface_base}/{ym}/e5.oper.an.sfc.*_{surface_variables[variable]}.*.nc"
        files_list.extend(glob.glob(pattern))

    if not files_list:
        print(f"No files found for variable {variable} in month {ym}")
    else:
        logger.info(f"    Loading files...")
        surface_data = convert_time_to_ns(xr.open_mfdataset(sorted(files_list), preprocess=surface_level_preprocess).load())
        surface_data = surface_data.rename({surface_variables_old_names[variable]: variable})

    # logger.info(surface_data)
    logger.info(f"    Output directory: {graphcast_data_directory}")
    logger.info(f"    Saving data...")
    surface_data.to_netcdf(f"{graphcast_data_directory}/{variable}.nc")
    surface_data.close()
    del surface_data
    gc.collect()

logger.info("Finished")

# Pressure levle variables
logger.info("Pressure Level Data")
pressure_level_base = "/gdex/data/d633000/e5.oper.an.pl"

pressure_levels = xr.DataArray(
    data = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
    dims=['level'],
    coords={'level': [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]}
)

pressure_level_variables = {
    "geopotential": "z",
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "specific_humidity": "q",
    "vertical_velocity": "w",
}

pressure_level_variables_old_names = {
    "geopotential": "Z",
    "temperature": "T",
    "u_component_of_wind": "U",
    "v_component_of_wind": "V",
    "specific_humidity": "Q",
    "vertical_velocity": "W",
}

for variable in pressure_level_variables.keys():
    files_list = []

    logger.info(f"  {variable}")
    for ym in yyyymm_strings:
        pattern = f"{pressure_level_base}/{ym}/e5.oper.an.pl.*_{pressure_level_variables[variable]}.*.nc"
        files_list.extend(glob.glob(pattern))

    if not files_list:
        logger.info(f"No files found for variable {variable} in month {ym}")
    else:
        logger.info(f"    Loading files...")
        pressure_level_data = convert_time_to_ns(xr.open_mfdataset(sorted(files_list), preprocess=pressure_level_preprocess).load())
        pressure_level_data = pressure_level_data.rename({pressure_level_variables_old_names[variable]: variable})

    # logger.info(pressure_level_data)
    logger.info(f"    Output directory: {graphcast_data_directory}")
    logger.info(f"    Saving data...")
    datetimes = pressure_level_data.datetime
    pressure_level_data.to_netcdf(f"{graphcast_data_directory}/{variable}.nc")
    pressure_level_data.close()
    del pressure_level_data
    gc.collect()

logger.info("Finished")

# Precipitation
logger.info("Precipitation")
target = f"{graphcast_data_directory}/total_precipitation_6hr.nc"
dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "total_precipitation",
    ],
    "year": years,
    "month": months,
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "time": [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
    ],
    "data_format": "netcdf",
    "download_format": "unarchived"
}

client = cdsapi.Client()
logger.info("    Downloading data...")
client.retrieve(dataset, request, target)

logger.info("    Regridding data...")
raw_precipitation_files = sorted(glob.glob(f"{graphcast_data_directory}/total_precipitation_6hr.nc"))
precipitation_data = xr.open_mfdataset(raw_precipitation_files)['tp'].rename({'valid_time': 'time'}).load()

six_hour_accumulated_precipitation = precipitation_data.resample(time='6h').sum()

target_grid = xr.Dataset(
        {
            "lat": (["lat"], np.arange(-90, 91, 1.0)),
            "lon": (["lon"], np.arange(0, 360, 1.0)),
        }
    )
regridder = xe.Regridder(six_hour_accumulated_precipitation, target_grid, "bilinear", reuse_weights=False)
regridded_precipitation = regridder(six_hour_accumulated_precipitation)
regridded_precipitation = convert_time_to_ns(regridded_precipitation.sel(time=datetimes.sel(batch=0).values))

regridded_precipitation.name = 'total_precipitation_6hr'
resave_data = True
if resave_data:
    logger.info(f"    Output directory: {graphcast_data_directory}")
    logger.info(f"    Saving data...")
    regridded_precipitation.to_netcdf(f"{graphcast_data_directory}/total_precipitation_6hr.nc", mode='w')
    regridded_precipitation.close()
    del regridded_precipitation
    gc.collect()

logger.info("Finished")

# TOA Solar Radiation
logger.info("TOA Solar Radiation")
target = f"{graphcast_data_directory}/toa_incident_solar_radiation.nc"
dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "toa_incident_solar_radiation"
    ],
    "year": years,
    "month": months,
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "time": [
        "00:00", "06:00","12:00","18:00"
    ],
    "data_format": "netcdf",
    "download_format": "unarchived"
}

client = cdsapi.Client()
logger.info("    Downloading data...")
client.retrieve(dataset, request, target)

logger.info("    Regridding data...")
toa_incident_solar_radiation_data = xr.open_mfdataset(target)['tisr'].rename({'valid_time':'time'}).load()

target_grid = xr.Dataset(
        {
            "lat": (["lat"], np.arange(-90, 91, 1.0)),
            "lon": (["lon"], np.arange(0, 360, 1.0)),
        }
    )
regridder = xe.Regridder(toa_incident_solar_radiation_data, target_grid, "bilinear", reuse_weights=False)
regridded_toa_incident_solar_radiation = regridder(toa_incident_solar_radiation_data)
regridded_toa_incident_solar_radiation = convert_time_to_ns(regridded_toa_incident_solar_radiation.sel(time=datetimes.sel(batch=0).values))
regridded_toa_incident_solar_radiation.name = 'toa_incident_solar_radiation'

resave_data = True
if resave_data:
    logger.info(f"    Output directory: {graphcast_data_directory}")
    logger.info(f"    Saving data...")
    regridded_toa_incident_solar_radiation.to_netcdf(f"{graphcast_data_directory}/toa_incident_solar_radiation.nc", mode='w')
    regridded_toa_incident_solar_radiation.close()
    del regridded_toa_incident_solar_radiation
    gc.collect()

logger.info("Geopotential at Surface")
os.system(
    f"cp /glade/u/home/sressel/spencer-scratch/geopotential_at_surface.nc {graphcast_data_directory}/geopotential_at_surface.nc"
)

logger.info("Land Sea Mask")
os.system(
    f"cp /glade/u/home/sressel/spencer-scratch/land_sea_mask.nc {graphcast_data_directory}/land_sea_mask.nc"
)

logger.info("Load all data into a single dataset")
all_data = xr.open_mfdataset(f"{graphcast_data_directory}/*.nc")
logger.info(f"    Saving data...")
all_data.to_netcdf(f"{graphcast_data_directory}/era5_data.nc")

logger.info("All Finished")