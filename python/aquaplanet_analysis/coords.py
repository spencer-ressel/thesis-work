import xarray as xr
import cftime
from datetime import timedelta

from config import (
    DATA_DIRECTORY,
    LATITUDE_SOUTH,
    LATITUDE_NORTH,
    INTRASEASONAL_LOWCUT,
    INTRASEASONAL_HIGHCUT,
    LARGE_SCALE_CUTOFF,
    SMALL_SCALE_CUTOFF,
    LOWER_LEVEL_PRESSURE,
    UPPER_LEVEL_PRESSURE,
    EXPERIMENTS,
)

# ============================
# Load grid DataArrays
# ============================
latitudes = xr.load_dataarray(f"{DATA_DIRECTORY}/grid/CAM6_latitudes.nc")
longitudes = xr.load_dataarray(f"{DATA_DIRECTORY}/grid/CAM6_longitudes.nc")
pressure_levels = xr.load_dataarray(f"{DATA_DIRECTORY}/grid/CAM6_pressure_levels.nc")
times = xr.load_dataarray(f"{DATA_DIRECTORY}/grid/CAM6_times.nc")

lat_skip_list = [
    -31.263158,
    -27.473684,
    -23.684211,
    -19.894737,
    -16.105263,
    -12.315789,
    -8.526316,
    -4.736842,
    -0.947368,
    0.947368,
    4.736842,
    8.526316,
    12.315789,
    16.105263,
    19.894737,
    23.684211,
    27.473684,
    31.263158
]

# ============================
# Time bounds and missing days
# ============================
missing_days = [
    cftime.DatetimeNoLeap(7, 2, d, 0, 0, 0, 0, has_year_zero=True)
    for d in range(4, 9)
]

missing_timesteps = [
    missing_days[0] + timedelta(hours=x) for x in range(0, 24 * 3, 3)
]

START_TIME = cftime.DatetimeNoLeap(3, 1, 3, 0, 0, 0, 0, has_year_zero=True)
END_TIME = cftime.DatetimeNoLeap(13, 1, 3, 0, 0, 0, 0, has_year_zero=True)

first_half_subset_bounds = slice(START_TIME, missing_days[0] - timedelta(days=1))
second_half_subset_bounds = slice(missing_days[-1] + timedelta(days=1), END_TIME)

# ============================
# Spatial bounds
# ============================
latitude_subset_bounds = slice(LATITUDE_SOUTH, LATITUDE_NORTH)
frequency_subset_bounds = slice(INTRASEASONAL_LOWCUT, INTRASEASONAL_HIGHCUT)
wavenumber_bounds = slice(LARGE_SCALE_CUTOFF, SMALL_SCALE_CUTOFF)
pressure_subset_bounds = slice(UPPER_LEVEL_PRESSURE, LOWER_LEVEL_PRESSURE)

# ============================
# Experiment metadata as DataArrays
# ============================

experiments = xr.DataArray(
    data=EXPERIMENTS,
    dims=["experiment"],
    coords={"experiment": EXPERIMENTS},
)

latitude_bounds = xr.DataArray(
    [[-15, -15, -15], [15, 15, 15]],
    dims=["bound", "experiment"],
    coords={"bound": ["lower", "upper"], "experiment": experiments},
)

latitude_mask = (
    (latitudes >= latitude_bounds.sel(bound="lower"))
    & (latitudes <= latitude_bounds.sel(bound="upper"))
)
