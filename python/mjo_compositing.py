#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# mjo_compositing.py                                                          #
# Spencer Ressel                                                              #
# 2022.12.6                                                                   #
###############################################################################

"""
This script is a compilation of functions that serve to recreate the figures of 
two papers, Wheeler and Hendon (2004) and Jiang et al. (2020).
The Wheeler and Hendon (2004) paper describes the creation of a real-time multivariate
MJO index (RMM), and splits the MJO into 8 phases based on the RMM index. 
The Jiang et al. (2020) paper is a broad overview of the MJO,
where the specific figure (Fig. 1) recreated is a composite of MJO-associated rainfall
anomalies during boreal winter for each MJO phase

Inputs:       Global 2.5° x 2.5° resolution, daily timeseries in netCDF format
                  - precipitation from TRMM
                  - outgoing longwave radiation (OLR) data from Liebmann and Smith (1996)
                  - 850 hPa zonal wind from ERA5 reanalysis
                  - 200 hPa zonal wind from ERA5 reanalysis
Dependencies: mjo_mean_state_diagnostics.py
            
"""
#%% Imports

## Data processing tools
import numpy as np
import scipy
import scipy.signal as signal
import xarray as xr
import mjo_mean_state_diagnostics as mjo

## Plotting
# Matplotlib
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec

# Seaborn
import seaborn as sns

# Cartopy
import cartopy.crs as ccrs
import cartopy.util as cutil
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

#%% Constants
# Set time bounds
TIME_MIN = 19990101
TIME_MAX = 20181231
SAMPLING_FREQUENCY = 1

# Set latitude bounds
LAT_MIN = -25
LAT_MAX = 25

# Set central longitude
CENTRAL_LONGITUDE = 160

# Set longitude bounds
LON_MIN = -180
LON_MAX = 180

# Cut-off periods for intraseasonal filtering
INTRASEASONAL_LOWCUT = 100
INTRASEASONAL_HIGHCUT = 20

# Seconds per day
SECONDS_PER_DAY = 24 * 3600
#%% Load data
#### Load precipitation data
data_directory_precip = r"C:/Users/resse/Desktop/Data/trmm/"
file_name_precip = r"data_trmm_daily_2018.nc"
data_precipitation = xr.open_dataset(
    data_directory_precip + file_name_precip, engine="netcdf4"
)
data_precipitation = data_precipitation.rename(
    {"lat": "latitude", "lon": "longitude", "prec": "precipitation"}
)

# Assign precipitation data to variables
precipitation = data_precipitation["precipitation"]
time = data_precipitation["time"]
latitude = data_precipitation["latitude"]
longitude = data_precipitation["longitude"]

#### Load OLR data
data_directory_olr = r"C:/Users/resse/Desktop/Data/noaa_olr/"
file_name_olr = r"olr.day.mean.nc"
data_olr = xr.open_dataset(data_directory_olr + file_name_olr, engine="netcdf4")
data_olr = data_olr.rename({"lat": "latitude", "lon": "longitude"})

# Assign OLR data to variables
olr = data_olr["olr"].isel(latitude=slice(None, None, -1))

# Convert OLR time units from datetime64 to YYYYMMDD
olr["time"] = mjo.datetime64_to_yyyymmdd(olr["time"].values)
olr["time"].attrs["units"] = "YYYYMMDD"
olr["time"].attrs["long_name"] = "time"

# Assign OLR data to variables
olr = data_olr["olr"].isel(latitude=slice(None, None, -1))

# Convert OLR time units from datetime64 to YYYYMMDD
olr["time"] = mjo.datetime64_to_yyyymmdd(olr["time"].values)
olr["time"].attrs["units"] = "YYYYMMDD"
olr["time"].attrs["long_name"] = "time"

#### Load zonal wind data
data_directory_zonal_wind = r"C:/Users/resse/Desktop/Data/era5/"
file_name_zonal_wind = r"daily_2_5_degree_zonal_wind_pressure_levels_1979_2020.nc"
data_zonal_wind = xr.open_dataset(
    data_directory_zonal_wind + file_name_zonal_wind, engine="netcdf4"
)

zonal_wind = data_zonal_wind["u"].isel(latitude=slice(None, None, -1))
zonal_wind["time"] = mjo.datetime64_to_yyyymmdd(zonal_wind["time"].values)
zonal_wind_longitudes = zonal_wind["longitude"].values
zonal_wind_longitudes[zonal_wind_longitudes < 0] = (
    zonal_wind_longitudes[zonal_wind_longitudes < 0] + 360
)
zonal_wind["longitude"] = zonal_wind_longitudes
zonal_wind = zonal_wind.sortby(zonal_wind.longitude)

upper_level_zonal_wind = zonal_wind.sel(level=200)
lower_level_zonal_wind = zonal_wind.sel(level=850)

#### Load meridional wind data
data_directory_meridional_wind = r"C:/Users/resse/Desktop/Data/era5/"
file_name_meridional_wind = (
    r"daily_2_5_degree_meridional_wind_pressure_levels_1979_2020.nc"
)
data_meridional_wind = xr.open_dataset(
    data_directory_meridional_wind + file_name_meridional_wind, engine="netcdf4"
)

meridional_wind = data_meridional_wind["v"].isel(latitude=slice(None, None, -1))
meridional_wind["time"] = mjo.datetime64_to_yyyymmdd(meridional_wind["time"].values)
meridional_wind_longitudes = meridional_wind["longitude"].values
meridional_wind_longitudes[meridional_wind_longitudes < 0] = (
    meridional_wind_longitudes[meridional_wind_longitudes < 0] + 360
)
meridional_wind["longitude"] = meridional_wind_longitudes
meridional_wind = meridional_wind.sortby(meridional_wind.longitude)

upper_level_meridional_wind = meridional_wind.sel(level=200)
lower_level_meridional_wind = meridional_wind.sel(level=850)

#### Subset the data
# Specifically select the times of interest and tropical latitudes
# Precipitation
tropical_precipitation = precipitation.sel(
    time=slice(TIME_MIN, TIME_MAX), latitude=slice(LAT_MIN, LAT_MAX)
)

# OLR
tropical_olr = olr.sel(
    time=slice(TIME_MIN, TIME_MAX), latitude=slice(LAT_MIN, LAT_MAX),
)

# Lower level zonal wind - u850
tropical_lower_level_zonal_wind = lower_level_zonal_wind.sel(
    time=slice(TIME_MIN, TIME_MAX), latitude=slice(LAT_MIN, LAT_MAX),
)

# Upper level zonal wind - u200
tropical_upper_level_zonal_wind = upper_level_zonal_wind.sel(
    time=slice(TIME_MIN, TIME_MAX), latitude=slice(LAT_MIN, LAT_MAX),
)

# Lower level meridional wind - v850
tropical_lower_level_meridional_wind = lower_level_meridional_wind.sel(
    time=slice(TIME_MIN, TIME_MAX), latitude=slice(LAT_MIN, LAT_MAX),
)

# Upper level meridional wind - v200
tropical_upper_level_meridional_wind = upper_level_meridional_wind.sel(
    time=slice(TIME_MIN, TIME_MAX), latitude=slice(LAT_MIN, LAT_MAX),
)

# Time
time = time.sel(time=slice(TIME_MIN, TIME_MAX))
[year, month, day] = mjo.yyyymmdd_y_m_d(time.values)

# Latitude
latitude_tropics = latitude.sel(latitude=slice(LAT_MIN, LAT_MAX))

#%% Process data
#### Remove the annual cycle and the first three harmonics
# Precipitation
[
    tropical_precipitation_anomalies,
    tropical_precipitation_cyc,
] = mjo.remove_annual_cycle_matrix(
    tropical_precipitation, time=time, lat=latitude_tropics, lon=longitude
)

# OLR
[tropical_olr_anomalies, tropical_olr_cyc] = mjo.remove_annual_cycle_matrix(
    tropical_olr, time=time, lat=latitude_tropics, lon=longitude
)

# Lower level zonal wind
[
    tropical_lower_level_zonal_wind_anomalies,
    tropical_olr_cyc,
] = mjo.remove_annual_cycle_matrix(
    tropical_lower_level_zonal_wind, time=time, lat=latitude_tropics, lon=longitude
)

# Upper level zonal wind
[
    tropical_upper_level_zonal_wind_anomalies,
    tropical_olr_cyc,
] = mjo.remove_annual_cycle_matrix(
    tropical_upper_level_zonal_wind, time=time, lat=latitude_tropics, lon=longitude
)

# Lower level meridional wind
[
    tropical_lower_level_meridional_wind_anomalies,
    tropical_olr_cyc,
] = mjo.remove_annual_cycle_matrix(
    tropical_lower_level_meridional_wind, time=time, lat=latitude_tropics, lon=longitude
)

# Upper level meridional wind
[
    tropical_upper_level_meridional_wind_anomalies,
    tropical_olr_cyc,
] = mjo.remove_annual_cycle_matrix(
    tropical_upper_level_meridional_wind, time=time, lat=latitude_tropics, lon=longitude
)

#### Detrend the data
## Specifically linear trends in time

# Precipitation
tropical_precipitation_anomalies_detrended = signal.detrend(
    tropical_precipitation_anomalies, axis=0
)

# OLR
tropical_olr_anomalies_detrended = signal.detrend(tropical_olr_anomalies, axis=0)

# Lower level zonal wind
tropical_lower_level_zonal_wind_anomalies_detrended = signal.detrend(
    tropical_lower_level_zonal_wind_anomalies, axis=0
)

# Upper level zonal wind
tropical_upper_level_zonal_wind_anomalies_detrended = signal.detrend(
    tropical_upper_level_zonal_wind_anomalies, axis=0
)

# Lower level meridional wind
tropical_lower_level_meridional_wind_anomalies_detrended = signal.detrend(
    tropical_lower_level_meridional_wind_anomalies, axis=0
)

# Upper level meridional wind
tropical_upper_level_meridional_wind_anomalies_detrended = signal.detrend(
    tropical_upper_level_meridional_wind_anomalies, axis=0
)

#### Filter the data on intraseasonal timescales
## Using a lanczos bandpass filter for variability between 20-100 days

# Precipitation
intraseasonal_filtered_tropical_precipitation = mjo.lanczos_bandpass_filter(
    tropical_precipitation_anomalies_detrended,
    lowcut=(1 / INTRASEASONAL_LOWCUT),
    highcut=(1 / INTRASEASONAL_HIGHCUT),
    fs=SAMPLING_FREQUENCY,
    filter_axis=0,
)

# OLR
intraseasonal_filtered_tropical_olr = mjo.lanczos_bandpass_filter(
    tropical_olr_anomalies_detrended,
    lowcut=(1 / INTRASEASONAL_LOWCUT),
    highcut=(1 / INTRASEASONAL_HIGHCUT),
    fs=SAMPLING_FREQUENCY,
    filter_axis=0,
)

# Lower level zonal wind
intraseasonal_filtered_tropical_lower_level_zonal_wind = mjo.lanczos_bandpass_filter(
    tropical_lower_level_zonal_wind_anomalies_detrended,
    lowcut=(1 / INTRASEASONAL_LOWCUT),
    highcut=(1 / INTRASEASONAL_HIGHCUT),
    fs=SAMPLING_FREQUENCY,
    filter_axis=0,
)

# Upper level zonal wind
intraseasonal_filtered_tropical_upper_level_zonal_wind = mjo.lanczos_bandpass_filter(
    tropical_upper_level_zonal_wind_anomalies_detrended,
    lowcut=(1 / INTRASEASONAL_LOWCUT),
    highcut=(1 / INTRASEASONAL_HIGHCUT),
    fs=SAMPLING_FREQUENCY,
    filter_axis=0,
)

# Lower level meridional wind
intraseasonal_filtered_tropical_lower_level_meridional_wind = mjo.lanczos_bandpass_filter(
    tropical_lower_level_meridional_wind_anomalies_detrended,
    lowcut=(1 / INTRASEASONAL_LOWCUT),
    highcut=(1 / INTRASEASONAL_HIGHCUT),
    fs=SAMPLING_FREQUENCY,
    filter_axis=0,
)

# Upper level meridional wind
intraseasonal_filtered_tropical_upper_level_meridional_wind = mjo.lanczos_bandpass_filter(
    tropical_upper_level_meridional_wind_anomalies_detrended,
    lowcut=(1 / INTRASEASONAL_LOWCUT),
    highcut=(1 / INTRASEASONAL_HIGHCUT),
    fs=SAMPLING_FREQUENCY,
    filter_axis=0,
)

#### Filter the data on MJO spatial scales
## Using a lanczos bandpass filter for variability between wavenumbers 1 and 5
# mjo_filtered_tropical_precipitation = mjo.lanczos_bandpass_filter(
#     intraseasonal_filtered_tropical_precipitation,
#     lowcut=(1 / 360),
#     highcut=(5 / 360),
#     fs=(1 / 2.5),
#     filter_axis=2,
# )

# mjo_filtered_tropical_olr = mjo.lanczos_bandpass_filter(
#     intraseasonal_filtered_tropical_olr,
#     lowcut=(1 / 360),
#     highcut=(5 / 360),
#     fs=(1 / 2.5),
#     filter_axis=2,
# )

#### Meridionally average the intraseasonally-filtered quantities
## This gives a 2d array with dimensions (time x longitude)

# Precipitation
meridional_mean_intraseasonal_precipitation = np.mean(
    intraseasonal_filtered_tropical_precipitation, axis=1
)

# OLR
meridional_mean_intraseasonal_olr = np.mean(intraseasonal_filtered_tropical_olr, axis=1)

# Lower level zonal wind
meridional_mean_intraseasonal_lower_level_zonal_wind = np.mean(
    intraseasonal_filtered_tropical_lower_level_zonal_wind, axis=1
)

# Upper level zonal wind
meridional_mean_intraseasonal_upper_level_zonal_wind = np.mean(
    intraseasonal_filtered_tropical_upper_level_zonal_wind, axis=1
)

#### Normalize the data
## Necessary before computing the EOFs of the data

# OLR
mean_intraseasonal_olr = np.mean(meridional_mean_intraseasonal_olr)
std_dev_intraseasonal_olr = np.std(meridional_mean_intraseasonal_olr)
normalized_meridional_mean_intraseasonal_olr = (
    meridional_mean_intraseasonal_olr - mean_intraseasonal_olr
) / std_dev_intraseasonal_olr

# Lower level zonal wind
mean_intraseasonal_lower_level_zonal_wind = np.mean(
    meridional_mean_intraseasonal_lower_level_zonal_wind
)
std_dev_intraseasonal_lower_level_zonal_wind = np.std(
    meridional_mean_intraseasonal_lower_level_zonal_wind
)
normalized_meridional_mean_intraseasonal_lower_level_zonal_wind = (
    meridional_mean_intraseasonal_lower_level_zonal_wind
    - mean_intraseasonal_lower_level_zonal_wind
) / std_dev_intraseasonal_lower_level_zonal_wind

# Upper level zonal wind
mean_intraseasonal_upper_level_zonal_wind = np.mean(
    meridional_mean_intraseasonal_upper_level_zonal_wind
)
std_dev_intraseasonal_upper_level_zonal_wind = np.std(
    meridional_mean_intraseasonal_upper_level_zonal_wind
)
normalized_meridional_mean_intraseasonal_upper_level_zonal_wind = (
    meridional_mean_intraseasonal_upper_level_zonal_wind
    - mean_intraseasonal_upper_level_zonal_wind
) / std_dev_intraseasonal_upper_level_zonal_wind

# Concatenate the three variables into a single array
# It must have shape (structure dimension x sampling dimension)
combined_data = np.concatenate(
    [
        normalized_meridional_mean_intraseasonal_olr,
        normalized_meridional_mean_intraseasonal_lower_level_zonal_wind,
        normalized_meridional_mean_intraseasonal_upper_level_zonal_wind,
    ],
    axis=1,
)

#### Calculate EOFs and PCs
[
    mjo_EOF,
    mjo_PC,
    mjo_eigval,
    mjo_eigval_explained_var,
    mjo_eigval_err,
    mjo_dof,
    mjo_phi_0,
    mjo_phi_L,
] = mjo.eof(combined_data.T)

# Extract the EOFs of each variable from the array
olr_EOF = mjo_EOF[:, : len(longitude)]
lower_level_zonal_wind_EOF = mjo_EOF[:, len(longitude) : 2 * len(longitude)]
upper_level_zonal_wind_EOF = mjo_EOF[:, 2 * len(longitude) :]

# Convert the principal components to an RMM-like index
RMM1 = mjo_PC[0] / np.std(mjo_PC[0])
RMM2 = -mjo_PC[1] / np.std(mjo_PC[1])
mjo_strength = np.sqrt(RMM1 ** 2 + RMM2 ** 2)

# Remove weak MJO events
RMM1[mjo_strength < 1] = 0
RMM2[mjo_strength < 1] = 0

# Only examine the DJF season
# DJF_indices = np.squeeze(np.where((month < 12) & (month > 2)))
# RMM1[not_DJF_indices] = 0
# RMM2[not_DJF_indices] = 0

# Only examine boreal winter (November-March)
not_boreal_winter_indices = np.squeeze(np.where((month < 11) & (month > 3)))
RMM1[not_boreal_winter_indices] = 0
RMM2[not_boreal_winter_indices] = 0

#### Calculate phases
def compute_mjo_phase_indices(RMM1, RMM2):
    phase_indices = {}
    # Find all of the points in phase 1
    phase_indices[1] = np.squeeze(
        np.where((RMM1 < 0) & (RMM2 < 0) & (np.abs(RMM1) > np.abs(RMM2)))
    )

    # Phase 2
    phase_indices[2] = np.squeeze(
        np.where((RMM1 < 0) & (RMM2 < 0) & (np.abs(RMM1) < np.abs(RMM2)))
    )

    # Phase 3
    phase_indices[3] = np.squeeze(
        np.where((RMM1 > 0) & (RMM2 < 0) & (np.abs(RMM1) < np.abs(RMM2)))
    )

    # Phase 4
    phase_indices[4] = np.squeeze(
        np.where((RMM1 > 0) & (RMM2 < 0) & (np.abs(RMM1) > np.abs(RMM2)))
    )

    # Phase 5
    phase_indices[5] = np.squeeze(
        np.where((RMM1 > 0) & (RMM2 > 0) & (np.abs(RMM1) > np.abs(RMM2)))
    )

    # Phase 6
    phase_indices[6] = np.squeeze(
        np.where((RMM1 > 0) & (RMM2 > 0) & (np.abs(RMM1) < np.abs(RMM2)))
    )

    # Phase 7
    phase_indices[7] = np.squeeze(
        np.where((RMM1 < 0) & (RMM2 > 0) & (np.abs(RMM1) < np.abs(RMM2)))
    )

    # Phase 8
    phase_indices[8] = np.squeeze(
        np.where((RMM1 < 0) & (RMM2 > 0) & (np.abs(RMM1) > np.abs(RMM2)))
    )

    return phase_indices


phase_indices = compute_mjo_phase_indices(RMM1, RMM2)

# Average over time for each phase
intraseasonal_rainfall_by_phase = {}
intraseasonal_olr_by_phase = {}
intraseasonal_lower_level_zonal_wind_by_phase = {}
intraseasonal_lower_level_meridional_wind_by_phase = {}

for phase in phase_indices:
    # Precipitation
    intraseasonal_rainfall_by_phase[phase] = np.mean(
        intraseasonal_filtered_tropical_precipitation[phase_indices[phase]], axis=0
    )

    # OLR
    intraseasonal_olr_by_phase[phase] = np.mean(
        intraseasonal_filtered_tropical_olr[phase_indices[phase]], axis=0
    )

    # Lower level zonal wind
    intraseasonal_lower_level_zonal_wind_by_phase[phase] = np.mean(
        intraseasonal_filtered_tropical_lower_level_zonal_wind[phase_indices[phase]],
        axis=0,
    )

    # Lower level meridional wind
    intraseasonal_lower_level_meridional_wind_by_phase[phase] = np.mean(
        intraseasonal_filtered_tropical_lower_level_meridional_wind[
            phase_indices[phase]
        ],
        axis=0,
    )
#%% Plotting
#### EOFs as a function of latitude
[fig, ax] = plt.subplots(1, 2)
fig.suptitle("EOFs of Intraseasonally filtered Tropical data")
plt.rcParams.update({"font.size": 24})

# EOF 1
ax[0].plot(longitude, olr_EOF[0], color="k", lw=4, label="OLR")
ax[0].plot(
    longitude, lower_level_zonal_wind_EOF[0], ls="--", color="k", lw=4, label="u850"
)
ax[0].plot(
    longitude, upper_level_zonal_wind_EOF[0], ls=":", color="k", lw=4, label="u200"
)
ax[0].axhline(y=0, lw=2, color="k")

# Set axis labels and limits
ax[0].set_title("EOF 1")
ax[0].set_xlabel("Longitude")
ax[0].set_ylabel("Normalized Magnitude")
ax[0].set_xlim(0, 360)
ax[0].set_ylim(-0.2, 0.2)
ax[0].set_aspect(360 / 0.8)

# Specify tick parameters
lon_formatter = LongitudeFormatter()
ax[0].xaxis.set_major_formatter(lon_formatter)
ax[0].xaxis.set_major_locator(mticker.FixedLocator([0, 60, 120, 180, 240, 300, 360]))
ax[0].tick_params(which="major", width=3 * 1.00, length=3 * 5, direction="in")
ax[0].xaxis.set_minor_locator(mticker.FixedLocator([30, 90, 150, 210, 270, 330]))
ax[0].tick_params(which="minor", width=3 * 0.75, length=3 * 2.5, direction="in")

ax[0].legend(loc="upper right")

# EOF 2
ax[1].plot(longitude, -olr_EOF[1], color="k", lw=4, label="OLR")
ax[1].plot(
    longitude, lower_level_zonal_wind_EOF[1], ls="--", color="k", lw=4, label="u850"
)
ax[1].plot(
    longitude, upper_level_zonal_wind_EOF[1], ls=":", color="k", lw=4, label="u200"
)
ax[1].axhline(y=0, lw=2, color="k")

# Set axis labels and limits
ax[1].set_title("EOF 2")
ax[1].set_xlabel("Longitude")
ax[1].set_ylabel("Normalized Magnitude")
ax[1].set_xlim(0, 360)
ax[1].set_ylim(-0.2, 0.2)
ax[1].set_aspect(360 / 0.8)

# Specify tick parameters
lon_formatter = LongitudeFormatter()
ax[1].xaxis.set_major_formatter(lon_formatter)
ax[1].xaxis.set_major_locator(mticker.FixedLocator([0, 60, 120, 180, 240, 300, 360]))
ax[1].tick_params(which="major", width=3 * 1.00, length=3 * 5, direction="in")
ax[1].xaxis.set_minor_locator(mticker.FixedLocator([30, 90, 150, 210, 270, 330]))
ax[1].tick_params(which="minor", width=3 * 0.75, length=3 * 2.5, direction="in")

ax[1].legend(loc="upper right")

#### Power spectra of PCs
# Specify imporant constants
SEGMENT_LENGTH = 2000
OVERLAP = 1000
scaling_mode = "density"

frequency = {}
spectrum = {}
area = {}
scaling_factor = {}

for i in range(1, 4):
    (frequency[i], spectrum[i]) = signal.welch(
        mjo_PC[i - 1] - np.mean(mjo_PC[i - 1]),
        fs=(1 / (SECONDS_PER_DAY)),
        window="hann",
        nperseg=SEGMENT_LENGTH,
        noverlap=OVERLAP,
        scaling=scaling_mode,
    )
    area[i] = scipy.integrate.trapz(spectrum[i], x=frequency[i])
    scaling_factor[i] = (mjo_eigval_explained_var[i - 1]) / area[i]
# Smooth each of the spectra
kernel = np.array([0.25, 0.5, 0.25])  # 1-2-1 filter for smoothing
for k in range(8):
    for i in spectrum:
        spectrum[i] = np.convolve(spectrum[i], kernel, mode="same")
# Plot the power spectra
[fig, ax] = plt.subplots()
ax.set_title("Power Spectra of Principal Components")
for i in spectrum:
    ax.plot(
        SECONDS_PER_DAY * frequency[i],
        scaling_factor[i] * spectrum[i],
        label=("PC" + str(i)),
    )
# Add vertical lines at the frequencies 0.01 and 0.05
ax.axvline(x=0.01, color="black", ls="--", lw=2)
ax.axvline(x=0.05, color="black", ls="--", lw=2)

# Format x-axos
ax.set_xlabel("Frequency (CPD)")
ax.set_xscale("log")
ax.xaxis.set_major_locator(mticker.FixedLocator([10 ** -3, 10 ** -2, 10 ** -1]))
ax.tick_params(which="major", width=3 * 1.00, length=3 * 5, direction="in")
ax.xaxis.set_minor_locator(
    mticker.FixedLocator(
        [
            0.0005,
            0.0006,
            0.0007,
            0.0008,
            0.0009,
            0.001,
            0.002,
            0.003,
            0.004,
            0.005,
            0.006,
            0.007,
            0.008,
            0.009,
            0.02,
            0.03,
            0.04,
            0.05,
            0.06,
            0.07,
            0.08,
            0.09,
        ]
    )
)
ax.tick_params(which="minor", width=3 * 0.75, length=3 * 2.5, direction="in")

ax.legend(loc="best")

#### RMM data
# Configure plot
[fig, ax] = plt.subplots()
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 3
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
plt.xlabel("RMM1")
plt.ylabel("RMM2")
ax.set_facecolor("white")

# Plot index points
colormap = sns.color_palette("viridis", as_cmap=True)
start_index = 0
end_index = len(RMM1) - 1
ax.plot(RMM1[start_index], RMM2[start_index], color="black", marker=".", ls="-", ms=30)
for i in range(start_index + 1, end_index + 1):
    ax.plot(
        RMM1[i],
        RMM2[i],
        color=colormap((i - start_index) / (end_index - start_index)),
        marker="o",
        ms=10,
    )
ax.plot(
    RMM1[phase_indices[1]],
    RMM2[phase_indices[1]],
    color="red",
    marker="o",
    ms=10,
    ls="",
)
# Add phase regions overlay
circle1 = plt.Circle((0, 0), 1.0, color="k", fill=False, lw=3, zorder=10)
ax.hlines(y=0, xmin=-4, xmax=-1, color="k", lw=3, ls="-")
ax.hlines(y=0, xmin=1, xmax=4, color="k", lw=3, ls="-")
ax.vlines(x=0, ymin=-4, ymax=-1, color="k", lw=3, ls="-")
ax.vlines(x=0, ymin=1, ymax=4, color="k", lw=3, ls="-")
ax.plot([np.sqrt(2) / 2, 4], [np.sqrt(2) / 2, 4], color="k", lw=3, ls="-")
ax.plot([np.sqrt(2) / 2, 4], [-np.sqrt(2) / 2, -4], color="k", lw=3, ls="-")
ax.plot([-4, -np.sqrt(2) / 2], [4, np.sqrt(2) / 2], color="k", lw=3, ls="-")
ax.plot([-4, -np.sqrt(2) / 2], [-4, -np.sqrt(2) / 2], color="k", lw=3, ls="-")
ax.add_patch(circle1)

ax.set_aspect("equal")

#### Precipitation by MJO phase
[fig, ax] = plt.subplots(
    8,
    1,
    figsize=(32, 16),
    subplot_kw={"projection": ccrs.PlateCarree(central_longitude=CENTRAL_LONGITUDE)},
)

plt.rcParams.update({"font.size": 24})

# Specify colormap
colormap = sns.color_palette("coolwarm", as_cmap=True)

# Plot phases
for i in range(1, 9):
    vmin = np.min(intraseasonal_rainfall_by_phase[i])
    vmax = np.max(intraseasonal_rainfall_by_phase[i])
    v = np.max((np.abs(vmin), np.abs(vmax)))
    norm = colors.TwoSlopeNorm(vmin=-v, vcenter=0, vmax=v)

    im = ax[i - 1].contourf(
        longitude,
        latitude_tropics,
        intraseasonal_rainfall_by_phase[i],
        transform=ccrs.PlateCarree(),
        cmap=colormap,
        levels=21,
        norm=norm,
    )

    cdata, clon = cutil.add_cyclic_point(
        intraseasonal_rainfall_by_phase[i], coord=longitude
    )
    ax[i - 1].contourf(
        clon, latitude_tropics, cdata, transform=ccrs.PlateCarree(), cmap=colormap
    )
    ax[i - 1].set_xlabel("")

    # cbar = fig.colorbar(im, location="bottom", fraction=0.1, aspect=50, shrink=1, pad=0.1)
    # cbar.set_label(r"mm day$^{-1}$")

    # Add the land overlay
    gl = ax[i - 1].gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=2,
        color="black",
        alpha=0.0,
        linestyle="-",
    )
    if i == 8:
        gl.xlocator = mticker.FixedLocator(np.arange(LON_MIN, LON_MAX + 30, 30))
    gl.right_labels = False
    gl.top_labels = False
    ax[i - 1].coastlines()
    ax[i - 1].spines[["bottom", "top", "left", "right"]].set_color("0")
# Plot it
# fig.tight_layout()
plt.show()

#### OLR and Wind vectors by MJO phase
# [fig, ax] = plt.subplots(
#     8,
#     1,
#     figsize=(32, 16),
#     subplot_kw={"projection": ccrs.PlateCarree(central_longitude=CENTRAL_LONGITUDE)},
# )

gridspec_args = dict(
    width_ratios=[1, 0.005], hspace=0.5, left=0.01, right=0.95, bottom=0.05, top=0.95,
)
[fig, ax] = plt.subplots(
    8,
    2,
    subplot_kw={"projection": ccrs.PlateCarree(central_longitude=CENTRAL_LONGITUDE)},
    gridspec_kw=gridspec_args,
)
gs = ax[2, 1].get_gridspec()
for axes in ax[:, 1]:
    axes.remove()
cax = fig.add_subplot(gs[2:6, 1])

plt.rcParams.update({"font.size": 24})

# Specify colormap
colormap = sns.color_palette("coolwarm", as_cmap=True)

# Plot phases
for i in range(1, 9):
    im = ax[i - 1, 0].contourf(
        longitude,
        latitude_tropics,
        intraseasonal_olr_by_phase[i],
        transform=ccrs.PlateCarree(),
        cmap=colormap,
        levels=21,
    )

    cdata, clon = cutil.add_cyclic_point(intraseasonal_olr_by_phase[i], coord=longitude)
    ax[i - 1, 0].contourf(
        clon, latitude_tropics, cdata, transform=ccrs.PlateCarree(), cmap=colormap
    )

    ax[i - 1, 0].quiver(
        longitude[::2],
        latitude_tropics[::2],
        intraseasonal_lower_level_zonal_wind_by_phase[i][::2, ::2],
        intraseasonal_lower_level_meridional_wind_by_phase[i][::2, ::2],
        color="black",
        scale=100,
    )

    # # Add the land overlay
    gl = ax[i - 1, 0].gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=2,
        color="black",
        alpha=0.0,
        linestyle="-",
    )

    if i == 8:
        gl.xlocator = mticker.FixedLocator(np.arange(LON_MIN, LON_MAX + 30, 30))
    gl.right_labels = False
    gl.top_labels = False
    ax[i - 1, 0].coastlines()
    ax[i - 1, 0].spines[["bottom", "top", "left", "right"]].set_color("0")
cbar = fig.colorbar(im, cax=cax)
cbar.set_label(r"W m$^{-2}$")
# Plot it
# fig.tight_layout()
plt.show()

#%% 2D data
mean_intraseasonal_olr = np.mean(intraseasonal_filtered_tropical_olr)
std_dev_intraseasonal_olr = np.std(intraseasonal_filtered_tropical_olr)
normalized_intraseasonal_filtered_tropical_olr = (
    intraseasonal_filtered_tropical_olr - mean_intraseasonal_olr
) / std_dev_intraseasonal_olr

# Lower level zonal wind
mean_intraseasonal_lower_level_zonal_wind = np.mean(
    intraseasonal_filtered_tropical_lower_level_zonal_wind
)
std_dev_intraseasonal_lower_level_zonal_wind = np.std(
    intraseasonal_filtered_tropical_lower_level_zonal_wind
)
normalized_intraseasonal_filtered_tropical_lower_level_zonal_wind = (
    intraseasonal_filtered_tropical_lower_level_zonal_wind
    - mean_intraseasonal_lower_level_zonal_wind
) / std_dev_intraseasonal_lower_level_zonal_wind

# Upper level zonal wind
mean_intraseasonal_upper_level_zonal_wind = np.mean(
    intraseasonal_filtered_tropical_upper_level_zonal_wind
)
std_dev_intraseasonal_upper_level_zonal_wind = np.std(
    intraseasonal_filtered_tropical_upper_level_zonal_wind
)
normalized_intraseasonal_filtered_tropical_upper_level_zonal_wind = (
    intraseasonal_filtered_tropical_upper_level_zonal_wind
    - mean_intraseasonal_upper_level_zonal_wind
) / std_dev_intraseasonal_upper_level_zonal_wind

# Normalize the data
normalized_intraseasonal_filtered_tropical_olr_flattened = normalized_intraseasonal_filtered_tropical_olr.reshape(
    [len(time), len(latitude_tropics) * len(longitude)]
)
normalized_intraseasonal_filtered_tropical_lower_level_zonal_wind_flattened = normalized_intraseasonal_filtered_tropical_lower_level_zonal_wind.reshape(
    [len(time), len(latitude_tropics) * len(longitude)]
)
normalized_intraseasonal_filtered_tropical_upper_level_zonal_wind_flattened = normalized_intraseasonal_filtered_tropical_upper_level_zonal_wind.reshape(
    [len(time), len(latitude_tropics) * len(longitude)]
)

# Concatenate the variables
combined_data_flattened = np.concatenate(
    [
        normalized_intraseasonal_filtered_tropical_olr_flattened,
        normalized_intraseasonal_filtered_tropical_lower_level_zonal_wind_flattened,
        normalized_intraseasonal_filtered_tropical_upper_level_zonal_wind_flattened,
    ],
    axis=1,
)

# Compute the EOF
[
    mjo_EOF_flattened,
    mjo_PC_flattened,
    mjo_eigval_flattened,
    mjo_eigval_explained_var_flattened,
    mjo_eigval_err_flattened,
    mjo_dof_flattened,
    mjo_phi_0_flattened,
    mjo_phi_L_flattened,
] = mjo.eof(combined_data_flattened.T)

# Separate out the components of the EOFs
olr_EOF_flattened = mjo_EOF_flattened[:, : len(latitude_tropics) * len(longitude)]

lower_level_zonal_wind_EOF_flattened = mjo_EOF_flattened[
    :,
    len(latitude_tropics) * len(longitude) : 2 * len(latitude_tropics) * len(longitude),
]

upper_level_zonal_wind_EOF_flattened = mjo_EOF_flattened[
    :, 2 * len(latitude_tropics) * len(longitude) :
]

# Reshape the EOFs into 3d matrices
olr_EOF_2d = olr_EOF_flattened.reshape(
    [len(olr_EOF_flattened), len(latitude_tropics), len(longitude)]
)
lower_level_zonal_wind_EOF_2d = lower_level_zonal_wind_EOF_flattened.reshape(
    [len(lower_level_zonal_wind_EOF_flattened), len(latitude_tropics), len(longitude)]
)
upper_level_zonal_wind_EOF_2d = upper_level_zonal_wind_EOF_flattened.reshape(
    [len(upper_level_zonal_wind_EOF_flattened), len(latitude_tropics), len(longitude)]
)

# Convert the principal components to an RMM-like index
RMM1_2d = mjo_PC_flattened[0] / np.std(mjo_PC_flattened[0])
RMM2_2d = mjo_PC_flattened[1] / np.std(mjo_PC_flattened[1])
mjo_strength = np.sqrt(RMM1_2d ** 2 + RMM2_2d ** 2)

# Remove weak MJO events
RMM1_2d[mjo_strength < 1] = 0
RMM2_2d[mjo_strength < 1] = 0

# Only examine the DJF season
# DJF_indices = np.squeeze(np.where((month < 12) & (month > 2)))
# RMM1_2d[not_DJF_indices] = 0
# RMM2_2d[not_DJF_indices] = 0

# Only examine boreal winter (November-March)
not_boreal_winter_indices = np.squeeze(np.where((month < 11) & (month > 3)))
RMM1_2d[not_boreal_winter_indices] = 0
RMM2_2d[not_boreal_winter_indices] = 0

phase_indices_2d = compute_mjo_phase_indices(RMM1_2d, RMM2_2d)

# Average over time for each phase
intraseasonal_rainfall_by_phase_2d = {}
for phase in phase_indices_2d:
    intraseasonal_rainfall_by_phase_2d[phase] = np.mean(
        intraseasonal_filtered_tropical_precipitation[phase_indices_2d[phase]], axis=0
    )
#### Plot first EOF of OLR and u850

[fig, ax] = plt.subplots(
    1,
    figsize=(32, 16),
    subplot_kw={"projection": ccrs.PlateCarree(central_longitude=CENTRAL_LONGITUDE)},
)
plt.rcParams.update({"font.size": 24})
ax.spines[["bottom", "top", "left", "right"]].set_color("0")
ax.set_title("First EOF of MJO-associated OLR and u850")

# Specify colormap
colormap = sns.color_palette("coolwarm", as_cmap=True)

# Plot the first EOF as a function of latitude and longitude
im = ax.contourf(
    longitude,
    latitude_tropics,
    olr_EOF_2d[0],
    transform=ccrs.PlateCarree(),
    cmap=colormap,
    levels=21,
)

cdata, clon = cutil.add_cyclic_point(olr_EOF_2d[0], coord=longitude)
ax.contourf(clon, latitude_tropics, cdata, transform=ccrs.PlateCarree(), cmap=colormap)

cbar = fig.colorbar(im, location="bottom", fraction=0.1, aspect=50, shrink=1, pad=0.1)

im = ax.contour(
    longitude,
    latitude_tropics,
    lower_level_zonal_wind_EOF_2d[0],
    transform=ccrs.PlateCarree(),
    colors="black",
    levels=8,
)

cdata, clon = cutil.add_cyclic_point(lower_level_zonal_wind_EOF_2d[0], coord=longitude)
ax.contour(clon, latitude_tropics, cdata, transform=ccrs.PlateCarree(), colors="black")

# Add the land overlay
gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=2,
    color="black",
    alpha=0.0,
    linestyle="-",
)
gl.xlocator = mticker.FixedLocator(np.arange(LON_MIN, LON_MAX + 30, 30))
gl.right_labels = False
gl.top_labels = False
ax.coastlines()

# Plot it
plt.show()

#%% Precipitation by season

#### Calculate precip
[year, month, day] = mjo.yyyymmdd_y_m_d(time.values)

# DJF
DJF_indices = np.squeeze(np.where((month == 12) | (month == 1) | (month == 2)))
DJF_tropical_precipitation = tropical_precipitation[DJF_indices]
DJF_mean_tropical_precipitation = DJF_tropical_precipitation.mean(dim="time")

JJA_indices = np.squeeze(np.where((month == 6) | (month == 7) | (month == 8)))
JJA_tropical_precipitation = tropical_precipitation[JJA_indices]
JJA_mean_tropical_precipitation = JJA_tropical_precipitation.mean(dim="time")

#### Set gridspec
fig = plt.figure(figsize=(32, 16))
gs = GridSpec(
    2, 2, width_ratios=(100, 1.5), wspace=0.03, height_ratios=(1, 1), hspace=0.0025
)

plt.suptitle("Tropical Precipitation by season")
ax = fig.add_subplot(
    gs[0, 0], projection=ccrs.PlateCarree(central_longitude=CENTRAL_LONGITUDE)
)

# Specify colormap
colormap = sns.color_palette("Blues", as_cmap=True)

#### DJF
ax.set_title("DJF")
im0 = ax.contourf(
    longitude,
    latitude_tropics,
    DJF_mean_tropical_precipitation,
    transform=ccrs.PlateCarree(),
    cmap=colormap,
    levels=20,
)

# Add cyclic point to data
cdata, clon = cutil.add_cyclic_point(DJF_mean_tropical_precipitation, coord=longitude)
ax.contourf(clon, latitude_tropics, cdata, transform=ccrs.PlateCarree(), cmap=colormap)

# Add the land overlay
# ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k')
gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    # xlocs=np.arange(-180,360,30),
    linewidth=2,
    color="black",
    alpha=0.5,
    linestyle="-",
)
gl.xlocator = mticker.FixedLocator(np.arange(LON_MIN, LON_MAX + 30, 30))
gl.right_labels = False
gl.top_labels = False
ax.coastlines()

#### JJA
ax = fig.add_subplot(
    gs[1, 0], projection=ccrs.PlateCarree(central_longitude=CENTRAL_LONGITUDE)
)

im1 = ax.contourf(
    longitude,
    latitude_tropics,
    JJA_mean_tropical_precipitation,
    transform=ccrs.PlateCarree(),
    cmap=colormap,
    levels=20,
)

cdata, clon = cutil.add_cyclic_point(JJA_mean_tropical_precipitation, coord=longitude)
ax.contourf(clon, latitude_tropics, cdata, transform=ccrs.PlateCarree(), cmap=colormap)
ax.set_title("JJA")

# Add the land overlay
# ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k')
gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    # xlocs=np.arange(-180,360,30),
    linewidth=2,
    color="black",
    alpha=0.5,
    linestyle="-",
)

gl.xlocator = mticker.FixedLocator(np.arange(LON_MIN, LON_MAX + 30, 30))
gl.right_labels = False
gl.top_labels = False
ax.coastlines()

# Set colorbar
cbar_ax = fig.add_subplot(gs[:, 1])
cbar = fig.colorbar(im0, cax=cbar_ax, fraction=0.1, aspect=50, shrink=1, pad=0.1)
cbar.set_label(r"mm day$^{-1}$")
# Plot it
# fig.tight_layout()
plt.show()
