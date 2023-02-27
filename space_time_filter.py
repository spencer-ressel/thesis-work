#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# space_time_filter.py                                                        #                                                #
# Spencer Ressel                                                              #
# 2022.06.24                                                                  #
###############################################################################

"""
This script takes in a 3D NOAA OLR or TRMM precipitation dataset and filters
it in space and time according to the specified bounds. The signal is also 
proccesed to remove the annual cycle, detrended, and meridionally averaged. 
By default, the signal will be filtered for MJO-like features
using a Butterworth filter, but any frequency cut-off is possible. 

Inputs:       Global resolution, daily timeseries in netCDF format
              - Current datasets include NOAA OLR (Liebmann and Smith 1996) and 
                TRMM daily precipitation data
Outputs:      The filtered signal 
Figures:      None
Dependencies: mjo_mean_state_diagnostics.py
              process_signal.py
              mask_land.py
            
"""
#%% Imports
import mjo_mean_state_diagnostics as mjo
import mask_land as ml
import numpy as np
import numpy.matlib
from process_signal import process_signal
from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors
import csv
import os

#%% Function
def space_time_filter(
    dataset,
    time_limits,
    max_latitude,
    temporal_resolution,
    spatial_resolution,
    temporal_cutoffs=(1 / 100, 1 / 20),
    spatial_cutoffs=(1 / 360, 5 / 360),
    filter_type="Butterworth",
    filter_order=4,
    mask_land=False,
):
    """
    This function takes in a 3D signal and returns a meridionally averaged,
    space-time filtered version of the signal. The default filtering 
    corresponds to MJO scales with land masking off, and is performed by a 
    butterworth bandpass filter

    Parameters
    ----------
    dataset : string
        The type of data to be used. Current options are 'TRMM' precipitation 
        or 'NOAA' Outgoing Longwave Radiation (OLR).
    time_limits : tuple
        The time span of interest for the signal, specified in YYYYMMDD format.
        For TRMM, the limits are 19980101 - 20181213. 
        For NOAA, the limits are 19740601 - 20210908.
    max_latitude : int
        The maximum latitude of interest, given as a positive number 
        between 0-90. 
    temporal_resolution : int
        The resolution of the data in time. Daily data has a period of 1 day 
        per sample, for example. 
    spatial_resolution : int
        The resolution of the data in space, in degrees of longitude. Both TRMM
        and NOAA OLR data are 2.5 degrees per sample. 
    temporal_cutoffs : tuple, optional
        The low frequency and high frequency temporal limits to be 
        filtered for, respectively. Any signal outside the limits will be 
        attenuated. The default is (1/100, 1/20).
    spatial_cutoffs : tuple, optional
        The low frequency and high frequency spatial limits to be 
        filtered for, respectively. Any signal outside the limits will be 
        attenuated. The default is (1/360, 5/360).
    filter_type : string. optional
        The type of filter to use in time. The default is 'Butterworth'.
    filter_order : int, optional
        The order of the filter. The default is 4, which corresponds to a 
        Butterworth filter. For a Lanczos filter, 101 is appropriate. 
    mask_land : bool, optional
        Whether or not to mask the land, more important for TRMM data.
        The default is False.

    Returns
    -------
    time : numpy.ndarray
        An array containing the values of the time array corresponding to the 
        processed data
    lat_tropics : numpy.ndarray
        An array containing the values of the latitude array corresponding to 
        the processed data
    lon : numpy.ndarray
        An array containing the values of the longitude array corresponding to
        the processed data
    filtered_signal : numpy.ndarray
        A 2D array of space-time filtered data, with dimensions of time and 
        longitude.
    meridionally_averaged_signal : numpy.ndarray
        A 2D array of meridionally averaged data, with dimensions of time and 
        longitude. The data is not filtered in any way. 

    """
    #### Load data
    print("======================")
    print("Loading Data")
    print(" → Dataset: " + dataset)
    print("======================")

    # Specify data location on resse/ or on the server
    if os.getcwd()[0] == "C":
        dir_in = "C:/Users/resse/Desktop/Data/"
    else:
        dir_in = "/home/disk/eos7/sressel/data/"
    if dataset == "NOAA":
        nan_value = 9.969209968386869e36
        nan_big_small = np.array([1])

        # Load in OLR data
        data = Dataset(dir_in + "noaa_olr/olr.day.mean.nc", "r", format="NETCDF4")

        # OLR has lat from 90~-90, must be reversed
        olr = np.flip(data["olr"][:], 1)
        lat = data.variables["lat"][:]
        lat = np.flip(lat, 0)
        lon = data.variables["lon"][:]
        input_signal = olr

        # OLR has data in hours from 1800, want YYYYMMDD format
        t_unit = data.variables["time"].units
        data_time = data.variables["time"][:]
        time_as_datetime = num2date(data_time, units=t_unit)
        time = np.array([int(i.strftime("%Y%m%d")) for i in time_as_datetime])
    elif dataset == "TRMM":
        nan_value = np.array([-9999.8])
        nan_big_small = np.array([0])

        # Load in TRMM data
        data = Dataset(dir_in + "trmm/data_trmm_daily_2018.nc", "r", format="NETCDF4")

        # Time has format YYYYMMDD
        time = data.variables["time"][:]
        lat = data.variables["lat"][:]
        lon = data.variables["lon"][:]
        input_signal = data.variables["prec"]
    else:
        nan_value = np.array([10 * 14])  # Change if diff data
        nan_big_small = np.array([1])  # Change if diff data
    # Process the data by trimming the time and latitude dimensions, as well as
    # removing the annual mean, first three harmonics, and any trends
    time, lat_tropics, lon, processed_signal = process_signal(
        time,
        lat,
        lon,
        input_signal,
        time_limits,
        max_latitude,
        nan_value,
        nan_big_small,
    )

    # Convert the units of the data, if necessary
    if (dataset != "NOAA") and (dataset != "TRMM"):  # original unit is kg/m^2/s
        processed_signal *= 86400
    #### Mask land data
    if mask_land == True:
        print("======================")
        processed_signal = ml.mask_land(
            processed_signal, max_latitude, 1 / spatial_resolution
        )
    # Perform a latitude-weighted average of the data in the
    # meridional direction
    meridionally_averaged_signal = np.average(
        processed_signal, axis=1, weights=np.cos(np.deg2rad(lat_tropics))
    )

    #### Filter Data
    print("Filtering Data")
    print(" → Filter: " + filter_type)
    print("======================")
    # Filter the data in space and time. The spatial filtering is performed
    # using FFT, while the temporal filtering can be performed using either a
    # Butterworth filter or a Lanczos filter
    if filter_type == "Butterworth":
        filtered_signal = mjo.butter_bandpass_filter(
            mjo.fft_bandpass_filter(
                meridionally_averaged_signal,
                spatial_cutoffs[0],
                spatial_cutoffs[1],
                1 / spatial_resolution,
            ).T,
            temporal_cutoffs[0],
            temporal_cutoffs[1],
            1 / temporal_resolution,
            order=filter_order,
        ).T
    elif filter_type == "Lanczos":
        filtered_signal = mjo.lanczos_bandpass_filter(
            mjo.fft_bandpass_filter(
                meridionally_averaged_signal,
                spatial_cutoffs[0],
                spatial_cutoffs[1],
                1 / spatial_resolution,
            ).T,
            temporal_cutoffs[0],
            temporal_cutoffs[1],
            1 / temporal_resolution,
            order=filter_order,
        ).T
    return time, lat_tropics, lon, filtered_signal, meridionally_averaged_signal


#%% Plot MJO Filtered Signal

# Parameters
mask_land = False
save_plots = False
plot_overlay = "intraseasonal"

# Times to sample
# tmin = 19980101 #YYYYMMDD
# tmax = 20160101
tmin = 19790601
tmax = 20110101
trange = str(tmin)[:4] + "_" + str(tmax)[:4]
time_limits = (tmin, tmax)

# Latitude range
max_latitude = 15  # max latitude for pr spectrum (WK1999)

# Data resolution in longitude and time (works for TRMM and NOAA OLR)
temporal_resolution = 1  # temporal resolution is 1 day (daily data)
spatial_resolution = 2.5  # longitudinal resolution is 2.5 deg

# Specify the low and high cutoffs in time and space, respectively
temporal_cutoffs = (1 / 100, 1 / 20)  # 20-100 day period
spatial_cutoffs = (1 / 360, 5 / 360)  # Zonal Wavenumbers 1-5
filter_type = "Butterworth"
if filter_type == "Butterworth":
    filter_order = 4
elif filter_type == "Lanczos":
    filter_order = 101
#### Run Code
dataset = "NOAA"
(
    time,
    lat_tropics,
    lon,
    mjo_filtered_signal,
    meridionally_averaged_signal,
) = space_time_filter(
    dataset,
    time_limits,
    max_latitude,
    temporal_resolution,
    spatial_resolution,
    temporal_cutoffs,
    spatial_cutoffs,
    filter_type=filter_type,
    filter_order=filter_order,
    mask_land=mask_land,
)

# Day at lag 0
center_day = 19930115  # YYYYMMDD
center_index = np.where(time == center_day)[0][0]

# Choose the number of days to plot
lag = 50
lag_days = np.arange(-lag, lag + 1, 1)

# Convert from YYYYMMDD format to MM/DD/YYYY format
date = [
    datetime.strptime(str(time[i]), "%Y%m%d").strftime("%m/%d/%Y")
    for i in range(len(time))
]

#### Plot Results
print("Plotting Filtered Data")
print("======================")
plt.figure(figsize=(18, 12))
# Automatically create the colormap based on the extrema of the data
def myround(x, base=5):
    return base * round(x / base)


# Find the absolute value of the data furthest from zero and round it up to the
# next multiple of 5 (e.g. [-5.32, 3.75] -> 10)
vmin = np.fix(np.min(mjo_filtered_signal[center_index - 50 : center_index + 51, :97]))
vmax = np.fix(np.max(mjo_filtered_signal[center_index - 50 : center_index + 51, :97]))
edge_value = myround(np.max([np.abs(vmin), np.abs(vmax)]) + 2.5)

# If that value is less than 15, create levels from -edge_value to edge_value
# in steps of 2.5, otherwise creates the levels in steps of 5
# if edge_value <= 15:
#     levels = np.arange( -edge_value, edge_value+2.5, 2.5)
# else:
#     levels = np.arange( -edge_value, edge_value+5, 5)

# Import the coolwarm palette from Seaborn and modify it's zero point to
# correspond to white instead of gray
# cmap = sns.color_palette('coolwarm', n_colors=15)
# cmap[7] = (1,1,1)
# cmap.insert(7, (1,1,1))
# cmap = LinearSegmentedColormap.from_list('coolwarm_mod', cmap, N=16)

# Adames and Kim 2016 Colormap
cbar_dir = r"C:\Users\resse\OneDrive\UW\Research\Color Maps/"
with open(cbar_dir + "AK2016_geopotential_colorbar.csv", newline="") as csvfile:
    reader = csv.reader(csvfile, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
    color_list = list(reader)
cmap = colors.ListedColormap(color_list, name="AK2016")

levels = np.linspace(-edge_value, edge_value, len(color_list) + 1)

# Plot the data as filled contours
CF = plt.contourf(
    lon,
    lag_days,
    mjo_filtered_signal[center_index - 50 : center_index + 51],
    cmap=cmap,
    levels=levels,
    norm=colors.CenteredNorm(),
)
cbar = plt.colorbar(CF, cmap=cmap, drawedges=True)
cbar.set_label(r"OLR (W m$^{-2}$)")
cbar.set_ticks(levels)

# Label the plot
plt.title(
    str(filter_type) + " filtered MJO signal \n Centered on " + str(date[center_index])
)
# plt.title('zonal wavenumber 1-5 filtered signal')
# plt.title('20-100 day filtered signal')
plt.xlabel("Longitude")
plt.ylabel("Lag Day")
plt.yticks(np.arange(-50, 60, 10))
plt.xticks(
    ticks=[0, 60, 120, 180, 240], labels=["0", "60 E", "120 E", "180 E", "240 E"]
)
plt.xlim([0, 240])

# Specify the file name for saving
fig_name = (
    "mjo_signal_"
    + str(time[center_index])
    + "_"
    + str(2 * lag)
    + "_day_lag_"
    + str(filter_type)
    + "_filter.png"
)

# Modify the plot labels and save name if the unfiltered data is plotted
if plot_overlay == "unfiltered":
    print("Plotting Unfiltered Data")
    print("======================")
    CS = plt.contour(
        lon,
        lag_days,
        meridionally_averaged_signal[center_index - 50 : center_index + 51],
        colors="black",
        alpha=0.3,
    )
    plt.title(
        str(filter_type)
        + " filtered MJO signal \n Centered on "
        + str(date[center_index])
        + ", unfiltered signal in contours"
    )
    fig_name = (
        "mjo_signal_"
        + str(time[center_index])
        + "_"
        + str(2 * lag)
        + "_day_lag_"
        + str(filter_type)
        + "_filter_unfiltered_contours.png"
    )
elif plot_overlay == "intraseasonal":
    print("Plotting Intraseasonal Data")
    print("======================")
    intraseasonal_signal = mjo.lanczos_bandpass_filter(
        meridionally_averaged_signal.T,
        temporal_cutoffs[0],
        temporal_cutoffs[1],
        1 / temporal_resolution,
        order=filter_order,
    ).T

    CS = plt.contour(
        lon,
        lag_days,
        intraseasonal_signal[center_index - 50 : center_index + 51],
        colors="black",
        alpha=0.5,
    )
    plt.title(
        str(filter_type)
        + " filtered MJO signal \n Centered on "
        + str(date[center_index])
        + ", intraseasonal signal in contours"
    )
    fig_name = (
        "mjo_signal_"
        + str(time[center_index])
        + "_"
        + str(2 * lag)
        + "_day_lag_"
        + str(filter_type)
        + "_filter_intraseasonal_contours.png"
    )
# Save the plot as a .png file
if save_plots == True:
    print("Saving Plots")
    print("======================")
    fig_dir = r"C:\Users\resse\OneDrive\UW\Research\Figures\MJO Hovmoller Diagrams/"
    plt.savefig(fig_dir + fig_name)
    plt.close()
