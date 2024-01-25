# -*- coding: utf-8 -*-
###############################################################################
# compute_spectrum.py                                                         #
# Written by Mu-Ting Chien                                                    #
# 2021.8.2                                                                    #
# Edited by Spencer Ressel                                                    #
# 2022.6.9                                                                    #
###############################################################################
"""
This script computes the space-time power spectrum of an input signal, using 
the method described in Wheeler and Kiladis 2009 (WK99), among others. 
Input: Global resolution, daily timeseries in netCDF format
       - Current datasets include NOAA OLR (Liebmann and Smith 1996) and 
         TRMM daily precipitation data
         
Output: Power Spectra including background and raw symmetric and asymmetric 
Figures: Space-time power spectra as in WK99.
Dependencies: mjo_mean_state_diagnostics.py
              one_two_one_filter.py
              
"""
#%% Imports
import os
import sys
# from one_two_one_filter import one_two_one_filter
sys.path.insert(0, '/home/disk/eos7/sressel/research/thesis-work/python/auxiliary_functions/')
import mask_land as ml
import numpy as np
import numpy.matlib
import math
from netCDF4 import Dataset, num2date
import scipy.signal as signal
from process_signal import process_signal


# import ipynb.fs.full.mjo_mean_state_diagnostics as mjo
from ipynb.fs.full.one_two_one_filter import one_two_one_filter

#%% Main
def compute_power_spectrum(
    dataset, time_limits, max_latitude, Fs_time, Fs_lon, mask_land=False, n_smooths=15
):

    print("======================")
    print("Loading Data")
    print(" → Dataset: " + dataset)
    print("======================")

    # Specify data location on resse/ or on the server
    if os.getcwd()[0] == "C":
        dir_in = "C:/Users/resse/Desktop/Data/"
    else:
        dir_in = "/home/disk/eos7/sressel/research/data/"
    if dataset == "NOAA":
        nan_value = 9.969209968386869e36
        nan_big_small = np.array([1])

        data = Dataset(dir_in + "NOAA/olr.day.mean.nc", "r", format="NETCDF4")

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
    # Find new size of data
    n_time = np.size(time)
    n_lat_tropics = np.size(lat_tropics)
    n_lon = np.size(lon)

    #### Mask land data
    if mask_land == True:
        processed_signal = ml.mask_land(processed_signal, max_latitude, Fs_lon)
    #### Calculate space-time spectra
    # Separate into symmetric/antisymmetric component
    n_lat_half = int((n_lat_tropics + 1) / 2)  # include equator
    symmetric_signal = np.zeros([n_time, n_lat_half, n_lon])
    asymmetric_signal = np.zeros([n_time, n_lat_half, n_lon])
    for ilat in range(0, n_lat_half):
        symmetric_signal[:, ilat, :] = (
            processed_signal[:, ilat, :]
            + processed_signal[:, n_lat_tropics - ilat - 1, :]
        ) / 2
        asymmetric_signal[:, ilat, :] = (
            -(
                processed_signal[:, ilat, :]
                - processed_signal[:, n_lat_tropics - ilat - 1, :]
            )
            / 2
        )
    # make sure nan becomes zero
    if np.sum(np.isnan(symmetric_signal)) != 0:
        print(
            "Masked signal has "
            + str(np.sum(np.isnan(symmetric_signal[0, :, :])))
            + " nan, replacing with zero"
        )
        print("======================")
        symmetric_signal[np.isnan(symmetric_signal)] = 0
        asymmetric_signal[np.isnan(asymmetric_signal)] = 0
    # Subset into segments in time (96 days, overlap 60 days)
    segment_length = 96
    overlap = 60
    window_width = 5

    # average segment_length (not counting the overlap part)
    avg_segment_length = int(segment_length - overlap)

    # Number of segments
    n_segments = math.floor((n_time - segment_length) / avg_segment_length) + 1

    # Initialize segmented signal arrays
    symmetric_signal_segmented = np.zeros(
        [n_segments, segment_length, n_lat_half, n_lon]
    )
    asymmetric_signal_segmented = np.zeros(
        [n_segments, segment_length, n_lat_half, n_lon]
    )

    # Define the Hann window
    HANN = np.concatenate(
        (
            np.hanning(window_width),
            np.ones(segment_length - window_width * 2),
            np.hanning(window_width),
        ),
        axis=0,
    )

    HANN = np.tile(HANN, (n_lon, n_lat_half, 1))
    HANN = HANN.transpose(2, 1, 0)

    # Detrend the segmented signals and apply the Hann window
    for iseg in range(0, n_segments):
        # iseg_n = int(iseg*avg_segment_length)
        symmetric_signal_segmented[iseg, :, :, :] = (
            signal.detrend(
                symmetric_signal[
                    iseg * avg_segment_length : iseg * avg_segment_length
                    + segment_length,
                    :,
                    :,
                ],
                axis=0,
            )
            * HANN
        )
        asymmetric_signal_segmented[iseg, :, :, :] = (
            signal.detrend(
                asymmetric_signal[
                    iseg * avg_segment_length : iseg * avg_segment_length
                    + segment_length,
                    :,
                    :,
                ],
                axis=0,
            )
            * HANN
        )
    print("Calculating space-time spectra")
    print("======================")
    # Initialize the FFT arrays
    FFT_symmetric_signal_segmented = np.zeros(
        [n_segments, segment_length, n_lon, n_lat_half], dtype=complex
    )
    FFT_aymmetric_signal_segmented = np.zeros(
        [n_segments, segment_length, n_lon, n_lat_half], dtype=complex
    )
    # Compute the FFTs of the segmented signals
    for iseg in range(0, n_segments):
        for ilat in range(0, n_lat_half):
            FFT_symmetric_signal_segmented[iseg, :, :, ilat] = (
                np.fft.fft2(symmetric_signal_segmented[iseg, :, ilat, :])
                / (n_lon * segment_length)
                * 4
            )
            FFT_aymmetric_signal_segmented[iseg, :, :, ilat] = (
                np.fft.fft2(asymmetric_signal_segmented[iseg, :, ilat, :])
                / (n_lon * segment_length)
                * 4
            )
    # Compute the power of the segmented signals
    symmetric_power_segmented = FFT_symmetric_signal_segmented * np.conj(
        FFT_symmetric_signal_segmented
    )
    asymmetric_power_segmented = FFT_aymmetric_signal_segmented * np.conj(
        FFT_aymmetric_signal_segmented
    )
    symmetric_power_segmented = np.real(symmetric_power_segmented)
    asymmetric_power_segmented = np.real(asymmetric_power_segmented)

    # Average over lat and between different segments, and shift to 0-centered
    raw_symmetric_power = np.fft.fftshift(
        np.nanmean(symmetric_power_segmented, axis=(3, 0)), axes=(1, 0)
    )

    raw_asymmetric_power = np.fft.fftshift(
        np.nanmean(asymmetric_power_segmented, axis=(3, 0)), axes=(1, 0)
    )

    # Calculate the frequency and zonal wavenumber axis coordinates
    frequency = (
        np.arange(-segment_length / 2, segment_length / 2) * Fs_time / segment_length
    )
    zonal_wavenumber = np.arange(-n_lon / 2, n_lon / 2) * Fs_lon / n_lon * 360
    x, y = np.meshgrid(zonal_wavenumber, -frequency)

    #### 1-2-1 Filtering
    print("1-2-1 Filtering background")
    print("======================")
    # Smooths the background spectrum 'n_smooths' times
    background_spectrum = (raw_symmetric_power + raw_asymmetric_power) / 2
    background_spectrum = one_two_one_filter(background_spectrum, n_smooths, "time")
    background_spectrum = one_two_one_filter(background_spectrum, n_smooths, "space")

    # Calculate signal strength as raw/smoothed background
    symmetric_power_spectrum = raw_symmetric_power / background_spectrum
    asymmetric_power_spectrum = raw_asymmetric_power / background_spectrum

    # remove artificial signal from satellite: only for olr, not precip
    if dataset == "NOAA":
        aa = np.array([1, -1])
        for a in range(0, 2):
            iymin = np.argwhere(zonal_wavenumber == 14).squeeze()
            iymax = np.argwhere(zonal_wavenumber == 15).squeeze()
            fmin = 0.1 * aa[a]
            fmax = 0.15 * aa[a]
            dmin = np.abs(frequency - fmin)
            dmax = np.abs(frequency - fmax)
            ixmin = np.argwhere(dmin == np.min(dmin)).squeeze()
            ixmax = np.argwhere(dmax == np.min(dmax)).squeeze()
            if a == 0:
                symmetric_power_spectrum[ixmin : ixmax + 1, iymin : iymax + 1] = 0
            elif a == 1:
                symmetric_power_spectrum[ixmax : ixmin + 1, iymin : iymax + 1] = 0
    return (
        x,
        y,
        (
            raw_symmetric_power,
            raw_asymmetric_power,
            background_spectrum,
            symmetric_power_spectrum,
            asymmetric_power_spectrum,
        ),
    )
