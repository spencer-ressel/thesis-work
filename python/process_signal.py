#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# process_signal.py                                                           # 
# Spencer Ressel                                                              #  
# 2022.7.11                                                                   #   
###############################################################################

'''
This script performs a series of signal processing procedures to global 
resolution time series data, so that the processed data can be used for 
space-time filtering, computing power spectra, etc. 
Inputs:        Global resolution, daily timeseries as a 2D numpy array, 
               along with the corresponding time, latitude, and longitude
               arrays
               - Current datasets include NOAA OLR (Liebmann and Smith 1996)
                 and TRMM daily precipitation data
              The time limits of interest, in YYYYMMDD format
              The maximum latitude of interest
              The format of the nan data associated with the input data
         
Outputs:      Processed time, latitude, longitude, and data arrays
Figures:      None
Dependencies: mjo_mean_state_diagnostics.py
              
'''
#%% Imports
import numpy as np
import mjo_mean_state_diagnostics as mjo

#%% Function
def process_signal(time, lat, lon, input_signal, time_limits, max_latitude, 
                   nan_value, nan_big_small):
    
    #### Trim data in time and latitude
    print('Sub-Sampling data')
    print('======================')

    # Trim time array
    tmin = time_limits[0]
    tmax = time_limits[1]
    
    itmin = np.argwhere(time==tmin).squeeze()
    itmax = np.argwhere(time==tmax).squeeze()
    try:
        time = time[itmin:itmax+1]
    except:
        raise Exception('Specified time limits outside data range')
    
    
    # Trim latitude array
    latmin_spec = -max_latitude
    imin = np.argwhere(lat==latmin_spec).squeeze()
    imax = np.argwhere(lat==max_latitude).squeeze()
    lat_tropics = lat[imin:imax+1]
    
    # Trim signal array
    signal_time_lat_sampled = input_signal[itmin:itmax+1,imin:imax+1,:]  

    # Replace any NaN values - otherwisw the MJO diagnostics will fail
    signal_time_lat_sampled_nan = mjo.filled_to_nan(signal_time_lat_sampled,
                                                    nan_value,
                                                    nan_big_small)
        
    #### Detrend data
    print('Removing annual cycle')
    # Data without NaN
    if np.sum(np.isnan(signal_time_lat_sampled_nan))==0:
        print(' → No nan data')
        print('======================')
        
    # Data with NaN
    else:
        print('data has nan, can\'t remove annual cycle')
        print('======================')

    # Remove annual cycle and first three harmonics (as in WK99)
    input_signal_tropics_ano,cyc = mjo.remove_anncycle_3d(
                                                signal_time_lat_sampled_nan,
                                                time,
                                                lat_tropics,
                                                lon)
    processed_signal = np.copy(input_signal_tropics_ano)
    
    return time, lat_tropics, lon, processed_signal