#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# mask_land.py                                                      #                                                #
# Spencer Ressel                                                              #
# 2022.7/7                                                                    #   
###############################################################################
'''
This script masks land in netCDF data files.
Input:        Global resolution, daily timeseries in netCDF format
              - Current datasets include NOAA OLR (Liebmann and Smith 1996) and 
              TRMM daily precipitation data
         
Output:       Masked data
Figures:      None
Dependencies: None
'''
#%% Imports
import numpy as np
from netCDF4 import Dataset
import os

#%% Main
def mask_land(signal, max_latitude, Fs_lon):
    '''
    This function takes in 3D data as a function of time, latitude, and 
    longitude, and applies a mask to it, so that all land has value 'np.nan'

    Parameters
    ----------
    data : numpy.ndarray
        The input signal to be masked.
    max_latitude : float
        The latitude band of interest.
    Fs_lon : float
        The degree spacing of the input data as a sampling frequency. This 
        determines what land dataset to use, although currently only 2.5x2.5 
        degrees is allowed. 

    Returns
    -------
    data : numpy.ndarray
        The masked data.

    '''
    #### Mask land data
    print('Masking land')
    
    # Specify the location of the land data
    if os.getcwd()[0] == "C":      
        dir_in = "C:/Users/resse/Desktop/Data/topo/"
    else:
        # dir_in = '/home/disk/eos7/sressel/data/topo/'
        dir_in = '/home/disk/eos7/sressel/research/data/NCEP/'
    
    # Determine which land data to use based on the spacing of the input 
    # data
    if Fs_lon == 1/2.5:
        file_name = 'NCEP_land_2.5deg.nc'
        
    # Load in the land data 
    data = Dataset(dir_in+file_name, "r", format="NETCDF4")
    lat_land = data.variables['lat'][:]
    land = np.array(data.variables['land'][:].squeeze())
    
    # Trim the land data to the region of interest
    imin_land = np.squeeze(np.argwhere(lat_land ==  max_latitude))
    imax_land = np.squeeze(np.argwhere(lat_land == -max_latitude))
    land = land[imin_land:imax_land+1,:]
    
    # Reverse the land data to be from -90~90
    if lat_land[0] > 0:
        land = np.flip(land,0) 
    
    # Mask the data by replacing any regions of land with a np.nan value
    signal = np.where(land==1, np.nan, signal) 
    
    return signal