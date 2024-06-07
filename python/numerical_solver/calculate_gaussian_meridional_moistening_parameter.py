import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
import sys
sys.path.insert(0, '/home/disk/eos7/sressel/research/thesis-work/python/auxiliary_functions/')
import ipynb.fs.full.mjo_mean_state_diagnostics as mjo
from config import *

def calculate_gaussian_meridional_moistening_parameter(domain_longitudes, domain_latitudes, length_scale_multiplier):
    # Estimate Gaussian Mean Moisture profile
    # Load ERA5 CWV data
    TCWV_data = xr.load_dataset('/home/disk/eos7/sressel/research/data/ECMWF/ERA5/monthly_reanalysis_CWV_CIT_2002_2014.nc')
    total_column_water_vapor = TCWV_data['tcwv'].sel(latitude=slice(90,-90))
    era5_latitude = TCWV_data['latitude'].sel(latitude=slice(90,-90))
    
    # Define the column integrated moisture profile <b>
    column_integrated_moisture_profile = 54.6
    
    # Define the product of the column integrated velocity and moisture profiles <Vb>
    column_integrated_velocity_moisture_product = -19.4

    # Calculate the time and zonal mean CWV from ERA5
    era5_mean_moisture_profile = total_column_water_vapor.sel(
        longitude=slice(domain_longitudes[0],domain_longitudes[1]),
        latitude=slice(domain_latitudes[0], domain_latitudes[1])
    ).mean(dim=['longitude', 'time'])
    
    era5_subset_latitudes = era5_latitude.sel(latitude=slice(domain_latitudes[0], domain_latitudes[1]))
    
    # Define the Gaussian fitting profile
    def Gauss(x, A, B, C, D): 
        y = A + B*np.exp(-1*((x-D)/C)**2) 
        return y 
    
    # Solve for the best fit 
    parameters, covariance = curve_fit(
        Gauss, 
        era5_subset_latitudes.values, 
        era5_mean_moisture_profile.values
    ) 
    
    gaussian_vertical_offset = parameters[0]/column_integrated_moisture_profile
    gaussian_magnitude = parameters[1]/column_integrated_moisture_profile
    gaussian_length_scale = parameters[2]*METERS_PER_DEGREE  
    gaussian_length_scale *= length_scale_multiplier
    gaussian_meridional_offset = parameters[3]*METERS_PER_DEGREE
    
    # Calculate the fitted moisture profile
    gaussian_mean_moisture_profile = Gauss(
        era5_subset_latitudes.values*METERS_PER_DEGREE, 
        gaussian_vertical_offset*column_integrated_moisture_profile,
        gaussian_magnitude*column_integrated_moisture_profile, 
        gaussian_length_scale, 
        # gaussian_meridional_offset,
        0
    ) 

    MERIDIONAL_MOISTENING_PARAMETER = (
        -column_integrated_velocity_moisture_product*2*gaussian_magnitude/(gaussian_length_scale**2)*LATENT_HEAT/SPECIFIC_HEAT
    )