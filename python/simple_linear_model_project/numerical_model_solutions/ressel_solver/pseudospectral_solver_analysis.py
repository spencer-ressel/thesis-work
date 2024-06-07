#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


print(f"==================")
print(f"Loading imports...")
import os
os.chdir(f"/home/disk/eos7/sressel/research/thesis-work/python/numerical_solver/")
import sys
import time
from glob import glob
import json
import numpy as np
import xarray as xr
import scipy as sp
from scipy import special
from scipy.signal import hilbert
from numpy.fft import fft, ifft, fftfreq

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':22})
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.gridspec as gs

from matplotlib.animation import FuncAnimation
from tqdm import tqdm

# # Cartopy
# from cartopy import crs as ccrs
# from cartopy import feature as cf
# from cartopy import util as cutil
# from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter, LongitudeLocator, LatitudeLocator

# Auxiliary Functions
import sys
sys.path.insert(0, '/home/disk/eos7/sressel/research/thesis-work/python/auxiliary_functions/')
import ipynb.fs.full.mjo_mean_state_diagnostics as mjo
from ipynb.fs.full.bmh_colors import bmh_colors
from ipynb.fs.full.rounding_functions import round_out, round_to_multiple
from ipynb.fs.full.modified_colormap import Ahmed21_colormap

Ahmed_cmap = Ahmed21_colormap()
from ipynb.fs.full.tapering_functions import taper_meridionally, fringe_region_damping_function
from ipynb.fs.full.perlin_noise import generate_perlin_noise_2d
from ipynb.fs.full.normalize_data import normalize_data

sys.path.insert(0, '/home/disk/eos7/sressel/research/thesis-work/python/numerical_solver/')
from ipynb.fs.full.numerical_solver_plotting_functions import plot_horizontal_structure, animate_horizontal_structure

# import obspy
print(f"Imports Loaded")
print(f"==================")

# Auxiliary functions
def day_to_index(day):
    return np.abs(downsampled_timepoints/SECONDS_PER_DAY - day).argmin()

save_timestamp = False

experiments_list = [
    # f"output/Ahmed-21/epst=0.50_epsq=0.17_r=0.2_nx=1.0_ny=1.00_wavenumber-filtered_quadratic-mean-moisture_non-diffusive-damped-moist-coupled-simulation",
    # f"output/Ahmed-21/epst=0.50_epsq=0.17_r=0.2_nx=1.0_ny=1.00_wavenumber-filtered_gaussian-mean-moisture_non-diffusive-damped-moist-coupled-simulation", 
    # f"output/Ahmed-21/epst=0.50_epsq-step-y=0.17_r=0.2_nx=1.0_ny=1.00_wavenumber-filtered_gaussian-mean-moisture_non-diffusive-damped-moist-coupled-simulation", 
    # f"output/Ahmed-21/epst=0.00_epsq-step-y=0.17_r=0.2_nx=1.0_ny=1.00_wavenumber-filtered_gaussian-mean-moisture_non-diffusive-damped-moist-coupled-simulation", 
    # f"output/Ahmed-21/epst=0.50_epsq=0.17_r=0.2_nx=1.0_ny=1.00_wavenumber-filtered_asymmetric-gaussian-mean-moisture_non-diffusive-damped-moist-coupled-simulation", 
    # f"output/Ahmed-21/epst=0.50_epsq-step-y=0.17_r=0.2_nx=1.0_ny=1.00_wavenumber-filtered_asymmetric-gaussian-mean-moisture_non-diffusive-damped-moist-coupled-simulation", 
    # f"output/Ahmed-21/epst=0.00_epsq-step-y=0.17_r=0.2_nx=1.0_ny=1.00_wavenumber-filtered_asymmetric-gaussian-mean-moisture_non-diffusive-damped-moist-coupled-simulation", 
]

for index, specified_output_file_directory in enumerate(experiments_list):
    klist = [1,2,3,4,5,6]    
    print(specified_output_file_directory)
    for k in klist:
        print(k)
        specified_initial_condition_name = f"k={k}.0_m=1_Kelvin-wave"
        initial_condition_type = specified_initial_condition_name.split('_')[-1]
        
        print(f"Experiment: {specified_output_file_directory}")
        print(f"Initial Condition: {specified_initial_condition_name}")
        print(f"======================")
        print(f"Loading data...")
        
        downsampled_data = xr.load_dataset(
            f"{specified_output_file_directory}/{initial_condition_type}_initial-condition/"
            + f"{specified_initial_condition_name}" 
            + f"_downsampled-model-data_compiled.nc"
        )
        
        output_zonal_velocity = downsampled_data['u'].to_numpy()
        output_meridional_velocity = downsampled_data['v'].to_numpy()
        output_column_temperature = downsampled_data['T'].to_numpy()
        output_column_moisture = downsampled_data['q'].to_numpy()
        
        output_zonal_gridpoints = downsampled_data.x.to_numpy()
        output_meridional_gridpoints = downsampled_data.y.to_numpy()
        downsampled_timepoints = downsampled_data.time.to_numpy()
        
        print(f"Experiment data loaded")
        
        #### Load experiment variables
        with open(
            f"{specified_output_file_directory}/experiment_variables.json", 'r') as json_file:
            loaded_experiment_variables = json.load(json_file)
        
        simulation_moisture = loaded_experiment_variables['simulation_moisture']
        moisture_advection = loaded_experiment_variables['moisture_advection']
        simulation_damping = loaded_experiment_variables['simulation_damping']
        moisture_coupling = loaded_experiment_variables['moisture_coupling']
        simulation_diffusion = loaded_experiment_variables['simulation_diffusion']
        fringe_region = loaded_experiment_variables['fringe_region']
        moisture_sensitivity_structure = loaded_experiment_variables['moisture_sensitivity_structure']
        temperature_sensitivity_structure = loaded_experiment_variables['temperature_sensitivity_structure']
        mean_moisture_profile = loaded_experiment_variables['mean_moisture_profile']
        gaussian_length_scale = loaded_experiment_variables['moisture_length_scale']
        GRAVITY = loaded_experiment_variables['GRAVITY']
        EQUIVALENT_DEPTH = loaded_experiment_variables['EQUIVALENT_DEPTH']
        CORIOLIS_PARAMETER = loaded_experiment_variables['CORIOLIS_PARAMETER']
        EARTH_RADIUS = loaded_experiment_variables['EARTH_RADIUS']
        AIR_DENSITY = loaded_experiment_variables['AIR_DENSITY']
        WATER_DENSITY = loaded_experiment_variables['WATER_DENSITY']
        LATENT_HEAT = loaded_experiment_variables['LATENT_HEAT']
        SPECIFIC_HEAT = loaded_experiment_variables['SPECIFIC_HEAT']
        DIFFUSIVITY = loaded_experiment_variables['DIFFUSIVITY']
        METERS_PER_DEGREE = loaded_experiment_variables['METERS_PER_DEGREE']
        SECONDS_PER_DAY = loaded_experiment_variables['SECONDS_PER_DAY']
        COLUMN_AVERAGE_MASS = loaded_experiment_variables['COLUMN_AVERAGE_MASS']
        GROSS_DRY_STABILITY = loaded_experiment_variables['GROSS_DRY_STABILITY']
        MOISTURE_SENSITIVITY = loaded_experiment_variables['MOISTURE_SENSITIVITY']
        TEMPERATURE_SENSITIVITY = loaded_experiment_variables['TEMPERATURE_SENSITIVITY']
        CLOUD_RADIATIVE_PARAMETER = loaded_experiment_variables['CLOUD_RADIATIVE_PARAMETER']
        sigma_x_multiplier = loaded_experiment_variables['sigma_x_multiplier']
        sigma_y_multiplier = loaded_experiment_variables['sigma_y_multiplier']
        ZONAL_MOISTENING_PARAMETER = loaded_experiment_variables['ZONAL_MOISTENING_PARAMETER']
        MERIDIONAL_MOISTENING_PARAMETER = loaded_experiment_variables['MERIDIONAL_MOISTENING_PARAMETER']
        MERIDIONAL_OFFSET_PARAMETER = loaded_experiment_variables['MERIDIONAL_OFFSET_PARAMETER']
        gravity_wave_phase_speed = loaded_experiment_variables['gravity_wave_phase_speed']
        time_scale = loaded_experiment_variables['time_scale']
        length_scale = loaded_experiment_variables['length_scale']
        gross_moisture_stratification = loaded_experiment_variables['gross_moisture_stratification']
        effective_sensitivity = loaded_experiment_variables['effective_sensitivity']
        effective_gross_moist_stability = loaded_experiment_variables['effective_gross_moist_stability']
        scaled_zonal_parameter = loaded_experiment_variables['scaled_zonal_parameter']
        scaled_meridional_parameter = loaded_experiment_variables['scaled_meridional_parameter']
        n_days = loaded_experiment_variables['n_days']
        n_chunks = loaded_experiment_variables['n_chunks']
        n_time_steps = loaded_experiment_variables['n_time_steps']
        meridional_domain_length = loaded_experiment_variables['meridional_domain_length']
        zonal_domain_length = loaded_experiment_variables['zonal_domain_length']
        nt = loaded_experiment_variables['nt']    
        nx = loaded_experiment_variables['nx']
        ny = loaded_experiment_variables['ny']
        zonal_grid_spacing = loaded_experiment_variables['zonal_grid_spacing']
        meridional_grid_spacing = loaded_experiment_variables['meridional_grid_spacing']
        simulation_length = loaded_experiment_variables['simulation_length']
        time_step = loaded_experiment_variables['time_step']
        zonal_step_size = loaded_experiment_variables['zonal_step_size']
        meridional_step_size = loaded_experiment_variables['meridional_step_size']
        CFL_x = loaded_experiment_variables['CFL_x']
        CFL_y = loaded_experiment_variables['CFL_y']
        fringe_region_latitude = loaded_experiment_variables['fringe_region_latitude']
        fringe_region_width = loaded_experiment_variables['fringe_region_width']
        fringe_region_strength = loaded_experiment_variables['fringe_region_strength']
        grid_scaling = loaded_experiment_variables['grid_scaling']
        additional_notes = loaded_experiment_variables['additional_notes']
        simulation_name = loaded_experiment_variables['simulation_name']
        output_file_directory = loaded_experiment_variables['output_file_directory']
        n_rk_steps = loaded_experiment_variables['n_rk_steps']
        save_downsampled = loaded_experiment_variables['save_downsampled']
        
        output_wavenumber = eval(specified_initial_condition_name.split('_')[0].split('=')[-1])/EARTH_RADIUS
        zonal_wavenumber      = 2*np.pi*fftfreq(nx, zonal_step_size)       # zonal wavenumbers
        meridional_wavenumber = 2*np.pi*fftfreq(ny, meridional_step_size)  # meridional wavenumbers
        frequencies           = 2*np.pi*fftfreq(nt, time_step)             # frequencies
        
        print("Experiment variables loaded")
        
        # Create a folder to save figures
        if not os.path.exists(f"{specified_output_file_directory}/{initial_condition_type}_initial-condition/figures/"):
            print("======================")
            print("Creating figures folder...")
            os.system(f"mkdir {specified_output_file_directory}/{initial_condition_type}_initial-condition/figures/")
            print("Figures folder created")
            print("======================")
        
        xlims = (-180/k, 180/k)
        ylims = (-35, 35)
        
        # # Visualize Simulation Output
        # # Horizontal Structure
        # # Single time
        # physical_parameters = (
        #     SPECIFIC_HEAT, 
        #     LATENT_HEAT,
        #     WATER_DENSITY,
        #     COLUMN_AVERAGE_MASS,
        #     EARTH_RADIUS,
        #     METERS_PER_DEGREE,
        #     SECONDS_PER_DAY,
        # )
        
        # plotting_parameters = (
        #     # (-180/k, 180/k),               # xlims
        #     xlims,
        #     # (-50, 50),                 # ylims 
        #     ylims,                 # ylims 
        #     20,12,                      # quiver plot spacing (x,y)
        #     True,                     # save plot to png
        #     'converted',               # plotting units - 'converted' or 'natural'
        #     0.6,                       # maximum moisture anomaly in 'mm' - must be set to '1' if units are 'natural'
        #     # 'natural',               # plotting units - 'converted' or 'natural'
        #     # 1,                       # maximum moisture anomaly in 'mm' - must be set to '1' if units are 'natural'
        #     grid_scaling,               # grid-scaling factor - DON'T CHANGE,
        #     save_timestamp             # Whether or not to timestamp the output file
        # )
        
        # # Plot the initial conditon below
        # plot_horizontal_structure(
        #     day_to_index(0),
        #     output_zonal_gridpoints,
        #     output_meridional_gridpoints,
        #     downsampled_timepoints,
        #     zonal_velocity = np.copy(output_zonal_velocity), 
        #     meridional_velocity = np.copy(output_meridional_velocity), 
        #     column_temperature = np.copy(output_column_temperature), 
        #     column_moisture = np.copy(output_column_moisture), 
        #     specified_output_file_directory = specified_output_file_directory,
        #     specified_initial_condition_name = specified_initial_condition_name,
        #     physical_parameters = physical_parameters,
        #     simulation_parameters = (simulation_moisture, fringe_region, fringe_region_latitude, fringe_region_width),
        #     plotting_parameters = plotting_parameters
        # )
        
        # # Plot the initial conditon below
        # plot_horizontal_structure(
        #     day_to_index(360),
        #     output_zonal_gridpoints,
        #     output_meridional_gridpoints,
        #     downsampled_timepoints,
        #     zonal_velocity = np.copy(output_zonal_velocity), 
        #     meridional_velocity = np.copy(output_meridional_velocity), 
        #     column_temperature = np.copy(output_column_temperature), 
        #     column_moisture = np.copy(output_column_moisture), 
        #     specified_output_file_directory = specified_output_file_directory,
        #     specified_initial_condition_name = specified_initial_condition_name,
        #     physical_parameters = physical_parameters,
        #     simulation_parameters = (simulation_moisture, fringe_region, fringe_region_latitude, fringe_region_width),
        #     plotting_parameters = plotting_parameters
        # )
        
        
        # # Animation
        # starting_frame = day_to_index(0)
        # ending_frame = day_to_index(360)
        # frame_index = day_to_index(5)
        
        # physical_parameters = (
        #     SPECIFIC_HEAT, 
        #     LATENT_HEAT,
        #     WATER_DENSITY,
        #     COLUMN_AVERAGE_MASS,
        #     EARTH_RADIUS,
        #     METERS_PER_DEGREE,
        #     SECONDS_PER_DAY,
        # )
        
        # plotting_parameters = (
        #     # (-180, 179),
        #     (-180/k, 180/k),
        #     (-35, 35),
        #     20, 12,
        #     # 16//k,12//k,
        #     True, 
        #     'converted',
        #     0.6,
        #     grid_scaling,
        #     save_timestamp            
        # )
        
        # animate_horizontal_structure(
        #     output_zonal_gridpoints,
        #     output_meridional_gridpoints,
        #     downsampled_timepoints,
        #     np.copy(output_zonal_velocity),
        #     np.copy(output_meridional_velocity),
        #     np.copy(output_column_temperature),
        #     np.copy(output_column_moisture),
        #     specified_output_file_directory = specified_output_file_directory,
        #     specified_initial_condition_name = specified_initial_condition_name,
        #     physical_parameters = physical_parameters,
        #     simulation_parameters = (simulation_moisture, fringe_region),
        #     plotting_parameters = plotting_parameters,
        #     frames = np.arange(starting_frame, ending_frame, frame_index),
        #     normalized_over_time=True
        # )
        
        
        # # Temporal Structure        
        # # Near-Equatorial Average
        # plt.style.use('bmh')
        # plt.rcParams.update({'font.size':24})
        
        # plotting_time_points = downsampled_timepoints
        
        # # Latitudes over which to average
        # south_bound = -10
        # north_bound = 10
        
        # # Find the indices of the meridional grid corresponding to those latitudes
        # south_lat_index = (
        #     np.abs(output_meridional_gridpoints/METERS_PER_DEGREE - south_bound).argmin()
        #     if south_bound >= np.min(output_meridional_gridpoints)/METERS_PER_DEGREE else 0
        # )
        # north_lat_index = (
        #     1 + np.abs(output_meridional_gridpoints/METERS_PER_DEGREE - north_bound).argmin() 
        #     if north_bound <= np.max(output_meridional_gridpoints)/METERS_PER_DEGREE else None
        # )
        
        # # Average the field variables over the specified latitudes
        # near_equatorial_column_temperature = np.mean(
        #     np.copy(output_column_temperature)[:, south_lat_index:north_lat_index, :],
        #     axis=1
        # )
        # near_equatorial_column_moisture = np.mean(
        #     np.copy(output_column_moisture)[:, south_lat_index:north_lat_index, :],
        #     axis=1
        # )
        # near_equatorial_zonal_velocity = np.mean(
        #     np.copy(output_zonal_velocity)[:, south_lat_index:north_lat_index, :],
        #     axis=1
        # )
        # near_equatorial_meridional_velocity = np.mean(
        #     np.copy(output_meridional_velocity)[:, south_lat_index:north_lat_index, :],
        #     axis=1
        # )
        
        # max_index = np.argmax(near_equatorial_column_temperature)
        # [t_index, x_index] = np.unravel_index(max_index, [nt, nx])
        
        # [fig, ax] = plt.subplots(figsize=(16,6))
        # ax.set_title(
        #     (
        #         f"{np.abs(south_bound)}°S-{np.abs(north_bound)}°N averaged {initial_condition_type} amplitude \n over time," 
        #       + r" ε$_t$ = " + f"{3600*TEMPERATURE_SENSITIVITY:0.3f}" + r" hr$^{-1},$"
        #       + r" ε$_q$ = " + f"{3600*MOISTURE_SENSITIVITY:0.3f}" + r" hr$^{-1}$"
        #     ), pad=15
        # )
        
        # ax.axhline(
        #     y=0,
        #     color='black',
        #     ls='--',
        #     alpha=0.75
        # )
        
        # ax.plot(
        #     plotting_time_points/SECONDS_PER_DAY, 
        #     near_equatorial_column_temperature[:, x_index]*gravity_wave_phase_speed/GROSS_DRY_STABILITY, 
        #     lw=3, 
        #     label=r"$\frac{c}{M_s}\langle T \rangle$",
        #     color=bmh_colors('blue')
        # )
        
        # ax.plot(
        #     plotting_time_points/SECONDS_PER_DAY, 
        #     near_equatorial_column_moisture[:, x_index]*gravity_wave_phase_speed/gross_moisture_stratification, 
        #     lw=3, 
        #     label=r"$\frac{c}{M_q}\langle q \rangle$",
        #     color=bmh_colors('red')
            
        # )
        
        # ax.plot(
        #     plotting_time_points/SECONDS_PER_DAY, 
        #     -near_equatorial_zonal_velocity[:, x_index], 
        #     lw=3, 
        #     ls='--',
        #     label='u',
        #     color=bmh_colors('purple')
            
        # )
        
        # ax.plot(
        #     plotting_time_points/SECONDS_PER_DAY, 
        #     -near_equatorial_meridional_velocity[:, x_index], 
        #     lw=3, 
        #     ls='--',
        #     label='v',
        #     color=bmh_colors('green')
        # )
        
        # # Maximum column temperature
        # ax.axhline(
        #     y=np.max(near_equatorial_column_temperature[:, x_index])*gravity_wave_phase_speed/GROSS_DRY_STABILITY,
        #     color=bmh_colors('blue'),
        #     ls=':',
        #     alpha=0.75
        # )
        
        # ax.axhline(
        #     y=np.min(near_equatorial_column_temperature[:, x_index])*gravity_wave_phase_speed/GROSS_DRY_STABILITY,
        #     color=bmh_colors('blue'),
        #     ls=':',
        #     alpha=0.75
        # )
        
        # # Maximum column moisture
        # ax.axhline(
        #     y=np.max(near_equatorial_column_moisture[:, x_index])*gravity_wave_phase_speed/gross_moisture_stratification,
        #     color=bmh_colors('red'),
        #     ls=':',
        #     alpha=0.75
        # )
        
        # ax.axhline(
        #     y=np.min(near_equatorial_column_moisture[:, x_index])*gravity_wave_phase_speed/gross_moisture_stratification,
        #     color=bmh_colors('red'),
        #     ls=':',
        #     alpha=0.75
        # )
        
        # ax.set_xlabel('Time (days)')
        # ax.set_ylabel(r"$\frac{m}{s}$", rotation=0, labelpad=20, fontsize=32)
        
        # ax.legend(loc='best', fontsize=18)
        
        # if initial_condition_type == 'EIG-wave' or initial_condition_type == 'WIG-wave':
        #     plt.xlim(-1, 5)
        

        # plt.savefig(
        #     f"{specified_output_file_directory}/{initial_condition_type}_initial-condition/figures/"
        #     + f"{specified_initial_condition_name}_temporal-structure"
        #     + (f"_{time.strftime('%Y%m%d-%H%M')}" if save_timestamp else '')
        #     + f".png", 
        #     bbox_inches='tight'
        # )
            
        # Pattern & Phase Speed Correlation
        # Calculate Phase Speed
        
        south_bound = -90
        north_bound = 90
        
        south_lat_index = (
            np.abs(output_meridional_gridpoints/METERS_PER_DEGREE - south_bound).argmin()
            if south_bound >= np.min(output_meridional_gridpoints)/METERS_PER_DEGREE else 0
        )
        north_lat_index = (
            1 + np.abs(output_meridional_gridpoints/METERS_PER_DEGREE - north_bound).argmin() 
            if north_bound <= np.max(output_meridional_gridpoints)/METERS_PER_DEGREE else None
        )
        
        phase_speed_moisture = np.copy(output_column_moisture)
        normalized_column_moisture = np.mean(phase_speed_moisture[:, south_lat_index:north_lat_index], axis=1)
        
        #### Phase Speed     
        phase_speed_correlation = np.einsum(
            'k,ik->i',
            np.exp(1j*output_wavenumber*output_zonal_gridpoints),
            normalized_column_moisture
        ) / len(output_zonal_gridpoints)
        
        # Calculate the phase
        phase = np.log(phase_speed_correlation).imag
        instantaneous_phase_speed_array = np.gradient(np.unwrap(phase), downsampled_timepoints)*(1/output_wavenumber)
        
        instantaneous_phase_speed =  xr.DataArray(
            data = instantaneous_phase_speed_array,
            dims = ['time'],
            coords = {'time' : downsampled_timepoints},
            name = 'phase speed',
            attrs = {'Latitude Bounds' : (south_bound, north_bound)}
        )
        
        # Save instantaneous phase speed as a netCDF file
        phase_speed_file_name = (
            f"{specified_output_file_directory}/{initial_condition_type}_initial-condition/"
            + f"{specified_initial_condition_name}_instantaneous-phase-speed" 
            + f"_{mjo.tick_labeller([south_bound], 'lat', False)[0]}-{mjo.tick_labeller([north_bound], 'lat', False)[0]}.nc"
        )
        instantaneous_phase_speed.to_netcdf(phase_speed_file_name)
        print(f"Instantaneous Phase Speed saved as:\n{phase_speed_file_name}")
        
        
        # Calculate Pattern Correlation
        south_bound = -90
        north_bound = 90
        
        south_lat_index = (
            np.abs(output_meridional_gridpoints/METERS_PER_DEGREE - south_bound).argmin()
            if south_bound >= np.min(output_meridional_gridpoints)/METERS_PER_DEGREE else 0
        )
        north_lat_index = (
            1 + np.abs(output_meridional_gridpoints/METERS_PER_DEGREE - north_bound).argmin() 
            if north_bound <= np.max(output_meridional_gridpoints)/METERS_PER_DEGREE else None
        )
        
        # Initialize arrays for the instantaneous phase speed & pattern correlation
        backward_pattern_correlation = xr.DataArray(
            data = np.empty((len(downsampled_timepoints))),
            dims = ['time'],
            coords = {'time' : downsampled_timepoints},
            name = 'correlation',
            attrs = {'Latitude Bounds' : (south_bound, north_bound)}
        )
        
        forward_pattern_correlation = xr.DataArray(
            data = np.empty((len(downsampled_timepoints))),
            dims = ['time'],
            coords = {'time' : downsampled_timepoints},
            name = 'correlation',
            attrs = {'Latitude Bounds' : (south_bound, north_bound)}
        )
        
        pattern_correlation_moisture = np.copy(output_column_moisture)
        
        # Iterate over the length of the simulation
        for day_index in range(len(downsampled_timepoints)):
        
            daily_column_moisture = pattern_correlation_moisture[day_index]
            final_column_moisture = pattern_correlation_moisture[-1]
            
            # Calculate the pattern correlation between
            backward_pattern_correlation[day_index] = np.einsum(
                'ij,ij->',
                daily_column_moisture, 
                final_column_moisture
            ) / (
                np.std(daily_column_moisture)
                * np.std(final_column_moisture) 
                * len(output_zonal_gridpoints) 
                * len(output_meridional_gridpoints)
            )
        
            # forward_pattern_correlation[day_index] = np.einsum(
            #     'ij,ij->',
            #     output_column_moisture[day_index], 
            #     output_column_moisture[0],
            # ) / (
            #     np.std(output_column_moisture[day_index])
            #     * np.std(output_column_moisture[0]) 
            #     * len(output_zonal_gridpoints) 
            #     * len(output_meridional_gridpoints)
            # )
        
        # Save pattern correlation as a netCDF file
        # Save instantaneous phase speed as a netCDF file
        pattern_correlation_file_name = (
            f"{specified_output_file_directory}/{initial_condition_type}_initial-condition/"
            + f"{specified_initial_condition_name}_pattern-correlation" 
            + f"_{mjo.tick_labeller([south_bound], 'lat', False)[0]}-{mjo.tick_labeller([north_bound], 'lat', False)[0]}.nc"
        )
        backward_pattern_correlation.to_netcdf(pattern_correlation_file_name)
        print(f"Pattern Correlation saved as:\n{pattern_correlation_file_name}")
        print(f"==============================")
        
        
        # Plot Phase Speed & Pattern Correlation
        # Pattern correlation and phase speed
        
        loaded_phase_speed = xr.load_dataarray(phase_speed_file_name)
        print(f"Instantaneous phase speed loaded from:\n{phase_speed_file_name}")
        print(f"===============================================================")
        
        loaded_pattern_correlation = xr.load_dataarray(pattern_correlation_file_name)
        print(f"Pattern correlation loaded from:\n{pattern_correlation_file_name}")
        print(f"===============================================================")
        
        peaks = sp.signal.find_peaks(loaded_pattern_correlation**2)[0]
        padded_peaks = np.insert(peaks, 0, 0)
        padded_peaks = np.append(padded_peaks, len(downsampled_timepoints)-1)
        
        ### Plot the phase speed
        plt.style.use('bmh')
        plt.rcParams.update({'font.size':22})
        plt.figure(figsize=(16,9))
        plt.title(
            f"k = {k} {initial_condition_type} initial condition, \n"
            + f"instantaneous phase speed (blue) and pattern correlation (red), "
            + f"{mjo.tick_labeller([south_bound], 'lat')[0]}-{mjo.tick_labeller([north_bound], 'lat')[0]}",
            fontsize=20,
            pad=10
        )
        plt.plot(
            downsampled_timepoints/SECONDS_PER_DAY,
            loaded_phase_speed,
            color=bmh_colors('blue'),
            lw=3,
            # marker='o'
        )
        
        # Add a line with the phase speed of A21
        # plt.axhline(y=6.7, color='black', ls='--', alpha=0.5, label='A21: 6.7 m/s')
        
        y_max = np.max(loaded_phase_speed[day_to_index(3):])
        y_min = np.min(loaded_phase_speed[day_to_index(3):])
        plt.xlim(3, 360)
        plt.ylim(0.9*y_min, 1.1*y_max)
        plt.xlabel('Day')
        plt.ylabel('m/s', rotation=0, labelpad=40, va='center')
        plt.gca().spines['left'].set_color(bmh_colors('blue'))
        plt.gca().spines['left'].set_linewidth(4)
        
        ### Plot the Pattern correlation
        plt.twinx()
        plt.grid(False)
        plt.plot(
            downsampled_timepoints/SECONDS_PER_DAY,
            loaded_pattern_correlation**2,
            color=bmh_colors('red'),
            lw=2,
            alpha=0.5,
            # marker='o'
        )
        plt.plot(
            downsampled_timepoints[padded_peaks]/SECONDS_PER_DAY, 
            loaded_pattern_correlation[padded_peaks]**2, 
            color=bmh_colors('red'),
            ls=':',
            lw=3
        )
        plt.axhline(y=0, color='k', alpha=0.2)
        plt.axhline(y=1, color='k', alpha=0.2)
        plt.gca().spines['right'].set_color(bmh_colors('red'))
        plt.gca().spines['right'].set_linewidth(4)
        plt.ylabel(r"r$^{2}$", rotation=0, va='center', labelpad=20)
        plt.ylim(-0.05, 1.05)
        
        plt.xlim(-5,365)
        plt.show()
        plt.savefig(
              f"{specified_output_file_directory}/{initial_condition_type}_initial-condition/figures/"
            + f"{specified_initial_condition_name}_phase-speed_pattern-correlation"
            + f"_{mjo.tick_labeller([south_bound], 'lat')[0]}-{mjo.tick_labeller([north_bound], 'lat')[0]}"
            + (f"_{time.strftime('%Y%m%d-%H%M')}" if save_timestamp else '')
            + f".png", 
            bbox_inches='tight'
        )
        
        # downsampled_data = xr.load_dataset(
        #     f"{specified_output_file_directory}/{initial_condition_type}_initial-condition/"
        #     + f"{specified_initial_condition_name}" 
        #     + f"_downsampled-model-data_compiled.nc"
        # )
        
        # output_zonal_velocity = downsampled_data['u'].to_numpy()
        # output_meridional_velocity = downsampled_data['v'].to_numpy()
        # output_column_temperature = downsampled_data['T'].to_numpy()
        # output_column_moisture = downsampled_data['q'].to_numpy()
        
        # output_zonal_gridpoints = downsampled_data.x.to_numpy()
        # output_meridional_gridpoints = downsampled_data.y.to_numpy()
        # downsampled_timepoints = downsampled_data.time.to_numpy()
        
        # print(f"Experiment data loaded")
        
    #     #### Load experiment variables
    #     with open(
    #         f"{specified_output_file_directory}/experiment_variables.json", 'r') as json_file:
    #         loaded_experiment_variables = json.load(json_file)
        
    #     simulation_moisture = loaded_experiment_variables['simulation_moisture']
    #     moisture_advection = loaded_experiment_variables['moisture_advection']
    #     simulation_damping = loaded_experiment_variables['simulation_damping']
    #     moisture_coupling = loaded_experiment_variables['moisture_coupling']
    #     simulation_diffusion = loaded_experiment_variables['simulation_diffusion']
    #     fringe_region = loaded_experiment_variables['fringe_region']
    #     moisture_sensitivity_structure = loaded_experiment_variables['moisture_sensitivity_structure']
    #     temperature_sensitivity_structure = loaded_experiment_variables['temperature_sensitivity_structure']
    #     mean_moisture_profile = loaded_experiment_variables['mean_moisture_profile']
    #     gaussian_length_scale = loaded_experiment_variables['moisture_length_scale']
    #     GRAVITY = loaded_experiment_variables['GRAVITY']
    #     EQUIVALENT_DEPTH = loaded_experiment_variables['EQUIVALENT_DEPTH']
    #     CORIOLIS_PARAMETER = loaded_experiment_variables['CORIOLIS_PARAMETER']
    #     EARTH_RADIUS = loaded_experiment_variables['EARTH_RADIUS']
    #     AIR_DENSITY = loaded_experiment_variables['AIR_DENSITY']
    #     WATER_DENSITY = loaded_experiment_variables['WATER_DENSITY']
    #     LATENT_HEAT = loaded_experiment_variables['LATENT_HEAT']
    #     SPECIFIC_HEAT = loaded_experiment_variables['SPECIFIC_HEAT']
    #     DIFFUSIVITY = loaded_experiment_variables['DIFFUSIVITY']
    #     METERS_PER_DEGREE = loaded_experiment_variables['METERS_PER_DEGREE']
    #     SECONDS_PER_DAY = loaded_experiment_variables['SECONDS_PER_DAY']
    #     COLUMN_AVERAGE_MASS = loaded_experiment_variables['COLUMN_AVERAGE_MASS']
    #     GROSS_DRY_STABILITY = loaded_experiment_variables['GROSS_DRY_STABILITY']
    #     MOISTURE_SENSITIVITY = loaded_experiment_variables['MOISTURE_SENSITIVITY']
    #     TEMPERATURE_SENSITIVITY = loaded_experiment_variables['TEMPERATURE_SENSITIVITY']
    #     CLOUD_RADIATIVE_PARAMETER = loaded_experiment_variables['CLOUD_RADIATIVE_PARAMETER']
    #     sigma_x_multiplier = loaded_experiment_variables['sigma_x_multiplier']
    #     sigma_y_multiplier = loaded_experiment_variables['sigma_y_multiplier']
    #     ZONAL_MOISTENING_PARAMETER = loaded_experiment_variables['ZONAL_MOISTENING_PARAMETER']
    #     MERIDIONAL_MOISTENING_PARAMETER = loaded_experiment_variables['MERIDIONAL_MOISTENING_PARAMETER']
    #     MERIDIONAL_OFFSET_PARAMETER = loaded_experiment_variables['MERIDIONAL_OFFSET_PARAMETER']
    #     gravity_wave_phase_speed = loaded_experiment_variables['gravity_wave_phase_speed']
    #     time_scale = loaded_experiment_variables['time_scale']
    #     length_scale = loaded_experiment_variables['length_scale']
    #     gross_moisture_stratification = loaded_experiment_variables['gross_moisture_stratification']
    #     effective_sensitivity = loaded_experiment_variables['effective_sensitivity']
    #     effective_gross_moist_stability = loaded_experiment_variables['effective_gross_moist_stability']
    #     scaled_zonal_parameter = loaded_experiment_variables['scaled_zonal_parameter']
    #     scaled_meridional_parameter = loaded_experiment_variables['scaled_meridional_parameter']
    #     n_days = loaded_experiment_variables['n_days']
    #     n_chunks = loaded_experiment_variables['n_chunks']
    #     n_time_steps = loaded_experiment_variables['n_time_steps']
    #     meridional_domain_length = loaded_experiment_variables['meridional_domain_length']
    #     zonal_domain_length = loaded_experiment_variables['zonal_domain_length']
    #     nt = loaded_experiment_variables['nt']    
    #     nx = loaded_experiment_variables['nx']
    #     ny = loaded_experiment_variables['ny']
    #     zonal_grid_spacing = loaded_experiment_variables['zonal_grid_spacing']
    #     meridional_grid_spacing = loaded_experiment_variables['meridional_grid_spacing']
    #     simulation_length = loaded_experiment_variables['simulation_length']
    #     time_step = loaded_experiment_variables['time_step']
    #     zonal_step_size = loaded_experiment_variables['zonal_step_size']
    #     meridional_step_size = loaded_experiment_variables['meridional_step_size']
    #     CFL_x = loaded_experiment_variables['CFL_x']
    #     CFL_y = loaded_experiment_variables['CFL_y']
    #     fringe_region_latitude = loaded_experiment_variables['fringe_region_latitude']
    #     fringe_region_width = loaded_experiment_variables['fringe_region_width']
    #     fringe_region_strength = loaded_experiment_variables['fringe_region_strength']
    #     grid_scaling = loaded_experiment_variables['grid_scaling']
    #     additional_notes = loaded_experiment_variables['additional_notes']
    #     simulation_name = loaded_experiment_variables['simulation_name']
    #     output_file_directory = loaded_experiment_variables['output_file_directory']
    #     n_rk_steps = loaded_experiment_variables['n_rk_steps']
    #     save_downsampled = loaded_experiment_variables['save_downsampled']
        
    #     output_wavenumber = eval(specified_initial_condition_name.split('_')[0].split('=')[-1])/EARTH_RADIUS
    #     zonal_wavenumber      = 2*np.pi*fftfreq(nx, zonal_step_size)       # zonal wavenumbers
    #     meridional_wavenumber = 2*np.pi*fftfreq(ny, meridional_step_size)  # meridional wavenumbers
    #     frequencies           = 2*np.pi*fftfreq(nt, time_step)             # frequencies
        
    #     print("Experiment variables loaded")
        
    #     # Budget Analysis
    #     # Define budget calculation function
    #     def calculate_budget(field, column_MSE):
    #         budget = np.einsum(
    #             '...ji, ...ji -> ...',
    #             field,
    #             column_MSE
    #         ) / np.einsum(
    #             '...ji, ...ji -> ...',
    #             column_MSE,
    #             column_MSE
    #         )
    #         return budget
        
        
    #     # Growth Budget 
    #     # Calculate growth budget
    #     south_bound = -90
    #     north_bound = 90
    #     south_lat_index = (
    #         np.abs(output_meridional_gridpoints/METERS_PER_DEGREE - south_bound).argmin()
    #         if south_bound >= np.min(output_meridional_gridpoints)/METERS_PER_DEGREE else 0
    #     )
    #     north_lat_index = (
    #         1 + np.abs(output_meridional_gridpoints/METERS_PER_DEGREE - north_bound).argmin() 
    #         if north_bound <= np.max(output_meridional_gridpoints)/METERS_PER_DEGREE else None
    #     )
        
    #     # Calculate the column MSE : <h> = <T> + <q>
    #     column_MSE = np.copy(output_column_temperature + output_column_moisture)
        
    #     # Calculate the MSE tendency directly from the variable
    #     MSE_tendency = np.gradient(column_MSE, downsampled_timepoints, axis=0)
        
    #     growth_budget = calculate_budget(
    #         MSE_tendency[:, south_lat_index:north_lat_index],
    #         column_MSE[:, south_lat_index:north_lat_index]
    #     )
        
    #     # Calculate the contribution of zonal advection : σ_x {u_1 <h>}
    #     zonal_advection_growth_contribution = ZONAL_MOISTENING_PARAMETER*calculate_budget(
    #         output_zonal_velocity[:, south_lat_index:north_lat_index, :], 
    #         column_MSE[:, south_lat_index:north_lat_index, :]
    #     )
        
    #     # Calculate the contribution of meridional advection : -σ_y {y v_1 <h>}
    #     meridional_advection_growth_contribution = -MERIDIONAL_MOISTENING_PARAMETER*calculate_budget(
    #         (
    #             ( # Quadratic base state σ_y * y * v_1
    #                 output_meridional_gridpoints
    #                 if mean_moisture_profile == 'quadratic' else np.ones_like(output_meridional_gridpoints)
    #             )
    #             * ( # Exponential base state σ_y * y * e^(-y^2) * v_1
    #                 (output_meridional_gridpoints*np.exp(-(output_meridional_gridpoints/gaussian_length_scale)**2))
    #                 if mean_moisture_profile == 'gaussian' else np.ones_like(output_meridional_gridpoints)
    #             )
    #             * ( # Offset gaussian base state σ_y * y * e^(-(y-δ)^2/σ^2) * v_1
    #                 (
    #                     (output_meridional_gridpoints-MERIDIONAL_OFFSET_PARAMETER)*np.exp(
    #                         -((output_meridional_gridpoints-MERIDIONAL_OFFSET_PARAMETER)/gaussian_length_scale)**2
    #                     )
    #                 )
    #                 if mean_moisture_profile == 'asymmetric-gaussian' else np.ones_like(output_meridional_gridpoints)
    #             )
    #         )[None, south_lat_index:north_lat_index, None]
    #         *output_meridional_velocity[:, south_lat_index:north_lat_index, :], 
    #         column_MSE[:, south_lat_index:north_lat_index, :]
    #     )
        
    #     ### Compute the divergence of the velocity field to get vertical velocity
    #     # dudx
    #     output_ux_fft = fft(np.copy(output_zonal_velocity), axis=2)
    #     output_dudx_fft = np.einsum(
    #         'i, kji -> kji',
    #         (1j*zonal_wavenumber),
    #         output_ux_fft
    #     )
    #     output_dudx = np.real(ifft(output_dudx_fft, axis=2))
        
    #     # dvdy
    #     output_vy_fft = fft(np.copy(output_meridional_velocity), axis=1)
    #     output_dvdy_fft = np.einsum(
    #         'j, kji -> kji',
    #         (1j*meridional_wavenumber),
    #         output_vy_fft
    #     )
    #     output_dvdy = np.real(ifft(output_dvdy_fft, axis=1))
        
    #     # ω = -(dudx + dvdy)
    #     divergence = output_dudx + output_dvdy
    #     vertical_velocity = -divergence
        
    #     # Calculate the contribution of vertical advection : m M_s {ω_1 <h>}
    #     vertical_advection_growth_contribution = (
    #         (GROSS_DRY_STABILITY - gross_moisture_stratification)/GROSS_DRY_STABILITY
    #         * GROSS_DRY_STABILITY
    #         * calculate_budget(
    #             vertical_velocity[:, south_lat_index:north_lat_index, :], 
    #             column_MSE[:, south_lat_index:north_lat_index, :]
    #         )
    #     )
        
    #     # Calculate column convective heating <Q_c> = ε_q<q> -ε_t<T> 
    #     moisture_sensitivity_array = MOISTURE_SENSITIVITY * (
    #         (
    #             np.exp(-(output_meridional_gridpoints/length_scale)**2)
    #             if moisture_sensitivity_structure == '-exp-y' else np.ones_like(output_meridional_gridpoints)
    #         )
    #         *(
    #         (1-fringe_region_damping_function(output_meridional_gridpoints/METERS_PER_DEGREE, -30, 30, 25, 1))
    #             if  moisture_sensitivity_structure == '-step-y' else np.ones_like(output_meridional_gridpoints)
    #         )
    #     )[None, :, None]
        
    #     temperature_sensitivity_array = TEMPERATURE_SENSITIVITY * (
    #         (
    #             np.exp(-(output_meridional_gridpoints/length_scale)**2)
    #             if temperature_sensitivity_structure == '-exp-y' else np.ones_like(output_meridional_gridpoints)
    #         )
    #         *(
    #         (1-fringe_region_damping_function(output_meridional_gridpoints/METERS_PER_DEGREE, -30, 30, 25, 1))
    #             if  temperature_sensitivity_structure == '-step-y' else np.ones_like(output_meridional_gridpoints)
    #         )
    #     )[None, :, None]
        
    #     column_convective_heating = (
    #         moisture_sensitivity_array * output_column_moisture
    #         - temperature_sensitivity_array * output_column_temperature
    #     ) 
        
    #     # Calculate column radiative heating <Q_r> = r<Q_c>
    #     column_radiative_heating = CLOUD_RADIATIVE_PARAMETER*column_convective_heating
        
    #     # Calculate the contribution of column radiative heating : {<Q_r> <h>}
    #     column_radiative_heating_growth_contribution = calculate_budget(
    #         column_radiative_heating[:, south_lat_index:north_lat_index, :], 
    #         column_MSE[:, south_lat_index:north_lat_index, :]
    #     )
        
    #     #### # Calculate the contribution of diffusion : {𝒟∇^2(<h>)*<h>}
    #     MSEx_fft = fft(column_MSE, axis=2)
    #     MSEy_fft = fft(column_MSE, axis=1)
        
    #     dMSEdx_dx_fft = np.einsum(
    #         'i, kji -> kji',
    #         (1j*zonal_wavenumber)**2, 
    #         MSEx_fft 
    #     )
        
    #     dMSEdy_dy_fft = np.einsum(
    #         'j, kji -> kji',   
    #         (1j*meridional_wavenumber)**2,
    #         MSEy_fft
    #     )
        
    #     dMSEdx_dx = np.real(ifft(dMSEdx_dx_fft, axis=2))
    #     dMSEdy_dy = np.real(ifft(dMSEdy_dy_fft, axis=1))
        
    #     laplacian_MSE = dMSEdx_dx + dMSEdy_dy 
    #     diffusion_growth_contribution = DIFFUSIVITY*calculate_budget(
    #         laplacian_MSE[:, south_lat_index:north_lat_index, :],
    #         column_MSE[:, south_lat_index:north_lat_index, :]
    #     )
        
    #     #### Residual
    #     residual_MSE_growth = growth_budget - (
    #         vertical_advection_growth_contribution
    #         + zonal_advection_growth_contribution
    #         + meridional_advection_growth_contribution
    #         + column_radiative_heating_growth_contribution
    #         + diffusion_growth_contribution
    #     )
        
    #     growth_budget_dataset = xr.Dataset(
    #         data_vars = {
    #             'growth' : (['time'], growth_budget),
    #             'omega'  : (['time'], vertical_advection_growth_contribution),
    #             'u'      : (['time'], zonal_advection_growth_contribution),
    #             'v'      : (['time'], meridional_advection_growth_contribution),
    #             'Qr'     : (['time'], column_radiative_heating_growth_contribution),
    #             'D'      : (['time'], diffusion_growth_contribution),
    #             'residual':(['time'], residual_MSE_growth),
    #         },
    #         coords = {
    #             'time' : downsampled_timepoints
    #         },
    #         attrs= {
    #             'Latitude Bounds' : (south_bound, north_bound),
    #             'Initial Wavenumber' : output_wavenumber*EARTH_RADIUS
    #         }
    #     )
        
    #     growth_budget_filename = (
    #         f"{specified_output_file_directory}/{initial_condition_type}_initial-condition/{specified_initial_condition_name}" 
    #         + f"_growth-budget_{mjo.tick_labeller([south_bound], 'lat', False)[0]}-{mjo.tick_labeller([north_bound], 'lat', False)[0]}.nc"
    #     )
    #     growth_budget_dataset.to_netcdf(growth_budget_filename)
    #     print(f"Growth budget saved as:\n{growth_budget_filename}")
        
        
    #     # Plot growth budget
    #     # Animation
    #     # Load growth budget
    #     loaded_growth_budget_filename = (
    #         f"{specified_output_file_directory}/{initial_condition_type}_initial-condition/{specified_initial_condition_name}" 
    #         + f"_growth-budget_{mjo.tick_labeller([south_bound], 'lat', False)[0]}-{mjo.tick_labeller([north_bound], 'lat', False)[0]}.nc"
    #     )
    #     loaded_growth_budget = xr.load_dataset(loaded_growth_budget_filename)
    #     print(f"Growth budget loaded from:\n{loaded_growth_budget_filename}")
    #     print(f"========================================================")
        
    #     starting_frame = day_to_index(2)
    #     ending_frame = day_to_index(360)
    #     frame_interval = day_to_index(5)
        
    #     bar_labels = [
    #         r'Growth',
    #         r'ω$_1$mM$_s$',
    #         r'σ$_x$u$_1$',
    #         r'-σ$_y$yv$_1$', 
    #         r'$\langle$Q$_r$$\rangle$',
    #         # r'$\mathcal{D}\nabla^{2} \langle$h$\rangle$', 
    #         r'residual'
    #     ]
    #     bar_colors = [
    #         '#ffa500', 
    #         '#1cf91a', 
    #         '#0533ff', 
    #         '#ff40ff', 
    #         '#4e2d8c', 
    #         # 'red', 
    #         '#bcbcbc'
    #     ]
        
    #     grand_max = np.max(
    #         (
    #             SECONDS_PER_DAY*loaded_growth_budget['growth'][starting_frame:ending_frame],
    #             SECONDS_PER_DAY*loaded_growth_budget['omega'][starting_frame:ending_frame],
    #             SECONDS_PER_DAY*loaded_growth_budget['u'][starting_frame:ending_frame],
    #             SECONDS_PER_DAY*loaded_growth_budget['v'][starting_frame:ending_frame],
    #             SECONDS_PER_DAY*loaded_growth_budget['Qr'][starting_frame:ending_frame],
    #             # SECONDS_PER_DAY*loaded_growth_budget['D'][starting_frame:ending_frame],
    #             SECONDS_PER_DAY*loaded_growth_budget['residual'][starting_frame:ending_frame],
    #         )
    #     )
        
    #     grand_min = np.min(
    #         (
    #             SECONDS_PER_DAY*loaded_growth_budget['growth'][starting_frame:ending_frame],
    #             SECONDS_PER_DAY*loaded_growth_budget['omega'][starting_frame:ending_frame],
    #             SECONDS_PER_DAY*loaded_growth_budget['u'][starting_frame:ending_frame],
    #             SECONDS_PER_DAY*loaded_growth_budget['v'][starting_frame:ending_frame],
    #             SECONDS_PER_DAY*loaded_growth_budget['Qr'][starting_frame:ending_frame],
    #             # SECONDS_PER_DAY*loaded_growth_budget['D'][starting_frame:ending_frame],
    #             SECONDS_PER_DAY*loaded_growth_budget['residual'][starting_frame:ending_frame],
    #         )
    #     )
        
    #     plt.style.use('default')
    #     plt.rcParams.update({'font.size':22})
    #     [fig, ax] = plt.subplots(1, 1, figsize=(16.5, 6.4))
        
    #     def update(t):
    #         bar_values = [
    #             SECONDS_PER_DAY*loaded_growth_budget['growth'][t],
    #             SECONDS_PER_DAY*loaded_growth_budget['omega'][t],
    #             SECONDS_PER_DAY*loaded_growth_budget['u'][t],
    #             SECONDS_PER_DAY*loaded_growth_budget['v'][t],
    #             SECONDS_PER_DAY*loaded_growth_budget['Qr'][t],
    #             # SECONDS_PER_DAY*loaded_growth_budget['D'][t],
    #             SECONDS_PER_DAY*loaded_growth_budget['residual'][t],
    #             ]
        
    #         ax.clear()
    #         ax.set_title(
    #             f"Growth budget, day {downsampled_timepoints[t]/SECONDS_PER_DAY :0.1f}, " 
    #             + f"{mjo.tick_labeller([south_bound], 'lat')[0]}-{mjo.tick_labeller([north_bound], 'lat')[0]}", 
    #             pad=10
    #     )
    #         ax.bar(bar_labels, bar_values, color=bar_colors, edgecolor='gray', linewidth=3)
    #         ax.axhline(y=0, color='gray', lw=3, alpha=0.75)
    #         ax.set_ylabel(r'day$^{-1}$', labelpad=20, rotation=0, va='center')
    #         ax.set_ylim(1.1*round_out(grand_min, 'tenths'), 1.1*round_out(grand_max, 'tenths'))
        
    #     # Run the animation
    #     anim = FuncAnimation(
    #         fig, 
    #         update, 
    #         frames=tqdm(
    #             np.arange(starting_frame, ending_frame, frame_interval), 
    #             ncols=100, 
    #             position=0, 
    #             leave=True
    #         ), interval=300
    #     )
        
    #     anim.save(
    #         f"{specified_output_file_directory}/{initial_condition_type}_initial-condition/figures/"
    #         + f"{specified_initial_condition_name}"
    #         + f"_growth-budget_animation"
    #         + (f"_{time.strftime('%Y%m%d-%H%M')}" if save_timestamp else '')
    #         + f".mp4", 
    #         dpi=200
    #     )
        
        
    #     # Propagation Budget
    #     # Calculate propagation budget
    #     south_bound = -90
    #     north_bound = 90
    #     south_lat_index = (
    #         np.abs(output_meridional_gridpoints/METERS_PER_DEGREE - south_bound).argmin()
    #         if south_bound >= np.min(output_meridional_gridpoints)/METERS_PER_DEGREE else 0
    #     )
    #     north_lat_index = (
    #         1 + np.abs(output_meridional_gridpoints/METERS_PER_DEGREE - north_bound).argmin() 
    #         if north_bound <= np.max(output_meridional_gridpoints)/METERS_PER_DEGREE else None
    #     )
        
    #     # Calculate the column MSE : <h> = <T> + <q>
    #     column_MSE = np.copy(output_column_temperature + output_column_moisture)
        
    #     # Calculate the MSE tendency directly from the variable
    #     MSE_tendency = np.gradient(column_MSE, downsampled_timepoints, axis=0)
        
    #     propagation_budget = calculate_budget(
    #         MSE_tendency[:, south_lat_index:north_lat_index, :],
    #         MSE_tendency[:, south_lat_index:north_lat_index, :]
    #     )
        
    #     # Calculate the contribution of zonal advection : σ_x {u_1 <h>}
    #     zonal_advection_propagation_contribution = ZONAL_MOISTENING_PARAMETER*calculate_budget(
    #         output_zonal_velocity[:, south_lat_index:north_lat_index, :], 
    #         MSE_tendency[:, south_lat_index:north_lat_index, :]
    #     )
        
    #     # Calculate the contribution of meridional advection : -σ_y {y v_1 <h>}
    #     meridional_advection_propagation_contribution = -MERIDIONAL_MOISTENING_PARAMETER*calculate_budget(
    #         (
    #             ( # Quadratic base state σ_y * y * v_1
    #                 output_meridional_gridpoints
    #                 if mean_moisture_profile == 'quadratic' else np.ones_like(output_meridional_gridpoints)
    #             )
    #             * ( # Gaussian base state σ_y * y * e^(-y^2/σ^2) * v_1
    #                 (output_meridional_gridpoints*np.exp(-(output_meridional_gridpoints/gaussian_length_scale)**2))
    #                 if mean_moisture_profile == 'gaussian' else np.ones_like(output_meridional_gridpoints)
    #             )
    #             * ( # Offset gaussian base state σ_y * y * e^(-(y-δ)^2/σ^2) * v_1
    #                 (
    #                     (output_meridional_gridpoints-MERIDIONAL_OFFSET_PARAMETER)*np.exp(
    #                         -((output_meridional_gridpoints-MERIDIONAL_OFFSET_PARAMETER)/gaussian_length_scale)**2
    #                     )
    #                 )
    #                 if mean_moisture_profile == 'asymmetric-gaussian' else np.ones_like(output_meridional_gridpoints)
    #             )
    #         )[None, south_lat_index:north_lat_index, None]
    #         * output_meridional_velocity[:, south_lat_index:north_lat_index, :], 
    #         MSE_tendency[:, south_lat_index:north_lat_index, :]
    #     )
        
    #     ### Compute the divergence of the velocity field to get vertical velocity
    #     # dudx
    #     output_ux_fft = fft(output_zonal_velocity, axis=2)
    #     output_dudx_fft = 1j*zonal_wavenumber[None,:]*output_ux_fft
    #     output_dudx = np.real(ifft(output_dudx_fft, axis=2))
        
    #     # dvdy
    #     output_vy_fft = fft(output_meridional_velocity, axis=1)
    #     output_dvdy_fft = 1j*meridional_wavenumber[:,None]*output_vy_fft
    #     output_dvdy = np.real(ifft(output_dvdy_fft, axis=1))
        
    #     # ω = -(dudx + dvdy)
    #     divergence = output_dudx + output_dvdy
    #     vertical_velocity = -divergence
        
    #     # Calculate the contribution of vertical advection : m M_s {ω_1 <h>}
    #     vertical_advection_propagation_contribution = (
    #         (GROSS_DRY_STABILITY - gross_moisture_stratification)/GROSS_DRY_STABILITY
    #         * GROSS_DRY_STABILITY
    #         * calculate_budget(
    #             vertical_velocity[:, south_lat_index:north_lat_index, :], 
    #             MSE_tendency[:, south_lat_index:north_lat_index, :]
    #         )
    #     )
        
    #     # Calculate column convective heating <Q_c> = ε_q<q> -ε_t<T> 
    #     moisture_sensitivity_array = MOISTURE_SENSITIVITY * (
    #         (
    #             np.exp(-(output_meridional_gridpoints/length_scale)**2)
    #             if moisture_sensitivity_structure == '-exp-y' else np.ones_like(output_meridional_gridpoints)
    #         )
    #         *(
    #         (1-fringe_region_damping_function(output_meridional_gridpoints/METERS_PER_DEGREE, -30, 30, 25, 1))
    #             if  moisture_sensitivity_structure == '-step-y' else np.ones_like(output_meridional_gridpoints)
    #         )
    #     )[None, :, None]
        
    #     temperature_sensitivity_array = TEMPERATURE_SENSITIVITY * (
    #         (
    #             np.exp(-(output_meridional_gridpoints/length_scale)**2)
    #             if temperature_sensitivity_structure == '-exp-y' else np.ones_like(output_meridional_gridpoints)
    #         )
    #         *(
    #         (1-fringe_region_damping_function(output_meridional_gridpoints/METERS_PER_DEGREE, -30, 30, 25, 1))
    #             if  temperature_sensitivity_structure == '-step-y' else np.ones_like(output_meridional_gridpoints)
    #         )
    #     )[None, :, None]
        
    #     column_convective_heating = (
    #         moisture_sensitivity_array * output_column_moisture
    #         - temperature_sensitivity_array * output_column_temperature
    #     ) 
            
    #     # Calculate column radiative heating <Q_r> = r<Q_c>
    #     column_radiative_heating = CLOUD_RADIATIVE_PARAMETER*column_convective_heating
        
    #     # Calculate the contribution of column radiative heating : {<Q_r> <h>}
    #     column_radiative_heating_propagation_contribution = calculate_budget(
    #         column_radiative_heating[:, south_lat_index:north_lat_index, :],
    #         MSE_tendency[:, south_lat_index:north_lat_index, :]
    #     )
        
    #     #### # Calculate the contribution of diffusion : {𝒟∇^2<h> <h>}
    #     MSEx_fft = fft(column_MSE, axis=2)
    #     MSEy_fft = fft(column_MSE, axis=1)
        
    #     dMSEdx_dx_fft = np.einsum(
    #         'i, kji -> kji',
    #         (1j*zonal_wavenumber)**2, 
    #         MSEx_fft 
    #     )
        
    #     dMSEdy_dy_fft = np.einsum(
    #         'j, kji -> kji',   
    #         (1j*meridional_wavenumber)**2,
    #         MSEy_fft
    #     )
        
    #     dMSEdx_dx = np.real(ifft(dMSEdx_dx_fft, axis=2))
    #     dMSEdy_dy = np.real(ifft(dMSEdy_dy_fft, axis=1))
        
    #     laplacian_MSE = dMSEdx_dx + dMSEdy_dy 
    #     diffusion_propagation_contribution = DIFFUSIVITY*calculate_budget(
    #         laplacian_MSE[:, south_lat_index:north_lat_index, :],
    #         MSE_tendency[:, south_lat_index:north_lat_index, :]
    #     )
        
    #     #### Residual
    #     residual_MSE_propagation = propagation_budget - (
    #         vertical_advection_propagation_contribution
    #         + zonal_advection_propagation_contribution
    #         + meridional_advection_propagation_contribution
    #         + column_radiative_heating_propagation_contribution
    #         + diffusion_propagation_contribution
    #     )
        
    #     propagation_budget_dataset = xr.Dataset(
    #         data_vars = {
    #             'propagation' : (['time'], propagation_budget),
    #             'omega'  : (['time'], vertical_advection_propagation_contribution),
    #             'u'      : (['time'], zonal_advection_propagation_contribution),
    #             'v'      : (['time'], meridional_advection_propagation_contribution),
    #             'Qr'     : (['time'], column_radiative_heating_propagation_contribution),
    #             'D'      : (['time'], diffusion_propagation_contribution),
    #             'residual':(['time'], residual_MSE_propagation),
    #         },
    #         coords = {
    #             'time' : downsampled_timepoints
    #         },
    #         attrs= {
    #             'Latitude Bounds' : (south_bound, north_bound),
    #             'Initial Wavenumber' : output_wavenumber*EARTH_RADIUS
    #         }
    #     )
        
    #     propagation_budget_filename = (
    #         f"{specified_output_file_directory}/{initial_condition_type}_initial-condition/{specified_initial_condition_name}" 
    #         + f"_propagation-budget_{mjo.tick_labeller([south_bound], 'lat', False)[0]}-{mjo.tick_labeller([north_bound], 'lat', False)[0]}.nc"
    #     )
    #     propagation_budget_dataset.to_netcdf(propagation_budget_filename)
    #     print(f"Propagation budget saved as:\n{propagation_budget_filename}")
        
        
    #     # Plot propagation budget
    #     # Animation
        
    #     starting_frame = day_to_index(2)
    #     ending_frame = day_to_index(360)
    #     frame_interval = day_to_index(5)
        
    #     # Load growth budget
    #     loaded_propagation_budget_filename = (
    #         f"{specified_output_file_directory}/{initial_condition_type}_initial-condition/{specified_initial_condition_name}" 
    #         + f"_propagation-budget_{mjo.tick_labeller([south_bound], 'lat', False)[0]}-{mjo.tick_labeller([north_bound], 'lat', False)[0]}.nc"
    #     )
    #     loaded_propagation_budget = xr.load_dataset(loaded_propagation_budget_filename)
    #     print(f"Propagation budget loaded from:\n{loaded_propagation_budget_filename}")
    #     print(f"========================================================")
        
    #     grand_max = np.max(
    #         (
    #             loaded_propagation_budget['propagation'][starting_frame:ending_frame],
    #             loaded_propagation_budget['omega'][starting_frame:ending_frame],
    #             loaded_propagation_budget['u'][starting_frame:ending_frame],
    #             loaded_propagation_budget['v'][starting_frame:ending_frame],
    #             loaded_propagation_budget['Qr'][starting_frame:ending_frame],
    #             # loaded_propagation_budget['D'][starting_frame:ending_frame],
    #             loaded_propagation_budget['residual'][starting_frame:ending_frame],
    #         )
    #     )
    #     grand_min = np.min(
    #         (
    #             loaded_propagation_budget['propagation'][starting_frame:ending_frame],
    #             loaded_propagation_budget['omega'][starting_frame:ending_frame],
    #             loaded_propagation_budget['u'][starting_frame:ending_frame],
    #             loaded_propagation_budget['v'][starting_frame:ending_frame],
    #             loaded_propagation_budget['Qr'][starting_frame:ending_frame],
    #             # loaded_propagation_budget['D'][starting_frame:ending_frame],
    #             loaded_propagation_budget['residual'][starting_frame:ending_frame],
    #         )
    #     )
        
    #     bar_labels = [
    #         r'Propagation',
    #         r'ω$_1$mM$_s$',
    #         r'σ$_x$u$_1$',
    #         r'-σ$_y$yv$_1$', 
    #         r'$\langle$Q$_r$$\rangle$',
    #         # r'$\mathcal{D}\nabla^{2} \langle$h$\rangle$', 
    #         r'residual'
    #     ]
    #     bar_colors = [
    #         '#ffa500',
    #         '#1cf91a',
    #         '#0533ff',
    #         '#ff40ff',
    #         '#4e2d8c',
    #         # 'red',
    #         '#bcbcbc'
    #     ]
        
    #     plt.style.use('default')
    #     plt.rcParams.update({'font.size':22})
    #     [fig, ax] = plt.subplots(1, 1, figsize=(16.5, 6.4))
        
        
    #     def update(t):
    #         bar_values = [
    #             loaded_propagation_budget['propagation'][t],
    #             loaded_propagation_budget['omega'][t],
    #             loaded_propagation_budget['u'][t],
    #             loaded_propagation_budget['v'][t],
    #             loaded_propagation_budget['Qr'][t],
    #             # loaded_propagation_budget['D'][t],
    #             loaded_propagation_budget['residual'][t],
    #             ]
        
    #         plt.cla()
    #         ax.set_title(
    #             f"Propagation budget, day {downsampled_timepoints[t]/SECONDS_PER_DAY :0.1f}, " 
    #             + f"{mjo.tick_labeller([south_bound], 'lat')[0]}-{mjo.tick_labeller([north_bound], 'lat')[0]}", 
    #             pad=10
    #     )
    #         ax.bar(bar_labels, bar_values, color=bar_colors, edgecolor='gray', linewidth=3)
    #         ax.axhline(y=0, color='gray', lw=3, alpha=0.75)
    #         ax.set_ylabel(r'hr$^{-1}$', labelpad=20, rotation=0, va='center')
    #         ax.set_ylim(1.1*round_out(grand_min, 'tenths'), 1.1*round_out(grand_max, 'tenths'))
        
    #     # Run the animation
    #     anim = FuncAnimation(
    #         fig, 
    #         update, 
    #         frames=tqdm(
    #             np.arange(starting_frame, ending_frame, frame_interval), 
    #             ncols=100, 
    #             position=0, 
    #             leave=True
    #         ), interval=300
    #     )
        
    #     anim.save(
    #         f"{specified_output_file_directory}/{initial_condition_type}_initial-condition/figures/"
    #         + f"{specified_initial_condition_name}"
    #         + f"_propagation-budget_animation"
    #         + (f"_{time.strftime('%Y%m%d-%H%M')}" if save_timestamp else '')
    #         + f".mp4", 
    #         dpi=300
    #     )
        
        
    #     # Horizontal structure of budget terms
    #     # Calculate horizontal structure of advection fields
    #     print(f"Calculating advection fields...")
    #     print(f"===============================")
        
    #     # Calculate the column MSE : <h> = <T> + <q>
    #     column_MSE = np.copy(output_column_temperature + output_column_moisture)
        
    #     # Calculate the MSE tendency directly from the variable
    #     MSE_tendency = np.gradient(column_MSE, downsampled_timepoints, axis=0)
        
    #     #### Zonal advection field
    #     zonal_advection_field = ZONAL_MOISTENING_PARAMETER*output_zonal_velocity
        
    #     #### Meridional advection field
    #     meridional_advection_field = -MERIDIONAL_MOISTENING_PARAMETER*(
    #         (
    #             ( # Quadratic base state σ_y * y * v_1
    #                 output_meridional_gridpoints
    #                 if mean_moisture_profile == 'quadratic' else np.ones_like(output_meridional_gridpoints)
    #             )
    #             * ( # Exponential base state σ_y * y * e^(-y^2) * v_1
    #                 (output_meridional_gridpoints*np.exp(-(output_meridional_gridpoints/gaussian_length_scale)**2))
    #                 if mean_moisture_profile == 'gaussian' else np.ones_like(output_meridional_gridpoints)
    #             )
    #             * ( # Offset gaussian base state σ_y * y * e^(-(y-δ)^2/σ^2) * v_1
    #                 (
    #                     (output_meridional_gridpoints-MERIDIONAL_OFFSET_PARAMETER)*np.exp(
    #                         -((output_meridional_gridpoints-MERIDIONAL_OFFSET_PARAMETER)/gaussian_length_scale)**2
    #                     )
    #                 )
    #                 if mean_moisture_profile == 'asymmetric-gaussian' else np.ones_like(output_meridional_gridpoints)
    #             )
    #         )[None, :, None]
    #         * output_meridional_velocity 
    #     )
        
    #     #### Vertical advection field
    #     ## Compute the divergence of the velocity field to get vertical velocity
    #     # dudx
    #     output_ux_fft = fft(output_zonal_velocity, axis=2)
    #     output_dudx_fft = np.einsum(
    #         'i, kji -> kji',
    #         (1j*zonal_wavenumber),
    #         output_ux_fft
    #     )
    #     output_dudx = np.real(ifft(output_dudx_fft, axis=2))
        
    #     # dvdy
    #     output_vy_fft = fft(output_meridional_velocity, axis=1)
    #     output_dvdy_fft = np.einsum(
    #         'j, kji -> kji',
    #         (1j*meridional_wavenumber),
    #         output_vy_fft
    #     )
    #     output_dvdy = np.real(ifft(output_dvdy_fft, axis=1))
        
    #     # ω = -(dudx + dvdy)
    #     divergence = output_dudx + output_dvdy
    #     vertical_velocity = -divergence
        
    #     vertical_advection_field = (
    #         (GROSS_DRY_STABILITY - gross_moisture_stratification)/GROSS_DRY_STABILITY
    #         * GROSS_DRY_STABILITY
    #         * vertical_velocity
    #     )
        
    #     # Calculate column convective heating <Q_c> = ε_q<q> -ε_t<T> 
    #     moisture_sensitivity_array = MOISTURE_SENSITIVITY * (
    #         (
    #             np.exp(-(output_meridional_gridpoints/length_scale)**2)
    #             if moisture_sensitivity_structure == '-exp-y' else np.ones_like(output_meridional_gridpoints)
    #         )
    #         *(
    #         (1-fringe_region_damping_function(output_meridional_gridpoints/METERS_PER_DEGREE, -30, 30, 25, 1))
    #             if  moisture_sensitivity_structure == '-step-y' else np.ones_like(output_meridional_gridpoints)
    #         )
    #     )[None, :, None]
        
    #     temperature_sensitivity_array = TEMPERATURE_SENSITIVITY * (
    #         (
    #             np.exp(-(output_meridional_gridpoints/length_scale)**2)
    #             if temperature_sensitivity_structure == '-exp-y' else np.ones_like(output_meridional_gridpoints)
    #         )
    #         *(
    #         (1-fringe_region_damping_function(output_meridional_gridpoints/METERS_PER_DEGREE, -30, 30, 25, 1))
    #             if  temperature_sensitivity_structure == '-step-y' else np.ones_like(output_meridional_gridpoints)
    #         )
    #     )[None, :, None]
        
    #     column_convective_heating = (
    #         moisture_sensitivity_array * output_column_moisture
    #         - temperature_sensitivity_array * output_column_temperature
    #     ) 
        
    #     # Calculate column radiative heating <Q_r> = r<Q_c>
    #     column_radiative_heating = CLOUD_RADIATIVE_PARAMETER*column_convective_heating
        
    #     print(f"Advection fields calculated")
        
        
    #     # Plot MSE structure and advection fields 
    #     bar_colors = [
    #         '#ffa500', 
    #         '#1cf91a',
    #         '#0533ff', 
    #         '#ff40ff', 
    #         '#4e2d8c', 
    #         # 'red', 
    #         '#bcbcbc'
    #     ]
        
    #     # Animation
        
    #     plt.style.use('default')
    #     plt.rcParams.update({'font.size':16})
    #     fig = plt.figure(figsize=(16.5, 6.5))
    #     # gs_main = gs.GridSpec(2, 1, height_ratios = [30,1], figure=fig)
    #     gs_main = gs.GridSpec(1, 1, figure=fig)
    #     gs_main.update(left=0.05, right=0.95, top=0.9, bottom=0.05, hspace=0.25)
        
    #     gs_maps = gs.GridSpecFromSubplotSpec(1, 2, wspace=0.05, subplot_spec=gs_main[0])
    #     # gs_cbar = gs.GridSpecFromSubplotSpec(1, 30, subplot_spec=gs_main[1])
        
    #     ax = []
    #     # Add an axis for the initial condition
    #     ax.append(fig.add_subplot(gs_maps[0]))
    #     ax.append(fig.add_subplot(gs_maps[1]))
        
    #     # cbar_ax = fig.add_subplot(gs_cbar[:, 1:-1])
        
    #     starting_frame = day_to_index(0)
    #     ending_frame = day_to_index(360)
    #     frame_interval = day_to_index(5)
    #     plt.suptitle(f"Day {downsampled_timepoints[starting_frame]/SECONDS_PER_DAY :0.1f}", y=0.995, fontsize=20)
        
    #     def update(t):
    #         plt.suptitle(f"Day {downsampled_timepoints[t]/SECONDS_PER_DAY :0.1f}", y=0.995, fontsize=20)
            
    #         #### First plot
    #         ax[0].clear()
    #         ax[0].set_title(r'(a) MSE, -σ$_y$yv$_1$, and σ$_x$u$_1$')
    #         CF_MSE = ax[0].contourf(
    #             output_zonal_gridpoints*grid_scaling,
    #             output_meridional_gridpoints*grid_scaling,
    #             column_MSE[t]/COLUMN_AVERAGE_MASS,
    #             cmap=Ahmed_cmap,
    #             norm=mcolors.CenteredNorm(vcenter=0),
    #             levels=21,
    #             # alpha=0.75
    #         )
            
    #         CS_v = ax[0].contour(
    #             output_zonal_gridpoints*grid_scaling,
    #             output_meridional_gridpoints*grid_scaling,
    #             meridional_advection_field[t],
    #             colors=bar_colors[3],
    #             levels=np.delete(np.linspace(np.min(meridional_advection_field[t]), np.max(meridional_advection_field[t]), 11), [5])
    #         )
            
    #         CS_u = ax[0].contour(
    #             output_zonal_gridpoints*grid_scaling,
    #             output_meridional_gridpoints*grid_scaling,
    #             zonal_advection_field[t],
    #             colors=bar_colors[2],
    #             levels=np.delete(np.linspace(np.min(zonal_advection_field[t]), np.max(zonal_advection_field[t]), 11), [5])
    #         )

    #         longitude_ticks = np.arange(xlims[0]+xlims[1]/3, xlims[1]+xlims[1]/3, xlims[1]/3)
    #         longitude_labels = mjo.tick_labeller(longitude_ticks, direction='lon')
    #         ax[0].set_xticks(longitude_ticks*METERS_PER_DEGREE*grid_scaling, labels=longitude_labels)
    #         latitude_ticks = np.arange(ylims[0], ylims[1]+ylims[1]/3, ylims[1]/3)
    #         latitude_labels = mjo.tick_labeller(latitude_ticks, direction='lat')
    #         ax[0].set_yticks(latitude_ticks*METERS_PER_DEGREE*grid_scaling, labels=latitude_labels)
    #         ax[0].set_xlim(xlims[0]*METERS_PER_DEGREE*grid_scaling,(xlims[1]-1)*METERS_PER_DEGREE*grid_scaling)
    #         ax[0].set_ylim(ylims[0]*METERS_PER_DEGREE*grid_scaling,ylims[1]*METERS_PER_DEGREE*grid_scaling)
    #         ax[0].set_aspect('auto')
    #         # cbar = plt.colorbar(CF_MSE, cax=cbar_ax, orientation='horizontal')
    #         # cbar.set_label('K', rotation=0, va='center', labelpad=20)
            
    #         #### Second Plot
    #         ax[1].clear()
    #         ax[1].set_title(r'(b) MSE, $\langle$Q$_r$$\rangle$, and ω$_1$mM$_s$')
    #         ax[1].contourf(
    #             output_zonal_gridpoints*grid_scaling,
    #             output_meridional_gridpoints*grid_scaling,
    #             column_MSE[t]/COLUMN_AVERAGE_MASS,
    #             cmap=Ahmed_cmap,
    #             norm=mcolors.CenteredNorm(vcenter=0),
    #             levels=21,
    #             # alpha=0.75
    #         )
            
    #         CS_Qr = ax[1].contour(
    #             output_zonal_gridpoints*grid_scaling,
    #             output_meridional_gridpoints*grid_scaling,
    #             column_radiative_heating[t],
    #             colors=bar_colors[4],
    #             levels=np.delete(np.linspace(np.min(column_radiative_heating[t]), np.max(column_radiative_heating[t]), 11), [5])
    #         )
            
    #         CS_omega = ax[1].contour(
    #             output_zonal_gridpoints*grid_scaling,
    #             output_meridional_gridpoints*grid_scaling,
    #             vertical_advection_field[t],
    #             colors=bar_colors[1],
    #             levels=np.delete(np.linspace(np.min(vertical_advection_field[t]), np.max(vertical_advection_field[t]), 11), [5])
    #         )
        
    #         longitude_ticks = np.arange(xlims[0]+xlims[1]/3, xlims[1]+xlims[1]/3, xlims[1]/3)
    #         longitude_labels = mjo.tick_labeller(longitude_ticks, direction='lon')
    #         ax[1].set_xticks(longitude_ticks*METERS_PER_DEGREE*grid_scaling, labels=longitude_labels)
    #         latitude_ticks = np.arange(ylims[0], ylims[1]+ylims[1]/3, ylims[1]/3)
    #         latitude_labels = mjo.tick_labeller(latitude_ticks, direction='lat')
    #         ax[1].set_yticks(latitude_ticks*METERS_PER_DEGREE*grid_scaling, labels=latitude_labels)
    #         ax[1].set_yticklabels('')
    #         ax[1].set_xlim(xlims[0]*METERS_PER_DEGREE*grid_scaling,(xlims[1]-1)*METERS_PER_DEGREE*grid_scaling)
    #         ax[1].set_ylim(ylims[0]*METERS_PER_DEGREE*grid_scaling,ylims[1]*METERS_PER_DEGREE*grid_scaling)
    #         ax[1].set_aspect('auto')
        
    #     # Run the animation
    #     anim = FuncAnimation(
    #         fig, 
    #         update, 
    #         frames=tqdm(
    #             np.arange(starting_frame, ending_frame, frame_interval), 
    #             ncols=100, 
    #             position=0, 
    #             leave=True
    #         ), interval=300
    #     )
        
    #     anim.save(
    #         f"{specified_output_file_directory}/{initial_condition_type}_initial-condition/figures/"
    #         + f"{specified_initial_condition_name}"
    #         + f"_MSE-structure_animation"
    #         + (f"_{time.strftime('%Y%m%d-%H%M')}" if save_timestamp else '')
    #         + f".mp4", 
    #         dpi=150
    #     )
        
        
    #     # Plot MSE Tendency structure and advection fields
    #     # Animation        
    #     plt.style.use('default')
    #     plt.rcParams.update({'font.size':16})
    #     fig = plt.figure(figsize=(16.5, 6.5))
    #     gs_main = gs.GridSpec(1, 1, figure=fig)
    #     gs_main.update(left=0.05, right=0.95, top=0.9, bottom=0.05, hspace=0.25)
        
    #     gs_maps = gs.GridSpecFromSubplotSpec(1, 2, wspace=0.05, subplot_spec=gs_main[0])
    #     # gs_cbar = gs.GridSpecFromSubplotSpec(1, 30, subplot_spec=gs_main[1])
        
    #     ax = []
    #     # Add an axis for the initial condition
    #     ax.append(fig.add_subplot(gs_maps[0]))
    #     ax.append(fig.add_subplot(gs_maps[1]))
        
    #     # cbar_ax = fig.add_subplot(gs_cbar[:, 1:-1])
        
    #     starting_frame = day_to_index(0)
    #     ending_frame = day_to_index(360)
    #     frame_interval = day_to_index(5)
    #     plt.suptitle(f"Day {downsampled_timepoints[starting_frame]/SECONDS_PER_DAY :0.1f}", y=0.995, fontsize=20)
        
    #     def update(t):
    #         plt.suptitle(f"Day {downsampled_timepoints[t]/SECONDS_PER_DAY :0.1f}", y=0.995, fontsize=20)
            
    #         #### First plot
    #         ax[0].clear()
    #         ax[0].set_title(r'(a) $\frac{dMSE}{dt}$, -σ$_y$yv$_1$, and σ$_x$u$_1$')
    #         CF_MSE = ax[0].contourf(
    #             output_zonal_gridpoints*grid_scaling,
    #             output_meridional_gridpoints*grid_scaling,
    #             MSE_tendency[t]/COLUMN_AVERAGE_MASS,
    #             cmap=Ahmed_cmap,
    #             norm=mcolors.CenteredNorm(vcenter=0),
    #             levels=21,
    #             # alpha=0.75
    #         )
            
    #         CS_v = ax[0].contour(
    #             output_zonal_gridpoints*grid_scaling,
    #             output_meridional_gridpoints*grid_scaling,
    #             meridional_advection_field[t],
    #             colors=bar_colors[3],
    #             levels=np.delete(np.linspace(np.min(meridional_advection_field[t]), np.max(meridional_advection_field[t]), 11), [5])
    #         )
            
    #         CS_u = ax[0].contour(
    #             output_zonal_gridpoints*grid_scaling,
    #             output_meridional_gridpoints*grid_scaling,
    #             zonal_advection_field[t],
    #             colors=bar_colors[2],
    #             levels=np.delete(np.linspace(np.min(zonal_advection_field[t]), np.max(zonal_advection_field[t]), 11), [5])
    #         )
            
    #         longitude_ticks = np.arange(xlims[0]+xlims[1]/3, xlims[1]+xlims[1]/3, xlims[1]/3)
    #         longitude_labels = mjo.tick_labeller(longitude_ticks, direction='lon')
    #         ax[0].set_xticks(longitude_ticks*METERS_PER_DEGREE*grid_scaling, labels=longitude_labels)
    #         latitude_ticks = np.arange(ylims[0], ylims[1]+ylims[1]/3, ylims[1]/3)
    #         latitude_labels = mjo.tick_labeller(latitude_ticks, direction='lat')
    #         ax[0].set_yticks(latitude_ticks*METERS_PER_DEGREE*grid_scaling, labels=latitude_labels)
    #         ax[0].set_xlim(xlims[0]*METERS_PER_DEGREE*grid_scaling,(xlims[1]-1)*METERS_PER_DEGREE*grid_scaling)
    #         ax[0].set_ylim(ylims[0]*METERS_PER_DEGREE*grid_scaling,ylims[1]*METERS_PER_DEGREE*grid_scaling)
    #         ax[0].set_aspect('auto')
    #         # cbar = plt.colorbar(CF_MSE, cax=cbar_ax, orientation='horizontal')
    #         # cbar.set_label('K/s', rotation=0, va='center', labelpad=20)
            
    #         #### Second Plot
    #         ax[1].clear()
    #         ax[1].set_title(r'(b) $\frac{dMSE}{dt}$, $\langle$Q$_r$$\rangle$, and ω$_1$mM$_s$')
    #         ax[1].contourf(
    #             output_zonal_gridpoints*grid_scaling,
    #             output_meridional_gridpoints*grid_scaling,
    #             MSE_tendency[t]/COLUMN_AVERAGE_MASS,
    #             cmap=Ahmed_cmap,
    #             norm=mcolors.CenteredNorm(vcenter=0),
    #             levels=21,
    #             # alpha=0.75
    #         )
            
    #         CS_Qr = ax[1].contour(
    #             output_zonal_gridpoints*grid_scaling,
    #             output_meridional_gridpoints*grid_scaling,
    #             column_radiative_heating[t],
    #             colors=bar_colors[4],
    #             levels=np.delete(np.linspace(np.min(column_radiative_heating[t]), np.max(column_radiative_heating[t]), 11), [5])
    #         )
            
    #         CS_omega = ax[1].contour(
    #             output_zonal_gridpoints*grid_scaling,
    #             output_meridional_gridpoints*grid_scaling,
    #             vertical_advection_field[t],
    #             colors=bar_colors[1],
    #             levels=np.delete(np.linspace(np.min(vertical_advection_field[t]), np.max(vertical_advection_field[t]), 11), [5])
    #         )
        
    #         longitude_ticks = np.arange(xlims[0]+xlims[1]/3, xlims[1]+xlims[1]/3, xlims[1]/3)
    #         longitude_labels = mjo.tick_labeller(longitude_ticks, direction='lon')
    #         ax[1].set_xticks(longitude_ticks*METERS_PER_DEGREE*grid_scaling, labels=longitude_labels)
    #         latitude_ticks = np.arange(ylims[0], ylims[1]+ylims[1]/3, ylims[1]/3)
    #         latitude_labels = mjo.tick_labeller(latitude_ticks, direction='lat')
    #         ax[1].set_yticks(latitude_ticks*METERS_PER_DEGREE*grid_scaling, labels=latitude_labels)
    #         ax[1].set_yticklabels('')
    #         ax[1].set_xlim(xlims[0]*METERS_PER_DEGREE*grid_scaling,(xlims[1]-1)*METERS_PER_DEGREE*grid_scaling)
    #         ax[1].set_ylim(ylims[0]*METERS_PER_DEGREE*grid_scaling,ylims[1]*METERS_PER_DEGREE*grid_scaling)
    #         ax[1].set_aspect('auto')
        
    #     # Run the animation
    #     anim = FuncAnimation(
    #         fig, 
    #         update, 
    #         frames=tqdm(
    #             np.arange(starting_frame, ending_frame, frame_interval), 
    #             ncols=100, 
    #             position=0, 
    #             leave=True
    #         ), interval=300
    #     )
        
    #     anim.save(
    #         f"{specified_output_file_directory}/{initial_condition_type}_initial-condition/figures/"
    #         + f"{specified_initial_condition_name}"
    #         + f"_MSE-tendency-structure_animation"
    #         + (f"_{time.strftime('%Y%m%d-%H%M')}" if save_timestamp else '')
    #         + f".mp4", 
    #         dpi=150
    #     )
        
    
    # # Compare across wavenumbers
    
    # # specified_output_file_directory = (
    # #     f"output/Ahmed-21/"
    # #     + f"epst=0.50_epsq=0.17_r=0.2_nx=1.0_ny=1.00" 
    # #     + f"_wavenumber-filtered_gaussian-mean-moisture_non-diffusive-damped-moist-coupled-simulation"
    # # )
    
    # experiment_to_load = specified_output_file_directory
    # initial_conditions_to_load = [
    #     f"k=1.0_m=1_Kelvin-wave", 
    #     f"k=2.0_m=1_Kelvin-wave", 
    #     f"k=3.0_m=1_Kelvin-wave", 
    #     f"k=4.0_m=1_Kelvin-wave",
    #     f"k=5.0_m=1_Kelvin-wave",
    #     f"k=6.0_m=1_Kelvin-wave", 
    # ]
    
    # initial_condition_type = initial_conditions_to_load[0].split('_')[-1]
    
    # south_bound = -90
    # north_bound = 90
    
    # phase_speeds = {}
    # pattern_correlations = {}
    # growth_budgets = {}
    # propagation_budgets = {}
    for ic in initial_conditions_to_load:
        phase_speeds[ic] = xr.load_dataarray(
            f"{specified_output_file_directory}/{initial_condition_type}_initial-condition/"
            + f"{ic}_instantaneous-phase-speed"
            + f"_{mjo.tick_labeller([south_bound], 'lat', False)[0]}-{mjo.tick_labeller([north_bound], 'lat', False)[0]}.nc"
            )
    #     pattern_correlations[ic] = xr.load_dataarray(
    #         f"{specified_output_file_directory}/{initial_condition_type}_initial-condition/"
    #         + f"{ic}_pattern-correlation"
    #         + f"_{mjo.tick_labeller([south_bound], 'lat', False)[0]}-{mjo.tick_labeller([north_bound], 'lat', False)[0]}.nc"
    #         )
    #     growth_budgets[ic] = xr.load_dataset(
    #         f"{specified_output_file_directory}/{initial_condition_type}_initial-condition/"
    #         + f"{ic}_growth-budget"
    #         + f"_{mjo.tick_labeller([south_bound], 'lat', False)[0]}-{mjo.tick_labeller([north_bound], 'lat', False)[0]}.nc"
    #         )
    #     propagation_budgets[ic] = xr.load_dataset(
    #         f"{specified_output_file_directory}/{initial_condition_type}_initial-condition/"
    #         + f"{ic}_propagation-budget"
    #         + f"_{mjo.tick_labeller([south_bound], 'lat', False)[0]}-{mjo.tick_labeller([north_bound], 'lat', False)[0]}.nc"
    #         )
    
    # print("Data loaded")
    
    
    # # Compare budgets
    # # Growth Budgets
    # # Animation
    # starting_frame = day_to_index(2)
    # ending_frame = day_to_index(360)
    # frame_interval = day_to_index(5)
    
    # # Label the bars 
    # bar_labels = [r'Growth', r'ω$_1$mM$_s$', r'σ$_x$u$_1$', r'-σ$_y$yv$_1$', r'$\langle$Q$_r$$\rangle$']
    # bar_colors = ['#ffa500', '#1cf91a', '#0533ff', '#ff40ff', '#4e2d8c']
    
    # wavenumber_max = []
    # wavenumber_min = []
    # for index, k in enumerate(growth_budgets):
    #     wavenumber_max.append(
    #         np.max(
    #             (
    #                 growth_budgets[k]['growth'][starting_frame:ending_frame],
    #                 growth_budgets[k]['omega'][starting_frame:ending_frame],
    #                 growth_budgets[k]['u'][starting_frame:ending_frame],
    #                 growth_budgets[k]['v'][starting_frame:ending_frame],
    #                 growth_budgets[k]['Qr'][starting_frame:ending_frame],
    #             )
    #         )
    #     )
    #     wavenumber_min.append(
    #         np.min(
    #             (
    #                 growth_budgets[k]['growth'][starting_frame:ending_frame],
    #                 growth_budgets[k]['omega'][starting_frame:ending_frame],
    #                 growth_budgets[k]['u'][starting_frame:ending_frame],
    #                 growth_budgets[k]['v'][starting_frame:ending_frame],
    #                 growth_budgets[k]['Qr'][starting_frame:ending_frame],
    #             )
    #         )
    #     )
    
    # grand_max = np.max(wavenumber_max)*SECONDS_PER_DAY
    # grand_min = np.min(wavenumber_min)*SECONDS_PER_DAY
    
    # plt.style.use('default')
    # plt.rcParams.update({'font.size':18})
    # [fig, ax] = plt.subplots(1, 1, figsize=(16.5, 6.5))
    
    # def update(frame):
    #     # Set the bar width
    #     width = 0.1375
    #     multiplier = 0
    #     x = np.arange(len(bar_labels))
    
    #     ax.clear()
        
    #     bar_max = []
    #     for index, k in enumerate(growth_budgets):
    #         offset = width * multiplier
    #         bar_values = [
    #                 SECONDS_PER_DAY*growth_budgets[k]['growth'][frame],
    #                 SECONDS_PER_DAY*growth_budgets[k]['omega'][frame],
    #                 SECONDS_PER_DAY*growth_budgets[k]['u'][frame],
    #                 SECONDS_PER_DAY*growth_budgets[k]['v'][frame],
    #                 SECONDS_PER_DAY*growth_budgets[k]['Qr'][frame]
    #             ]
    #         bar_max.append(np.max(bar_values))
            
    #         rects = ax.bar(
    #             x + offset, 
    #             bar_values, 
    #             label = k, 
    #             color = bar_colors, 
    #             width = width,
    #             edgecolor='gray', 
    #             linewidth=3,
    #             align='center'
    #         )    
    #         # ax.bar_label(rects, padding=0.3)
    #         multiplier += 1
        
    #     for l in range(len(bar_labels)):
    #         # ax.text(l, 1.2*np.max(bar_max), s=bar_labels[l])
    #         ax.text(l, 0.05+1.1*round_out(grand_max, 'tenths'), s=bar_labels[l])
        
    #     ax.axhline(y = 0, color='gray', lw=2)
        
    #     ax.set_title(
    #         f"Growth Budget, Day {growth_budgets[k]['time'][frame]/SECONDS_PER_DAY:0.1f}," 
    #         + f" {mjo.tick_labeller([south_bound], 'lat')[0]}-{mjo.tick_labeller([north_bound], 'lat')[0]}",
    #         pad=40
    #     ) 
    #     ax.set_xticks(
    #         np.concatenate((x, x+offset/5, x+2*offset/5, x+3*offset/5, x+4*offset/5, x+offset)),
    #         sorted(np.arange(len(bar_labels)*len(initial_conditions_to_load))%len(initial_conditions_to_load)+1)
    #     )
    #     ax.set_ylabel(r'day$^{-1}$', rotation=0, labelpad=22)
    #     ax.set_xlabel('Zonal Wavenumber', labelpad=10)
    #     ax.set_ylim(1.1*round_out(grand_min, 'tenths'), 1.1*round_out(grand_max, 'tenths'))
    
    # # Run the animation
    # anim = FuncAnimation(
    #     fig, 
    #     update, 
    #     frames=tqdm(
    #         np.arange(starting_frame, ending_frame, frame_interval), 
    #         ncols=100, 
    #         position=0, 
    #         leave=True
    #     ), interval=300,
    # )
    
    # anim.save(
    #     f"{specified_output_file_directory}/{initial_condition_type}_initial-condition/figures/"
    #     + f"{initial_condition_type}-initial-condition_multi-scale_growth-budget_animation"
    #     + f"_{mjo.tick_labeller([south_bound], 'lat', False)[0]}-{mjo.tick_labeller([north_bound], 'lat', False)[0]}"
    #     + (f"_{time.strftime('%Y%m%d-%H%M')}" if save_timestamp else '')
    #     + f".mp4", 
    #     dpi=150,
    # )
    
    
    # # Propagation Budgets
    # # Animation
    # starting_frame = day_to_index(2)
    # ending_frame = day_to_index(360)
    # frame_interval = day_to_index(5)
    
    # # Label the bars 
    # bar_labels = [r'Propagation', r'ω$_1$mM$_s$', r'σ$_x$u$_1$', r'-σ$_y$yv$_1$', r'$\langle$Q$_r$$\rangle$']
    # bar_colors = ['#ffa500', '#1cf91a', '#0533ff', '#ff40ff', '#4e2d8c']
    
    # wavenumber_max = []
    # wavenumber_min = []
    # for index, k in enumerate(growth_budgets):
    #     wavenumber_max.append(
    #         np.max(
    #             (
    #                 propagation_budgets[k]['propagation'][starting_frame:ending_frame],
    #                 propagation_budgets[k]['omega'][starting_frame:ending_frame],
    #                 propagation_budgets[k]['u'][starting_frame:ending_frame],
    #                 propagation_budgets[k]['v'][starting_frame:ending_frame],
    #                 propagation_budgets[k]['Qr'][starting_frame:ending_frame],
    #             )
    #         )
    #     )
    #     wavenumber_min.append(
    #         np.min(
    #             (
    #                 propagation_budgets[k]['propagation'][starting_frame:ending_frame],
    #                 propagation_budgets[k]['omega'][starting_frame:ending_frame],
    #                 propagation_budgets[k]['u'][starting_frame:ending_frame],
    #                 propagation_budgets[k]['v'][starting_frame:ending_frame],
    #                 propagation_budgets[k]['Qr'][starting_frame:ending_frame],
    #             )
    #         )
    #     )
    
    # grand_max = np.max(wavenumber_max)
    # grand_min = np.min(wavenumber_min)
    
    # plt.style.use('default')
    # plt.rcParams.update({'font.size':18})
    # [fig, ax] = plt.subplots(1, 1, figsize=(16.5, 6.5))
    
    # def update(frame):
    #     # Set the bar width
    #     width = 0.15
    #     multiplier = 0
    #     x = np.arange(len(bar_labels))
    
    #     ax.clear()
        
    #     bar_max = []
    #     for index, k in enumerate(propagation_budgets):
    #         offset = width * multiplier
    #         bar_values = [
    #             propagation_budgets[k]['propagation'][frame],
    #             propagation_budgets[k]['omega'][frame],
    #             propagation_budgets[k]['u'][frame],
    #             propagation_budgets[k]['v'][frame],
    #             propagation_budgets[k]['Qr'][frame]
    #             ]
    #         bar_max.append(np.max(bar_values))
            
    #         rects = ax.bar(
    #             x + offset, 
    #             bar_values, 
    #             label = k, 
    #             color = bar_colors, 
    #             width = width,
    #             edgecolor='gray', 
    #             linewidth=3,
    #             align='center'
    #         )    
    #         # ax.bar_label(rects, padding=0.3)
    #         multiplier += 1
        
    #     for l in range(len(bar_labels)):
    #         # ax.text(l, 1.2*np.max(bar_max), s=bar_labels[l])
    #         ax.text(l, 0.08+1.1*round_out(grand_max, 'tenths'), s=bar_labels[l])
        
    #     ax.axhline(y = 0, color='gray', lw=2)
        
    #     ax.set_title(
    #         f"Propagation Budget, Day {propagation_budgets[k]['time'][frame]/SECONDS_PER_DAY:0.1f}," 
    #         + f" {mjo.tick_labeller([south_bound], 'lat')[0]}-{mjo.tick_labeller([north_bound], 'lat')[0]}",
    #         pad=40
    #     ) 
    #     ax.set_xticks(
    #         # np.concatenate((x, x+offset/3, x+2*offset/3, x+offset)),
    #         np.concatenate((x, x+offset/5, x+2*offset/5, x+3*offset/5, x+4*offset/5, x+offset)),
    #         sorted(np.arange(len(bar_labels)*len(initial_conditions_to_load))%len(initial_conditions_to_load)+1)
    #     )
    #     # ax.set_ylabel(r'day$^{-1}$', rotation=0, labelpad=22)
    #     ax.set_xlabel('Zonal Wavenumber', labelpad=10)
    #     # ax.set_ylim(-2, 2)
    #     ax.set_ylim(1.1*round_out(grand_min, 'tenths'), 1.1*round_out(grand_max, 'tenths'))
    
    # # Run the animation
    # anim = FuncAnimation(
    #     fig, 
    #     update, 
    #     frames=tqdm(
    #         np.arange(starting_frame, ending_frame, frame_interval), 
    #         ncols=100, 
    #         position=0, 
    #         leave=True
    #     ), interval=300,
    # )
    
    # anim.save(
    #     f"{specified_output_file_directory}/{initial_condition_type}_initial-condition/figures/"
    #     + f"{initial_condition_type}-initial-condition_multi-scale_propagation-budget_animation"
    #     + f"_{mjo.tick_labeller([south_bound], 'lat', False)[0]}-{mjo.tick_labeller([north_bound], 'lat', False)[0]}"
    #     + (f"_{time.strftime('%Y%m%d-%H%M')}" if save_timestamp else '')
    #     + f".mp4", 
    #     dpi=150,
    # )

    
    # # Compare phase speeds
    # plt.style.use('bmh')
    # plt.rcParams.update({'font.size':22})
    # [fig, ax] = plt.subplots(1, 1, figsize=(16.5, 6.5))
    # for index, k in enumerate(phase_speeds):
    #     ax.plot(
    #         phase_speeds[k]['time']/SECONDS_PER_DAY, 
    #         phase_speeds[k],
    #         label=f"k = {(index+1):0.1f}", 
    #         lw=3
    #     )
    
    # # ax.set_xlim(10, 170)
    # ax.set_ylim(-2, 20)
    # ax.set_title(
    #     f"Instantaneous Phase Speed over "
    #     + f"{np.abs(phase_speeds[k].attrs['Latitude Bounds'][0])}°S"
    #     + f" - {np.abs(phase_speeds[k].attrs['Latitude Bounds'][1])}°N", 
    #     pad=15
    # )
    # ax.set_xlabel('Day')
    # ax.set_ylabel(r'$\frac{m}{s}$', rotation=0, labelpad=25, fontsize=40)
    # ax.legend(fontsize=16)
    # # plt.show()
    # plt.savefig(
    #       f"{specified_output_file_directory}/{initial_condition_type}_initial-condition/figures/"
    #     + f"{initial_condition_type}-initial-condition_multi-scale_instantaneous-phase-speeds"
    #     + (f"_{time.strftime('%Y%m%d-%H%M')}" if save_timestamp else '')
    #     + f".png", 
    #     bbox_inches='tight'
    # )

    # # Plot dispersion curves
    # mean_freq = []
    # for index, exp in enumerate(phase_speeds):
    #     mean_freq.append(np.mean(phase_speeds[exp])*(index+1)/EARTH_RADIUS)

    # plt.style.use('bmh')
    # plt.rcParams.update({'font.size':18})
    # plt.figure(figsize=(9, 9))
    # plt.plot(
    #     np.array([1,2,3,4,5,6]),
    #     SECONDS_PER_DAY*np.array(mean_freq), marker='o'
    # )

    # A21_frequencies = [0.103609, 0.181382, 0.265887, 0.356728, 0.453978]
    # plt.plot([1,2,3,4,5], A21_frequencies, label='A21', marker='o', color='red')
    # plt.xlabel('Wavenumber')
    # plt.xticks(ticks=np.array([1,2,3,4,5,6]), labels=np.array([1,2,3,4,5,6]))
    # plt.ylabel(r'day$^{-1}$')
    # plt.ylim(0, 0.5)
    # plt.yticks(ticks=np.arange(0, 0.6, 0.1))
    # plt.legend(loc='upper left')
    # plt.savefig(
    #       f"{specified_output_file_directory}/{initial_condition_type}_initial-condition/figures/"
    #     + f"{initial_condition_type}-initial-condition_multi-scale_dispersion-curve-plot"
    #     + (f"_{time.strftime('%Y%m%d-%H%M')}" if save_timestamp else '')
    #     + f".png", 
    #     bbox_inches='tight'
    # )
    
    # plt.style.use('bmh')
    # plt.rcParams.update({'font.size':22})
    # [fig, ax] = plt.subplots(1, 1, figsize=(16.5, 6.5))
    # for index, exp in enumerate(phase_speeds):
    #     ax.plot(
    #     pattern_correlations[exp]['time']/SECONDS_PER_DAY, 
    #     pattern_correlations[exp],
    #     label=f"k = {(index+1):0.1f}", 
    #     lw=3
    #     )
    
    # # ax.set_xlim(10, 100)
    # ax.set_ylim(-1, 1.05)
    # ax.set_title(
    #     f"Pattern correlation over "
    #     + f"{np.abs(pattern_correlations[exp].attrs['Latitude Bounds'][0])}°S"
    #     + f" - {np.abs(pattern_correlations[exp].attrs['Latitude Bounds'][1])}°N", 
    #     pad=15
    # )
    # ax.set_xlabel('Day')
    # # ax.set_ylabel(r'$\frac{m}{s}$', rotation=0, labelpad=25, fontsize=40)
    # ax.legend()
    # # plt.show()
    # plt.savefig(
    #       f"{specified_output_file_directory}/{initial_condition_type}_initial-condition/figures/"
    #     + f"{initial_condition_type}-initial-condition_multi-scale_pattern-correlation"
    #     + (f"_{time.strftime('%Y%m%d-%H%M')}" if save_timestamp else '')
    #     + f".png", 
    #     bbox_inches='tight'
    # )
