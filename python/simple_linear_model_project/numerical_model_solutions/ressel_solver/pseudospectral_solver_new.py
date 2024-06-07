#!/usr/bin/env python
# coding: utf-8

# # Documentation
# 
# **Author:** Spencer Ressel
# 
# **Created:** June 14th, 2023
# 
# ---
# 
# This code numerically solves the governing equations from Matsuno (1966). It was initially written by Daniel Lloveras as a project for the course ATM S 582.  The solver uses the pseudospectral method with leapfrog time differencing to solve the equations of motion.
# 
# ---

# Imports
print(f"{'':{'='}^{50}}")
print("Importing modules...")
import os
os.chdir(f"/home/disk/eos7/sressel/research/thesis-work/python/numerical_solver/")
import sys
import time
import json
from glob import glob
import numpy as np
import xarray as xr
from config import *
from dynamical_cores import dry_Kelvin_wave_dynamical_core, dry_Matsuno_dynamical_core, Ahmed21_dynamical_core
from compute_derivatives import compute_meridional_derivative_reflected, compute_meridional_derivative_finite_difference

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':22})
from tqdm import tqdm

# Auxiliary Functions
import sys
sys.path.insert(0, '/home/disk/eos7/sressel/research/thesis-work/python/auxiliary_functions/')
import ipynb.fs.full.mjo_mean_state_diagnostics as mjo
from ipynb.fs.full.bmh_colors import bmh_colors
from ipynb.fs.full.rounding_functions import round_out, round_to_multiple
from ipynb.fs.full.modified_colormap import Ahmed21_colormap
Ahmed_cmap = Ahmed21_colormap()
from ipynb.fs.full.tapering_functions import taper_meridionally, fringe_region_damping_function
from ipynb.fs.full.normalize_data import normalize_data

sys.path.insert(0, '/home/disk/eos7/sressel/research/thesis-work/python/numerical_solver/')
from ipynb.fs.full.numerical_solver_plotting_functions import plot_horizontal_structure, animate_horizontal_structure

print("Imports complete")
print(f"{'':{'='}^{50}}")

# Specify Physical Constants
simulation_moisture = False
moisture_advection = False
simulation_damping = False
moisture_coupling = False
simulation_diffusion = False
fringe_region = False
wavenumber_filtering = False
rayleigh_friction = False

dynamical_core = 'dry-Kelvin'
# moisture_sensitivity_structure = 'logistic'
# moisture_sensitivity_structure = 'gaussian'
# temperature_sensitivity_structure = 'logistic'
moisture_sensitivity_structure = 'constant'
temperature_sensitivity_structure = 'constant'
mean_moisture_profile = ''
# mean_moisture_profile = 'asymmetric-gaussian'
# mean_moisture_profile = 'quadratic'
# moisture_stratification_structure = 'gaussian'
moisture_stratification_structure = 'constant'
zonal_moistening_zonal_structure = 'constant'

# Implement diffusion
DIFFUSIVITY = 0.2                      # D   [m^2 s^-1]
laplacian_u = 0
laplacian_v = 0
laplacian_T = 0
laplacian_q = 0

##########################

#### Ahmed (2021) constants ####
MOISTURE_SENSITIVITY = (1/(6*3600) if simulation_moisture else 0)  # ε_q [s^-1]
TEMPERATURE_SENSITIVITY = (1/(2*3600) if simulation_damping else 0)   # ε_t [s^-1]
CLOUD_RADIATIVE_PARAMETER = (0.2 if simulation_damping else 0)   # r [-]
# TEMPERATURE_SENSITIVITY = (0 if simulation_damping else 0)   # ε_t [s^-1]
# CLOUD_RADIATIVE_PARAMETER = 0.21*np.exp(-242614*int(sys.argv[1])/EARTH_RADIUS)

# sigma_x_multiplier = float(sys.argv[2])
sigma_x_multiplier = (1 if simulation_moisture and moisture_advection else 0) # n_σ_x[-]
sigma_y_multiplier = (1 if simulation_moisture and moisture_advection else 0) # n_σ_y[-]

ZONAL_MOISTENING_PARAMETER = 5e-4*sigma_x_multiplier        # σ_x [K kg m^-3]
if mean_moisture_profile == 'quadratic':
    MERIDIONAL_MOISTENING_PARAMETER = 9e-9*sigma_y_multiplier   # σ_y [K kg m^-4]
else:
    MERIDIONAL_MOISTENING_PARAMETER = 6.06e-9*sigma_y_multiplier   # σ_y [K kg m^-4]
MERIDIONAL_OFFSET_PARAMETER = 0*METERS_PER_DEGREE           # δ_y [m]

sensitivity_limit = 90
sensitivity_width = 0

if (moisture_sensitivity_structure == 'logistic') or (temperature_sensitivity_structure == 'logistic'):
    sensitivity_limit = 30
    sensitivity_width = 25
    # sensitivity_limit = 25
    # sensitivity_width = 15

RAYLEIGH_FRICTION_COEFFICIENT = (1/(10*SECONDS_PER_DAY) if rayleigh_friction else 0)
##############################################################################

################################ Derived quantities ##################################
effective_sensitivity = (
    MOISTURE_SENSITIVITY 
    + TEMPERATURE_SENSITIVITY*(1+CLOUD_RADIATIVE_PARAMETER)
)                                                                    # ε_a [s^-1]
effective_gross_moist_stability = (
    (GROSS_DRY_STABILITY - gross_moisture_stratification)/GROSS_DRY_STABILITY
)*(1+CLOUD_RADIATIVE_PARAMETER) - CLOUD_RADIATIVE_PARAMETER          # m_eff [-]
scaled_zonal_parameter = ZONAL_MOISTENING_PARAMETER/GROSS_DRY_STABILITY # ξ_x
scaled_meridional_parameter = MERIDIONAL_MOISTENING_PARAMETER/GROSS_DRY_STABILITY # ξ_y


print(f"{'Important Parameter Values':^60}")
print(f"{'':=^60}")
print(f"Dry Gravity Wave Phase Speed:     {gravity_wave_phase_speed:>10.2f} m/s")
print(f"Time Scale:                       {time_scale*24/SECONDS_PER_DAY:>10.2f} hours")
print(f"Length Scale:                     {length_scale/1e3:>10.2f} km")
print(f"")
print(f"Damping: {simulation_damping}")
print(f"- Temperature Sensitivity:        {3600*TEMPERATURE_SENSITIVITY:>10.2f} hr^-1")
print(f"- Cloud-Radiative Parameter:      {CLOUD_RADIATIVE_PARAMETER:>10.2f}")
print(f"")
print(f"Moisture: {simulation_damping}")
print(f"- Mean Profile:                   {mean_moisture_profile:>14}")
print(f"- Moisture Sensitivity:           {3600*MOISTURE_SENSITIVITY:>10.2f} hr^-1")
print(f"- Zonal Moistening Parameter:     {sigma_x_multiplier:>10.2f} x {ZONAL_MOISTENING_PARAMETER:>0.1e}")
print(f"- Meridional Moistening Parameter:{sigma_y_multiplier:>10.2f} x {MERIDIONAL_MOISTENING_PARAMETER:>0.1e}")
print(f"- Meridional Offset Parameter:    {MERIDIONAL_OFFSET_PARAMETER/1e3:>10.2f} km")

print(f"{'':=^60}")
######################################################################################

########################### Define simiulaton grid ############################
#### Standard simulation
# n_days = 1.157                                               # number of days in simulation
n_days = 10                                               # number of days in simulation
simulation_length = n_days*SECONDS_PER_DAY                   # simulation length in seconds
n_chunks = 1                                                 # number of chunks over which the time stepping will be split

time_step = 500                                              # time step in seconds 
n_time_steps = int(simulation_length/time_step + 1)

# n_time_steps             = int(1.2*2**12)                    # number of time steps
# n_time_steps             = int(1.2*2**10)                    # number of time steps
# n_time_steps = 5
# n_time_steps = 200
meridional_domain_length = 7000e3                            # length of half y domain in m
zonal_domain_length      = 2*np.pi*EARTH_RADIUS              # length of x domain in m

nx = 257                                                     # number of steps +1 in the zonal grid
ny = 129                                                     # number of steps +1 in the meridional grid
# nx = 513
# ny = 257

zonal_grid_spacing = zonal_domain_length/(nx-1)              # spacing between zonal grid points in m
meridional_grid_spacing = 2*meridional_domain_length/(ny-1)  # spacing between meridional grid points in m



# Define the temporal grid points
if n_chunks == 1:
    time_points = np.linspace(
        0, 
        simulation_length, 
        n_time_steps
    )                                                         # Array of simulation time points  
    
else:
    time_points = np.linspace(
        0,
        simulation_length,
        n_chunks*n_time_steps
    )

time_step = np.diff(time_points)[0]                           # Length of a time step in s

# Define the spatial grid points
zonal_gridpoints = np.arange(-(nx-1)/2, (nx-1)/2, 1)*zonal_grid_spacing
meridional_gridpoints = np.arange(-(ny-1)/2, (ny-1)/2, 1)*meridional_grid_spacing + meridional_grid_spacing/2

# Redefine nx, ny, based on actual grid
nt = len(time_points)                            # number of time steps
ny = len(meridional_gridpoints)                  # number of zonal grid points
nx = len(zonal_gridpoints)                       # number of meridional grid points

zonal_step_size = np.diff(zonal_gridpoints)[0]
meridional_step_size = np.diff(meridional_gridpoints)[0]

# Define Fourier arrays 
zonal_wavenumber      = 2*np.pi*np.fft.fftfreq(nx, zonal_step_size)       # zonal wavenumbers
meridional_wavenumber = 2*np.pi*np.fft.fftfreq(ny, meridional_step_size)  # meridional wavenumbers
frequencies           = 2*np.pi*np.fft.fftfreq(nt, time_step)             # frequencies

# Calculate CFL condition
CFL_x = gravity_wave_phase_speed*time_step/zonal_step_size
CFL_y = gravity_wave_phase_speed*time_step/meridional_step_size

# Create a fringe region
fringe_region_latitude = 30
fringe_region_width = 5
fringe_region_strength = 0.000

if fringe_region:
    fringe_region_strength = 0.003
    
fringe_region_damping = fringe_region_damping_function(
    meridional_gridpoints/METERS_PER_DEGREE,
    -fringe_region_latitude, 
    fringe_region_latitude, 
    fringe_region_width, 
    fringe_region_strength
)

print(f"{'Simulation Parameters':^48}")
print(f"{'':=^48}")
print(
    f"{'Lx =':4}" + 
    f"{zonal_domain_length/1e3:>6.0f}{' km':<6}" + 
    f"{'| Δx = ':>5}" + 
    f"{zonal_step_size/1e3:>8.1f}" + 
    f"{' km':<5}" + 
    f"{'| nx = ':<5}" + 
    f"{nx:>5.0f}"
)
print(
    f"{'Ly =':4}" + 
    f"{2*meridional_domain_length/1e3:>6.0f}" + 
    f"{' km':<6}{'| Δy = ':>5}" + 
    f"{meridional_step_size/1e3:>8.1f}" + 
    f"{' km':<5}" + 
    f"{'| ny = ':<4}" + 
    f"{ny:>5.0f}"
)
print(
    f"{'T  =':4}" + 
    f"{simulation_length/SECONDS_PER_DAY:>6.0f}" + 
    f"{' days':<6}{'| Δt = ':>5}" + 
    f"{time_step:>8.1f}" + 
    f"{' sec':<5}" + 
    f"{'| nt = ':<5}" + 
    f"{nt:>5.0f}"
)
print(f"{'':=^48}")

print(f"CFL = {CFL_x + CFL_y:0.3f}", end="")
if (CFL_x + CFL_y < 1):
    print(", numerically stable ✔")
    
else:
    print(" > 1, numerically unstable!!")
    
print(f"{'':=^48}")
###########################################################################################
# Document simulation parameters

# additional_notes = '_zonally-varying-zonal-advection'
additional_notes = 'test_'

#### Document simulation details ####
os.chdir("/home/disk/eos7/sressel/research/thesis-work/python/numerical_solver/")
simulation_name = (
    (f"epst{temperature_sensitivity_structure}={3600*TEMPERATURE_SENSITIVITY:0.2f}" if simulation_damping else '')
  + (f"_t-lat={sensitivity_limit}" if temperature_sensitivity_structure == 'logistic' else '')
  + (f"_t-width={sensitivity_width}" if temperature_sensitivity_structure == 'logistic' else '')
  + (f"_epsq{moisture_sensitivity_structure}={3600*MOISTURE_SENSITIVITY:0.2f}" if simulation_moisture else '')
  + (f"_q-lat={sensitivity_limit}" if moisture_sensitivity_structure == 'logistic' else '')
  + (f"_q-width={sensitivity_width}" if moisture_sensitivity_structure == 'logistic' else '')
  + (f"_r={CLOUD_RADIATIVE_PARAMETER:0.1f}" if simulation_damping else '')
  + (f"_nx={sigma_x_multiplier:0.1f}" if moisture_advection else '') 
  + (f"_ny={sigma_y_multiplier:0.2f}" if moisture_advection else '')
  + (f"_eps-rayleigh={RAYLEIGH_FRICTION_COEFFICIENT*SECONDS_PER_DAY:0.1f}" if rayleigh_friction else '')
  + f"{additional_notes}"
  + (f"_{'fringe-region'}" if fringe_region else '')
  + (f"_{'wavenumber-filtered'}" if wavenumber_filtering else '')
  + (f"_{mean_moisture_profile}-mean-moisture" if simulation_moisture else '')
  + ("_diffusive" if simulation_diffusion else "")
  + ("_damped" if simulation_damping else "free")
  + ("_moist" if simulation_moisture else "_dry")
  + ("_coupled" if moisture_coupling else "_uncoupled")
  + "-simulation"
)

output_file_directory = f"output/{dynamical_core}/{simulation_name}"
if not os.path.exists(output_file_directory):
    os.makedirs(output_file_directory, exist_ok=True)
    print(f"Output folder created")
else:
    print(f"Output folder already created")
print(f"Simulation details: {simulation_name}")


# Export experiment variables
# Get all local variables in the current namespace
experiment_variables = dict(
    dynamical_core = dynamical_core,
    simulation_moisture = simulation_moisture,
    moisture_advection = moisture_advection,
    simulation_damping = simulation_damping,
    moisture_coupling = moisture_coupling,
    simulation_diffusion = simulation_diffusion,
    fringe_region = fringe_region,
    rayleigh_friction = rayleigh_friction,
    moisture_sensitivity_structure = moisture_sensitivity_structure,
    temperature_sensitivity_structure = temperature_sensitivity_structure,
    moisture_stratification_structure = moisture_stratification_structure,
    zonal_moistening_zonal_structure = zonal_moistening_zonal_structure,
    sensitivity_limit = sensitivity_limit,
    sensitivity_width = sensitivity_width,
    mean_moisture_profile = mean_moisture_profile,
    # moisture_length_scale = (gaussian_length_scale if mean_moisture_profile == 'gaussian' else quadratic_width),
    moisture_length_scale = 0,
    GRAVITY = GRAVITY,
    EQUIVALENT_DEPTH = EQUIVALENT_DEPTH,
    CORIOLIS_PARAMETER = CORIOLIS_PARAMETER,
    EARTH_RADIUS = EARTH_RADIUS,
    AIR_DENSITY = AIR_DENSITY,
    WATER_DENSITY = WATER_DENSITY,
    LATENT_HEAT = LATENT_HEAT,
    SPECIFIC_HEAT = SPECIFIC_HEAT,
    DIFFUSIVITY = DIFFUSIVITY,
    METERS_PER_DEGREE = METERS_PER_DEGREE,
    SECONDS_PER_DAY = SECONDS_PER_DAY,
    COLUMN_AVERAGE_MASS = COLUMN_AVERAGE_MASS,
    GROSS_DRY_STABILITY = GROSS_DRY_STABILITY,
    MOISTURE_SENSITIVITY = MOISTURE_SENSITIVITY,
    TEMPERATURE_SENSITIVITY = TEMPERATURE_SENSITIVITY,
    CLOUD_RADIATIVE_PARAMETER = CLOUD_RADIATIVE_PARAMETER,
    RAYLEIGH_FRICTION_COEFFICIENT = RAYLEIGH_FRICTION_COEFFICIENT,
    sigma_x_multiplier = sigma_x_multiplier,
    sigma_y_multiplier = sigma_y_multiplier,
    ZONAL_MOISTENING_PARAMETER = ZONAL_MOISTENING_PARAMETER,
    MERIDIONAL_MOISTENING_PARAMETER = MERIDIONAL_MOISTENING_PARAMETER,
    MERIDIONAL_OFFSET_PARAMETER = MERIDIONAL_OFFSET_PARAMETER,
    # MERIDIONAL_OFFSET_PARAMETER = gaussian_meridional_offset,
    gravity_wave_phase_speed = gravity_wave_phase_speed,
    time_scale = time_scale,
    length_scale = length_scale,
    gross_moisture_stratification = gross_moisture_stratification,
    effective_sensitivity = effective_sensitivity,
    effective_gross_moist_stability = effective_gross_moist_stability,
    scaled_zonal_parameter = scaled_zonal_parameter,
    scaled_meridional_parameter = scaled_meridional_parameter,
    n_days = n_days,
    n_chunks = n_chunks,
    n_time_steps = n_time_steps,
    meridional_domain_length = meridional_domain_length,
    zonal_domain_length = zonal_domain_length,
    nt = nt,    
    nx = nx,
    ny = ny,
    zonal_grid_spacing = zonal_grid_spacing,
    meridional_grid_spacing = meridional_grid_spacing,
    simulation_length = simulation_length,
    time_step = time_step,
    zonal_step_size = zonal_step_size,
    meridional_step_size = meridional_step_size,
    CFL_x = CFL_x,
    CFL_y = CFL_y,
    fringe_region_latitude = fringe_region_latitude,
    fringe_region_width = fringe_region_width,
    fringe_region_strength = fringe_region_strength,
    grid_scaling = 1e-6,
    additional_notes = additional_notes,
    simulation_name = simulation_name,
    output_file_directory = output_file_directory,
    # n_rk_steps = n_rk_steps,
    # save_downsampled = save_downsampled,
    save_time = time.strftime('%Y%m%d-%H%M'),
)

# Save to a JSON file

print(f"Saving experiment variables as JSON")
print(f"===================================")
with open(
    f"{output_file_directory}/experiment_variables.json", 'w') as json_file:
    
        if os.path.exists(f"{output_file_directory}/experiment_variables.json"):
            print(f"JSON already exists, overwriting...")
        else:
            print(f"Creating JSON...")
        json.dump(experiment_variables, json_file)
    
print(f"===================================")
print("Experiment variables saved")


# # Estimate Gaussian Mean Moisture profile
# # Load ERA5 CWV data
# TCWV_data = xr.load_dataset('/home/disk/eos7/sressel/research/data/ECMWF/ERA5/monthly_reanalysis_CWV_CIT_2002_2014.nc')
# total_column_water_vapor = TCWV_data['tcwv'].sel(latitude=slice(90,-90))
# era5_latitude = TCWV_data['latitude'].sel(latitude=slice(90,-90))
# era5_longitude = TCWV_data['longitude']
# era5_time = TCWV_data['time']


# # Fit Gaussian to data
# from scipy.optimize import curve_fit 

# # Define the column integrated moisture profile <b>
# column_integrated_moisture_profile = 54.6

# # Define the product of the column integrated velocity and moisture profiles <Vb>
# column_integrated_velocity_moisture_product = -19.4

# avg_lons = [60,160]
# avg_lats = [60,-60]

# # Calculate the time and zonal mean CWV from ERA5
# era5_mean_moisture_profile = total_column_water_vapor.sel(
#     longitude=slice(avg_lons[0],avg_lons[1]),
#     latitude=slice(avg_lats[0], avg_lats[1])
# ).mean(dim=['longitude', 'time'])

# era5_subset_latitudes = era5_latitude.sel(latitude=slice(avg_lats[0], avg_lats[1]))

# # Define the Gaussian fitting profile
# def Gauss(x, A, B, C, D): 
#     y = A + B*np.exp(-1*((x-D)/C)**2) 
#     return y 

# # Solve for the best fit 
# parameters, covariance = curve_fit(
#     Gauss, 
#     era5_subset_latitudes.values, 
#     era5_mean_moisture_profile.values
# ) 

# gaussian_vertical_offset = parameters[0]/column_integrated_moisture_profile

# gaussian_magnitude = parameters[1]/column_integrated_moisture_profile

# # The length scale of the Gaussian fit
# gaussian_length_scale = parameters[2]*METERS_PER_DEGREE  
# # gaussian_length_scale *= 0.8
# # The meridional offset of the CWV relative to the equator
# gaussian_meridional_offset = parameters[3]*METERS_PER_DEGREE

# # Calculate the fitted moisture profile
# gaussian_mean_moisture_profile = Gauss(
#     era5_subset_latitudes.values*METERS_PER_DEGREE, 
#     gaussian_vertical_offset*column_integrated_moisture_profile,
#     gaussian_magnitude*column_integrated_moisture_profile, 
#     gaussian_length_scale, 
#     # gaussian_meridional_offset,
#     0
# ) 

# def Quadratic(x, A, B, C):
#     y = A - 0.5*B*(x-C)**2
#     return y

# quadratic_parameters, quadratic_covariance = curve_fit(
#     Quadratic, 
#     era5_subset_latitudes.sel(latitude=slice(20,-20)).values*METERS_PER_DEGREE, 
#     era5_mean_moisture_profile.sel(latitude=slice(20,-20)).values
# ) 
# quadratic_vertical_offset = quadratic_parameters[0]/column_integrated_moisture_profile
# quadratic_width = quadratic_parameters[1]/column_integrated_moisture_profile
# quadratic_meridional_offset = quadratic_parameters[2]/column_integrated_moisture_profile

# quadratic_mean_moisture_profile = Quadratic(
#     era5_subset_latitudes.sel(latitude=slice(60,-60)).values*METERS_PER_DEGREE, 
#     quadratic_vertical_offset*column_integrated_moisture_profile,
#     quadratic_width*column_integrated_moisture_profile,
#     # quadratic_meridional_offset*column_integrated_moisture_profile
#     0
# )

# if mean_moisture_profile == 'gaussian':
#     MERIDIONAL_MOISTENING_PARAMETER = (
#         -column_integrated_velocity_moisture_product*2*gaussian_magnitude/(gaussian_length_scale**2)*LATENT_HEAT/SPECIFIC_HEAT
#     )
#     print(f"σ_y = {MERIDIONAL_MOISTENING_PARAMETER:0.3e}")

# # # Plot the mean moisture profile and fitted profiles
# plt.figure(figsize=(8.5,6.5), dpi=500)
# plt.plot(
#     era5_subset_latitudes*METERS_PER_DEGREE/1e3, 
#     era5_mean_moisture_profile.values,
#     color='red',
#     label='ERA5'
# ) 
# plt.plot(
#     era5_subset_latitudes*METERS_PER_DEGREE/1e3, 
#     gaussian_mean_moisture_profile,
#     color='k', 
#     ls='-', 
#     label='Gaussian'
# ) 
# plt.plot(
#     era5_subset_latitudes.sel(latitude=slice(60,-60))*METERS_PER_DEGREE/1e3, 
#     quadratic_mean_moisture_profile, 
#     color='black', 
#     ls='--', 
#     lw=2,
#     label='A21 Quadratic'
# ) 
# # plt.plot(
# #     meridional_gridpoints/1e3, 
# #      (
# #          column_integrated_moisture_profile*(gaussian_vertical_offset + gaussian_magnitude*np.exp(-(meridional_gridpoints/gaussian_length_scale)**2))
# #      ), 
# #     color='k',
# #     label='Idealized'
# # )
# plt.axvline(x=0, ls=':', color='k', alpha=0.5)

# plt.xticks(
#     ticks=np.arange(-60*METERS_PER_DEGREE/1e3, 90*METERS_PER_DEGREE/1e3, 30*METERS_PER_DEGREE/1e3), 
#     labels=mjo.tick_labeller(np.arange(-60, 90, 30), 'lat')
# )
# # plt.xlabel('km')

# plt.ylim(0, 60)

# plt.ylabel('mm', rotation=0, va='center', labelpad=35)
# plt.legend(loc='best', fontsize=14)
# plt.gca().set_aspect('auto')
# plt.show()

gaussian_fitting_longitudes =  [60,160]
gaussian_fitting_latitudes = [60,-60]

# Initial Conditions
# Initial Condition Function
def generate_wavelike_initial_condition(
    initial_wave,
    n_wavelengths = 2,
    save_initial_condition=False,
    
):
    
    """
    This function generates data for the initial two time steps required to perform leap-frog numerical integration. 
    Options include wave type, number of wavelengths in the zonal domain, whether to include damping, moisture, and moisture 
    coupling in the simulation, and whether or not to save the output data to a netCDF file. 
    
    Keyword arguments:
    initial_wave (str)            : The wave type to be used in the initial condition. 
                                    Options are 'Kelvin', 'Rossby', 'EIG', or 'WIG'
    n_wavelengths (int)           : The number of wavelengths in the zonal direction (default 1) 
    simulation_damping (bool)     : Whether the simulation is free or damped (default False)
    simulation_moisture (bool)    : Whether the simulation is dry or includes moisture (default False)
    moisture_coupling (bool)      : Whether the moisture in the simulation feeds back to 
                                    affect the dry variables (default False)
    save_initial_condition (bool) : Whether or not the initial condition data is saved to a netCDF file (default False)
    
    """
    initial_wavenumber = 2*np.pi*n_wavelengths/zonal_domain_length
    initial_temperature_anomaly = 0.1*COLUMN_AVERAGE_MASS/1.6

    #### Initialize arrays
    zonal_velocity = np.zeros((2,ny,nx))
    meridional_velocity = np.zeros((2,ny,nx))
    column_temperature = np.zeros((2,ny,nx))
    column_moisture = np.zeros((2,ny,nx))

    # Generate data arrays 
    if initial_wave == 'Kelvin-wave':

        # u(x,y,t=0) = c × ψ(y/L, 1) × e^(ikx) [m s^-1]
        zonal_velocity[0] = gravity_wave_phase_speed*np.real(
            np.einsum(
                'i,j->ij',
                mjo.parabolic_cylinder_function(meridional_gridpoints/length_scale, 0),
                np.exp(1j*initial_wavenumber*zonal_gridpoints)
            )
        )

        # <T>(x,y,t=0)
        column_temperature[0] = (GROSS_DRY_STABILITY/gravity_wave_phase_speed**2)*gravity_wave_phase_speed**2*np.real(
            np.einsum(
                'i,j->ij',
                mjo.parabolic_cylinder_function(meridional_gridpoints/length_scale, 0),
                np.exp(1j*initial_wavenumber*zonal_gridpoints)
            )
        )

        if simulation_moisture == True:
            column_moisture[0] = -(
                gross_moisture_stratification/gravity_wave_phase_speed**2
            )*gravity_wave_phase_speed**2*np.real(
                np.einsum(
                    'i,j->ij',
                    mjo.parabolic_cylinder_function(meridional_gridpoints/length_scale, 0),
                    np.exp(1j*initial_wavenumber*zonal_gridpoints)
                )
            )

        # Rescale the anomalies so that ϕ[0] has magnitude 'initial_geopotential_anomaly'
        zonal_velocity[0] *= initial_temperature_anomaly/np.max(column_temperature[0])
        column_moisture[0] *= initial_temperature_anomaly/np.max(column_temperature[0])
        column_temperature[0] *= initial_temperature_anomaly/np.max(column_temperature[0])


    else:
        if initial_wave == 'Rossby-wave':
            initial_frequency = (
                CORIOLIS_PARAMETER*initial_wavenumber
                /(initial_wavenumber**2 + (CORIOLIS_PARAMETER/gravity_wave_phase_speed)*(2*mode_number+1))
            )

        elif initial_wave == 'EIG-wave':
            initial_frequency = -gravity_wave_phase_speed*np.sqrt(
                    initial_wavenumber**2 + (CORIOLIS_PARAMETER/gravity_wave_phase_speed)*(2*mode_number + 1)
                )

        elif initial_wave == 'WIG-wave':
            initial_frequency = gravity_wave_phase_speed*np.sqrt(
                    initial_wavenumber**2 + (CORIOLIS_PARAMETER/gravity_wave_phase_speed)*(2*mode_number + 1)
                )


        # v(x,y,t=0) = i(1/ß)(ω^2-(ck)^2) × ψ(y/L, m) × e^(ikx) [m s^-1]
        meridional_velocity[0] = np.real(
                np.einsum(
                    'i,j->ij',
                    (
                        1j*(initial_frequency**2 - gravity_wave_phase_speed**2*initial_wavenumber**2)*(1/CORIOLIS_PARAMETER)
                            *mjo.parabolic_cylinder_function(meridional_gridpoints/length_scale, mode_number)
                    ),
                    np.exp(1j*initial_wavenumber*zonal_gridpoints)
                )
            )

        # u(x,y,t=0) = L × (0.5(ω-ck)ψ(y/L, m+1) + m(ω+ck)ψ(y/L, m-1)) × e^(ikx) [m s^-1]
        zonal_velocity[0] = (gravity_wave_phase_speed/CORIOLIS_PARAMETER)**(1/2)*np.real(
                np.einsum(
                    'i,j->ij',
                    (
                        0.5*(initial_frequency - gravity_wave_phase_speed*initial_wavenumber)
                            *mjo.parabolic_cylinder_function(meridional_gridpoints/length_scale, mode_number + 1)
                        + mode_number*(initial_frequency + gravity_wave_phase_speed*initial_wavenumber)
                            *mjo.parabolic_cylinder_function(meridional_gridpoints/length_scale, mode_number - 1)
                    ),
                    np.exp(1j*initial_wavenumber*zonal_gridpoints)
                )
            )

        # <T>(x,y,t=0) =  (M_s/c) x L × (0.5(ω-ck)ψ(y/L, m+1) - m(ω+ck)ψ(y/L, m-1)) × e^(ikx) [K kg m^-2]
        column_temperature[0] = (
            GROSS_DRY_STABILITY/gravity_wave_phase_speed
        )*(gravity_wave_phase_speed/CORIOLIS_PARAMETER)**(1/2)*np.real(
                np.einsum(
                    'i,j->ij',
                    (
                        0.5*(initial_frequency - gravity_wave_phase_speed*initial_wavenumber)
                            *mjo.parabolic_cylinder_function(meridional_gridpoints/length_scale, mode_number + 1)
                        - mode_number*(initial_frequency + gravity_wave_phase_speed*initial_wavenumber)
                            *mjo.parabolic_cylinder_function(meridional_gridpoints/length_scale, mode_number - 1)
                    ),
                    np.exp(1j*initial_wavenumber*zonal_gridpoints)
                )
            )

        if simulation_moisture == True:
            column_moisture[0] = -(
                gross_moisture_stratification/gravity_wave_phase_speed
            )*(gravity_wave_phase_speed/CORIOLIS_PARAMETER)**(1/2)*np.real(
                    np.einsum(
                        'i,j->ij',
                        (
                            0.5*(initial_frequency - gravity_wave_phase_speed*initial_wavenumber)
                                *mjo.parabolic_cylinder_function(meridional_gridpoints/length_scale, mode_number + 1)
                            - mode_number*(initial_frequency + gravity_wave_phase_speed*initial_wavenumber)
                                *mjo.parabolic_cylinder_function(meridional_gridpoints/length_scale, mode_number - 1)
                        ),
                        np.exp(1j*initial_wavenumber*zonal_gridpoints)
                    )
                )

        # Rescale the anomalies so that ϕ[0] has magnitude 'initial_geopotential_anomaly'
        zonal_velocity[0] *= initial_temperature_anomaly/np.max(column_temperature[0])
        meridional_velocity[0] *= initial_temperature_anomaly/np.max(column_temperature[0])
        column_moisture[0] *= initial_temperature_anomaly/np.max(column_temperature[0])
        column_temperature[0] *= initial_temperature_anomaly/np.max(column_temperature[0])

    if save_initial_condition == True:
        print("Attempting to save data...")
        initial_condition_data = xr.Dataset(
            data_vars = {
                'u' : (["it", "y", "x"], zonal_velocity[-2:]),
                'v' : (["it", "y", "x"], meridional_velocity[-2:]),
                'T' : (["it", "y", "x"], column_temperature[-2:]),
                'q' : (["it", "y", "x"], column_moisture[-2:])
            },
            coords = {
            "x" : zonal_gridpoints,
            "y" : meridional_gridpoints,
            "it": np.array([0,1])
            }
        )
        
        print(
            f"Data to be saved as:\n{output_file_directory}/{initial_wave}_initial-condition/"
              + f"{initial_condition_name}_initial-condition-data.nc"
        )
        
        # Check if the data already exists
        if os.path.exists(
            f"{output_file_directory}/{initial_wave}_initial-condition/"
            + f"{initial_condition_name}_initial-condition-data.nc"
        ):
            
            # Prompt the user to overwrite the existing data
            # overwrite_data = input("Data already exists. Overwrite? [y/n]")
            overwrite_data = 'y'
            if overwrite_data == 'y':
                os.system(
                    f"rm {output_file_directory}/{initial_wave}_initial-condition/"
                    + f"{initial_condition_name}_initial-condition-data.nc"
                )
                print("Data overwritten")
                initial_condition_data.to_netcdf(
                    f"{output_file_directory}/{initial_wave}_initial-condition/"
                    + f"{initial_condition_name}_initial-condition-data.nc"
                )
                print("Initial condition data saved as netCDF")
            else:
                print("Original data retained")
        else:
            initial_condition_data.to_netcdf(
                f"{output_file_directory}/{initial_wave}_initial-condition/"
                + f"{initial_condition_name}_initial-condition-data.nc"
            )
            print("Initial condition data saved as netCDF")
    
    return zonal_velocity, meridional_velocity, column_temperature, column_moisture


# Specify initial conditions
# Specify the type of wave for the initial condition
# initial_condition_type = 'rolled-moisture-blob'
initial_condition_type = 'Kelvin-wave'
# n_wavelengths = 2
# for n_wavelengths in [1,2,3,4,5,6]:
n_wavelengths = int(sys.argv[1])
mode_number = 1
initial_condition_name = f"k={n_wavelengths:0.1f}_m={mode_number}_{initial_condition_type}"

print(f"Output directory:\n{output_file_directory}")
print(f"Initial condition: {initial_condition_name}")
if not os.path.exists(f"{output_file_directory}/{initial_condition_type}_initial-condition/"):
    # os.system(f"mkdir {output_file_directory}/{initial_condition_type}_initial-condition/")
    os.makedirs(f"{output_file_directory}/{initial_condition_type}_initial-condition/", exist_ok=True)
    print(f"Creating subfolder for initial condition...")
else:
    print(f"Subfolder already exists for initial condition")
print(f"==============================================\n")

# #### Generate new initial condition
[
    generated_zonal_velocity,
    generated_meridional_velocity,
    generated_column_temperature,
    generated_column_moisture
] = generate_wavelike_initial_condition(initial_condition_type, n_wavelengths = n_wavelengths, save_initial_condition=True)

##### Specify which initial condition to use when solving
# Generated
initial_zonal_velocity = generated_zonal_velocity
initial_meridional_velocity = generated_meridional_velocity
initial_column_temperature = generated_column_temperature
initial_column_moisture = generated_column_moisture

# loaded_final_timestep = xr.load_dataset(
#     f"output/Ahmed-21/"
#     + f"epst=0.50_epsq=0.17_r=0.2_nx=1.0_ny=1.00_test-long_wavenumber-filtered_quadratic-mean-moisture_non-diffusive-damped-moist-coupled-simulation/"
#     + f"Kelvin-wave_initial-condition/"
#     + f"k={n_wavelengths}.0_m=1_Kelvin-wave_final-timesteps-model-data_chunk-24of24.nc"
# )

# initial_zonal_velocity = np.copy(loaded_final_timestep['u'])[None, :, :]
# initial_meridional_velocity = np.copy(loaded_final_timestep['v'])[None, :, :]
# initial_column_temperature = np.copy(loaded_final_timestep['T'])[None, :, :]
# initial_column_moisture = np.copy(loaded_final_timestep['q'])[None, :, :]

# Create a folder to save figures
if not os.path.exists(f"{output_file_directory}/{initial_condition_type}_initial-condition/figures/"):
    print("======================")
    print("Creating figures folder...")
    os.makedirs(f"{output_file_directory}/{initial_condition_type}_initial-condition/figures/", exist_ok=True)
    print("Figures folder created")
    print("======================")

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
#     (-180, 180), 
#     # (-60, 60),
#     (-30, 30),
#     30, 14,
#     True, 
#     'converted',
#     0.6,
#     # 'natural',
#     # 1,
#     1e-6,
#     False
# )

# # Plot the initial conditon below
# plot_horizontal_structure(
#     0,
#     zonal_gridpoints,
#     meridional_gridpoints,
#     time_points,
#     zonal_velocity = initial_zonal_velocity, 
#     meridional_velocity = initial_meridional_velocity, 
#     column_temperature = initial_column_temperature, 
#     column_moisture = initial_column_moisture, 
#     specified_output_file_directory=output_file_directory,
#     specified_initial_condition_name=initial_condition_name,
#     physical_parameters = physical_parameters,
#     simulation_parameters = (simulation_moisture, fringe_region, fringe_region_latitude, fringe_region_width),
#     plotting_parameters = plotting_parameters
# )

# Run Simulation
# Runge-Kutta Timestepping

n_rk_steps = 3
save_downsampled = True
overwrite_data = 'y'

# Construct moisture sensitivity array
if simulation_moisture:
    if moisture_sensitivity_structure == 'constant':
        moisture_sensitivity_array = MOISTURE_SENSITIVITY
    elif moisture_sensitivity_structure == 'logistic':
        moisture_sensitivity_array = MOISTURE_SENSITIVITY * (
            1-fringe_region_damping_function(
                meridional_gridpoints/METERS_PER_DEGREE, 
                -sensitivity_limit, sensitivity_limit, sensitivity_width, 1
            )
        )[:, None]
    elif moisture_sensitivity_structure == 'gaussian':
        moisture_sensitivity_array = MOISTURE_SENSITIVITY * np.exp(-(meridional_gridpoints/gaussian_length_scale)**2)[:, None]
    else:
        raise ValueError(f"Invalid moisture sensitivity structure: {moisture_sensitivity_structure}")
    print(f"Moisture sensitivty structure: {moisture_sensitivity_structure}")


# Construct temperature sensitivity array
if simulation_damping:
    if temperature_sensitivity_structure == 'constant':
        temperature_sensitivity_array = TEMPERATURE_SENSITIVITY
    elif temperature_sensitivity_structure == 'logistic':
        temperature_sensitivity_array = TEMPERATURE_SENSITIVITY * (
            1-fringe_region_damping_function(
                meridional_gridpoints/METERS_PER_DEGREE, 
                -sensitivity_limit, sensitivity_limit, sensitivity_width, 1
            )
        )[:, None]
    elif temperature_sensitivity_structure == 'gaussian':
        temperature_sensitivity_array = TEMPERATURE_SENSITIVITY * np.exp(-(meridional_gridpoints/gaussian_length_scale)**2)[:, None]
    else:
        raise ValueError(f"Invalid temperature sensitivity structure: {temperature_sensitivity_structure}")
    print(f"Temperature sensitivty structure: {temperature_sensitivity_structure}")

# Construct moisture stratification array
if simulation_moisture:
    if moisture_stratification_structure == 'constant':
        moisture_stratification_array = gross_moisture_stratification
    elif moisture_stratification_structure == 'logistic':
        moisture_stratification_array = gross_moisture_stratification * (
            1-fringe_region_damping_function(
                meridional_gridpoints/METERS_PER_DEGREE, 
                -sensitivity_limit, sensitivity_limit, sensitivity_width, 1
            )
        )[:, None]
    elif moisture_stratification_structure == 'gaussian':
        moisture_stratification_array = gross_moisture_stratification * np.exp(-(meridional_gridpoints/gaussian_length_scale)**2)[:, None]
    else:
        raise ValueError(f"Invalid moisture stratification structure: {moisture_stratification_structure}")
    print(f"Moisture stratification structure: {moisture_stratification_structure}")

if simulation_moisture and moisture_advection: 
    if zonal_moistening_zonal_structure == 'constant':
        zonal_moistening_array = ZONAL_MOISTENING_PARAMETER
    elif zonal_moistening_zonal_structure == 'symmetric cosine':
        zonal_moistening_array = ZONAL_MOISTENING_PARAMETER * 0.5*(1+np.cos(zonal_gridpoints/EARTH_RADIUS))[None, :]
    elif zonal_moistening_zonal_structure == 'asymmetric cosine':
        zonal_moistening_array = ZONAL_MOISTENING_PARAMETER * np.cos(zonal_gridpoints/EARTH_RADIUS)[None, :]
    else:
        raise ValueError(f"Invalid zonal moistening zonal structure: {zonal_moistening_zonal_structure}")
    print(f"Moisture advection by zonal winds structure: {zonal_moistening_zonal_structure}")

meridional_moistening_array = MERIDIONAL_MOISTENING_PARAMETER

for chunk in range(1, n_chunks+1):
    chunk_length = nt//n_chunks
    # chunk_length = nt
    
    # Initialize the variables to simulate
    zonal_velocity = np.zeros((chunk_length, ny, nx))
    meridional_velocity = np.zeros((chunk_length, ny, nx))
    column_temperature = np.zeros((chunk_length, ny, nx))
    column_moisture = np.zeros((chunk_length, ny, nx))
    
    print(f"Chunk {chunk}/{n_chunks}")
    
    # The first chunk is initialized with preset initial conditions
    if chunk == 1:
        zonal_velocity[0] = initial_zonal_velocity[0]
        meridional_velocity[0] = initial_meridional_velocity[0]
        column_temperature[0] = initial_column_temperature[0]
        column_moisture[0] = initial_column_moisture[0]
     
    # Subsequent chunks are initialized with the output of the previous chunk
    elif chunk > 1:
        final_timesteps_model_data_input = xr.load_dataset(
            f"{output_file_directory}/{initial_condition_type}_initial-condition/"
            + f"{initial_condition_name}_final-timesteps-model-data_chunk-{chunk-1}of{n_chunks}.nc"
        )

        zonal_velocity[0] = final_timesteps_model_data_input['u'].to_numpy()
        meridional_velocity[0] = final_timesteps_model_data_input['v'].to_numpy()
        column_temperature[0] = final_timesteps_model_data_input['T'].to_numpy()
        column_moisture[0] = final_timesteps_model_data_input['q'].to_numpy()

    ### Step forward using Runge-Kutta time-differencing
    # for it in tqdm(range(1, chunk_length), position=0, leave=True, ncols=100):
    for it in range(1, chunk_length):
        zonal_velocity_rk       = np.zeros((n_rk_steps+1, ny, nx))
        meridional_velocity_rk  = np.zeros((n_rk_steps+1, ny, nx))
        column_temperature_rk   = np.zeros((n_rk_steps+1, ny, nx))
        column_moisture_rk      = np.zeros((n_rk_steps+1, ny, nx))

        zonal_velocity_rk[0]       = zonal_velocity[it-1]
        meridional_velocity_rk[0]  = meridional_velocity[it-1]
        column_temperature_rk[0]   = column_temperature[it-1]
        column_moisture_rk[0]      = column_moisture[it-1]
        
        for rk_step in range(1, n_rk_steps+1):
            
            if dynamical_core == 'dry-Kelvin':
                field_variables = (zonal_velocity_rk[rk_step-1], column_temperature_rk[rk_step-1])
                
                column_temperature_forcing = dry_Kelvin_wave_dynamical_core(
                    zonal_wavenumber, meridional_gridpoints, field_variables
                )

                column_temperature_rk[rk_step] = (
                    column_temperature[it-1] + (time_step/(n_rk_steps - rk_step + 1))*column_temperature_forcing
                )

                # dTdy = compute_meridional_derivative_reflected(meridional_gridpoints, np.copy(column_temperature_rk[rk_step]), sign=1)
                dTdy = compute_meridional_derivative_finite_difference(meridional_gridpoints, np.copy(column_temperature_rk[rk_step]))
                # compute_meridional_derivative_finite_difference
                zonal_velocity_rk[rk_step] = (
                    -1/(CORIOLIS_PARAMETER * meridional_gridpoints[:, None]) * (gravity_wave_phase_speed**2/GROSS_DRY_STABILITY) * dTdy
                )
                
                meridional_velocity_rk[rk_step] = np.zeros_like(column_temperature_rk[rk_step])
                column_moisture_rk[rk_step] = np.zeros_like(column_temperature_rk[rk_step])
                
            elif dynamical_core == 'Ahmed-21':
                field_variables = (
                    zonal_velocity_rk[rk_step-1], 
                    meridional_velocity_rk[rk_step-1],
                    column_temperature_rk[rk_step-1],
                    column_moisture_rk[rk_step-1]
                )

                Ahmed_parameters = (
                    temperature_sensitivity_array, 
                    moisture_sensitivity_array, 
                    zonal_moistening_array, 
                    meridional_moistening_array,
                    moisture_stratification_array
                )
                
                [
                    zonal_velocity_forcing, 
                    meridional_velocity_forcing,
                    column_temperature_forcing,
                    column_moisture_forcing
                ] = Ahmed21_dynamical_core(
                    zonal_wavenumber, meridional_gridpoints, field_variables, Ahmed_parameters
                )

                zonal_velocity_rk[rk_step] = (
                    zonal_velocity[it-1] + (time_step/(n_rk_steps - rk_step + 1))*zonal_velocity_forcing
                )
                meridional_velocity_rk[rk_step] = (
                    meridional_velocity[it-1] + (time_step/(n_rk_steps - rk_step + 1))*meridional_velocity_forcing
                )
                column_temperature_rk[rk_step] = (
                    column_temperature[it-1] + (time_step/(n_rk_steps - rk_step + 1))*column_temperature_forcing
                )
                column_moisture_rk[rk_step] = (
                    column_moisture[it-1] + (time_step/(n_rk_steps - rk_step + 1))*column_moisture_forcing
                )

            elif dynamical_core == 'dry-Matsuno':
                field_variables = (
                    zonal_velocity_rk[rk_step-1], 
                    meridional_velocity_rk[rk_step-1],
                    column_temperature_rk[rk_step-1],
                    column_moisture_rk[rk_step-1]
                )

                [
                    zonal_velocity_forcing, 
                    meridional_velocity_forcing,
                    column_temperature_forcing,
                    column_moisture_forcing
                ] = dry_Matsuno_dynamical_core(
                    zonal_wavenumber, meridional_wavenumber, meridional_gridpoints, field_variables
                )

                zonal_velocity_rk[rk_step] = (
                    zonal_velocity[it-1] + (time_step/(n_rk_steps - rk_step + 1))*zonal_velocity_forcing
                )
                meridional_velocity_rk[rk_step] = (
                    meridional_velocity[it-1] + (time_step/(n_rk_steps - rk_step + 1))*meridional_velocity_forcing
                )
                column_temperature_rk[rk_step] = (
                    column_temperature[it-1] + (time_step/(n_rk_steps - rk_step + 1))*column_temperature_forcing
                )
                column_moisture_rk[rk_step] = (
                    column_moisture[it-1] + (time_step/(n_rk_steps - rk_step + 1))*column_moisture_forcing
                )
            
            # Meridional Boundary conditions
            # meridional_velocity_rk[rk_step, 0] = 0.
            # meridional_velocity_rk[rk_step, -1] = 0.

            # zonal_velocity_rk[rk_step, 0] = 0.
            # zonal_velocity_rk[rk_step, -1] = 0.
            
            # column_temperature_rk[rk_step, 0] = 0.
            # column_temperature_rk[rk_step, -1] = 0.
            
            # column_moisture_rk[rk_step, 0] = 0.
            # column_moisture_rk[rk_step, -1] = 0.
            
        # The full time step is the result of the RK-time stepping
        zonal_velocity[it] = zonal_velocity_rk[-1]
        meridional_velocity[it] = meridional_velocity_rk[-1]
        column_temperature[it] = column_temperature_rk[-1]
        column_moisture[it] = column_moisture_rk[-1]

        # Meridional boundary conditions
        # meridional_velocity[it,0] = 0.
        # meridional_velocity[it,-1] = 0.

        # zonal_velocity[it,0] = 0.
        # zonal_velocity[it,-1] = 0.
        
        # column_temperature[it,0] = 0.
        # column_temperature[it,-1] = 0.
        
        # column_moisture[it,0] = 0.
        # column_moisture[it,-1] = 0.

        if wavenumber_filtering:
            zonal_velocity_fft = np.fft.fft(zonal_velocity[it], axis=1)
            meridional_velocity_fft = np.fft.fft(meridional_velocity[it], axis=1)
            column_temperature_fft = np.fft.fft(column_temperature[it], axis=1)
            column_moisture_fft = np.fft.fft(column_moisture[it], axis=1)
        
            zonal_velocity_fft_masked = np.copy(zonal_velocity_fft)
            meridional_velocity_fft_masked = np.copy(meridional_velocity_fft)
            column_temperature_fft_masked = np.copy(column_temperature_fft)
            column_moisture_fft_masked = np.copy(column_moisture_fft)
    
            zonal_velocity_fft_masked[:, np.where(np.abs(zonal_wavenumber) != zonal_wavenumber[n_wavelengths])] = 0. + 0.*1j
            meridional_velocity_fft_masked[:, np.where(np.abs(zonal_wavenumber) != zonal_wavenumber[n_wavelengths])] = 0. + 0.*1j
            column_temperature_fft_masked[:, np.where(np.abs(zonal_wavenumber) != zonal_wavenumber[n_wavelengths])] = 0. + 0.*1j
            column_moisture_fft_masked[:, np.where(np.abs(zonal_wavenumber) != zonal_wavenumber[n_wavelengths])] = 0. + 0.*1j
            
            zonal_velocity[it] = np.real(np.fft.ifft(zonal_velocity_fft_masked, axis=1))
            meridional_velocity[it] = np.real(np.fft.ifft(meridional_velocity_fft_masked, axis=1))
            column_temperature[it] = np.real(np.fft.ifft(column_temperature_fft_masked, axis=1))
            column_moisture[it] = np.real(np.fft.ifft(column_moisture_fft_masked, axis=1))

    #### Downsample data 
    if save_downsampled:       
        
        downsample_interval = round_to_multiple(np.floor(6*3600/time_step), n_chunks)
        full_indices = np.array(
            [i * downsample_interval for i in range(len(time_points[::downsample_interval]))]
        )
        chunked_indices = full_indices % (chunk_length-1)
        full_data_indices = np.split(chunked_indices, np.where(np.diff(chunked_indices) < 0)[0] + 1)
        data_indices = full_data_indices[chunk-1]
        
        full_time_indices = np.split(full_indices, np.where(np.diff(chunked_indices) < 0)[0] + 1)
        time_indices = full_time_indices[chunk-1]

        if not os.path.exists(f"{output_file_directory}/{initial_condition_type}_initial-condition/"):
            print(f"Creating subfolder for {initial_condition_type} initial condition...")
            os.system(f"mkdir {output_file_directory}/{initial_condition_type}_initial-condition/")
            
        print(f"Downsampled k = {n_wavelengths} data")
        # Store model variables in an xarray Dataset
        downsampled_model_data = xr.Dataset(
            data_vars = {
                'u' : (["time", "y", "x"], zonal_velocity[:-1,:,:][data_indices]),
                'v' : (["time", "y", "x"], meridional_velocity[:-1,:,:][data_indices]),
                'T' : (["time", "y", "x"], column_temperature[:-1,:,:][data_indices]),
                'q' : (["time", "y", "x"], column_moisture[:-1,:,:][data_indices]),
            },
            coords = {
            "x" : zonal_gridpoints,
            "y" : meridional_gridpoints,
            "time" : time_points[time_indices]
            }
        )

        print(f"---- Saving data...")
        # Save the Dataset to a netCDF file
        downsampled_model_data.to_netcdf(
            f"{output_file_directory}/{initial_condition_type}_initial-condition/" 
            + f"{initial_condition_name}_downsampled-model-data_chunk-{chunk}of{n_chunks}.nc"
        )
        print(f"---- Data saved \n")

    #### Save the temperature and moisture of the entire final chunk
    print("Saving final chunk heating...")
    # Store model variables in an xarray Dataset
    if chunk == n_chunks:
        final_chunk_data = xr.Dataset(
            data_vars = {
                'T' : (["time", "y", "x"], column_temperature),
                'q' : (["time", "y", "x"], column_moisture),
            },
            coords = {
            "time" : time_points[:chunk_length],
            "x" : zonal_gridpoints,
            "y" : meridional_gridpoints,
            }
        )
    
        print(f"---- Saving data...")
        # Save the Dataset to a netCDF file
        final_chunk_data.to_netcdf(
            f"{output_file_directory}/{initial_condition_type}_initial-condition/" 
            + f"{initial_condition_name}_final-chunk-data.nc"
        )
        print(f"---- Data saved \n")

    # Save the final timestep of the chunk
    print("Final timestep data")
    final_timesteps_model_data = xr.Dataset(
        data_vars = {
            'u' : (["y", "x"], zonal_velocity[-1]),
            'v' : (["y", "x"], meridional_velocity[-1]),
            'T' : (["y", "x"], column_temperature[-1]),
            'q' : (["y", "x"], column_moisture[-1])
        },
        coords = {
        "x" : zonal_gridpoints,
        "y" : meridional_gridpoints,
        }
    )

    print(f"---- Saving data...")
    final_timesteps_model_data.to_netcdf(
        f"{output_file_directory}/{initial_condition_type}_initial-condition/" 
        + f"{initial_condition_name}_final-timesteps-model-data_chunk-{chunk}of{n_chunks}.nc"
    )
    print(f"---- Data saved \n")

print("===================")
print("Simulation complete")


# Compile downsampled data
data_to_compile = []
print(f"{output_file_directory}/{initial_condition_type}_initial-condition/")
print(f"==============================================================")
for chunk in range(1, n_chunks+1):
    data_to_compile.append(
        xr.load_dataset(
            f"{output_file_directory}/{initial_condition_type}_initial-condition/" 
            + f"{initial_condition_name}_downsampled-model-data_chunk-{chunk}of{n_chunks}.nc"
        )
    )
    print(f"{initial_condition_name}_downsampled-model-data_chunk-{chunk}of{n_chunks}.nc")
print(f"==============================================================")
if os.path.exists(
    f"{output_file_directory}/{initial_condition_type}_initial-condition/" 
    + f"{initial_condition_name}_downsampled-model-data_compiled.nc"
):
    print(f"Overwriting existing data... \n")
else:
    print(f"Compiling data... \n")

compiled_data = xr.concat(data_to_compile, dim='time')
compiled_data.to_netcdf(
            f"{output_file_directory}/{initial_condition_type}_initial-condition/" 
            + f"{initial_condition_name}_downsampled-model-data_compiled.nc"
        )

print(
    f"Data compiled as:\n"
    + f"{output_file_directory}/{initial_condition_type}_initial-condition/" 
    + f"{initial_condition_name}_downsampled-model-data_compiled.nc"
)

print("All simulations complete")