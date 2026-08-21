# Set my local timezone
import time
import os
os.environ["TZ"] = "US/Pacific"   # or your timezone
time.tzset()

# ============================
# File paths
# ============================
INPUT_DATA_DIRECTORY = "/glade/u/home/sressel/spencer-scratch/graphcast_input_data"
GRAPHCAST_DATA_DIRECTORY = "/glade/u/home/sressel/spencer-scratch/graphcast_output"
ROOT_DIRECTORY = "/glade/u/home/sressel/thesis-work/python/graphcast"
OUTPUT_DIRECTORY = f"{ROOT_DIRECTORY}/output"

# ============================
# Simulation Parameters
# ============================
TIMESTEPS_PER_DAY = 4              

# ============================
# Physical constants
# ============================
SECONDS_PER_DAY = 24 * 3600
EARTH_RADIUS = 6378137                   # m
GRAVITY = 9.81                           # m s^-2
HEAT_OF_VAPORIZATION = 2.26*10**6        # J kg^-1
HEAT_OF_FUSION = 334*10**3               # J kg^-1
SPECIFIC_HEAT = 1005                     # J kg^-1 K^-1
DRY_AIR_GAS_CONSTANT = 287               # J kg^-1 K^-1
WATER_VAPOR_GAS_CONSTANT = 461           # J kg^-1 K^-1
LIQUID_WATER_DENSITY = 1000              # kg m^-3
STEFAN_BOLTZMANN_CONSTANT = 5.67*10**-8  # W m^-2 K^-4

PREDICTED_VARIABLES = {
    "expver",
    "number",
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "vertical_velocity",
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation_6hr",
    "initial_conditions_filepath"
}

VARIABLE_SHORTNAMES = {
    # Pressure‑level variables
    "geopotential": "Z",
    "temperature": "T",
    "u_component_of_wind": "U",
    "v_component_of_wind": "V",
    "specific_humidity": "q",
    "vertical_velocity": "ω",
    "velocity_potential": "VP",

    # Surface variables
    "2m_temperature": "T2m",
    "mean_sea_level_pressure": "MSLP",
    "10m_u_component_of_wind": "U10",
    "10m_v_component_of_wind": "V10",
    "total_precipitation_6hr": "TP6",

    # Other variables
    "Column Water Vapor": "CWV"
}

VARIABLE_LONGNAMES = {
    # Pressure‑level variables
    "geopotential": "Geopotential",
    "temperature": "Temperature",
    "u_component_of_wind": "Zonal Wind",
    "v_component_of_wind": "Meridional Wind",
    "specific_humidity": "Specific Humidity",
    "vertical_velocity": "Vertical Wind",
    "velocity_potential": "Velocity Potential",

    # Surface variables
    "2m_temperature": "2m Temperature",
    "mean_sea_level_pressure": "Mean Sea Level Pressure",
    "10m_u_component_of_wind": "10m Zonal Wind",
    "10m_v_component_of_wind": "10m Meridional Wind",
    "total_precipitation_6hr": "Total Six Hourly Precipitation",

    # Other variables
    "Column Water Vapor": "Column Water Vapor"
}

VARIABLE_UNITS = {
    # Pressure‑level variables
    "geopotential":        r"m$^{2}$ s$^{-2}$",
    "temperature":         r"K",
    "u_component_of_wind": r"m s$^{-1}$",
    "v_component_of_wind": r"m s$^{-1}$",
    "specific_humidity":   r"kg kg$^{-1}$",
    "vertical_velocity":   r"Pa s$^{-1}$",
    "velocity_potential":   r"m$^{2}$ s$^{-1}$",

    # Surface variables
    "2m_temperature":          r"K",
    "mean_sea_level_pressure": r"Pa",
    "10m_u_component_of_wind": r"m s$^{-1}$",
    "10m_v_component_of_wind": r"m s$^{-1}$",
    "total_precipitation_6hr": r"m",

    # Other variables
    "Column Water Vapor": r"kg m$^{-2}$"
}