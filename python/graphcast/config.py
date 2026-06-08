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
OUTPUT_DIRECTORY = "/glade/u/home/sressel/thesis-work/python/graphcast/output"

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