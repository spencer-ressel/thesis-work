# Set my local timezone
import time
import os
os.environ["TZ"] = "US/Pacific"   # or your timezone
time.tzset()

# ============================
# File paths
# ============================
DATA_DIRECTORY = "/glade/campaign/univ/uwas0152/post_processed_data"
AQUAPLANET_OUTPUT_DIRECTORY = (
    "/glade/u/home/sressel/thesis-work/python/aquaplanet_analysis/output/"
)

# Used for plotting separater lines
SEP_WIDTH = 40

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

# ============================
# Plotting metadata (parameters only)
# ============================
PLOTTING_ATTRIBUTES = {
    "Precipitation": {
        "cmap": "YlGnBu",
        "d_cmap_params": ("BrBG", "white", 0.05, 0.05),
    },
    "Outgoing Longwave Radiation": {"d_cmap": "gray_r"},
    "Zonal Wind": {
        "cmap_params": ("coolwarm", "white", 0.05, 0.05),
        "norm_center": 0,
        "d_cmap_params": ("coolwarm", "white", 0.05, 0.05),
    },
    "Meridional Wind": {
        "cmap": "coolwarm",
        "norm_center": 0,
        "d_cmap": "coolwarm",
    },
    "Vertical Wind": {
        "cmap_params": ("coolwarm", "white", 0.05, 0.05),
        "norm_center": 0,
        "d_cmap": "coolwarm",
    },
    "Temperature": {"cmap": "YlOrRd", "d_cmap": "RdYlBu_r"},
    "Moisture": {"cmap": "YlGnBu", "d_cmap": "BrBG"},
    "Moist Static Energy": {
        "d_cmap_params": ("coolwarm", "white", 0.05, 0.05)
    },
    "Geopotential Height": {"cmap": "YlOrBr", "d_cmap": "PuOr"},
    "Longwave Heating Rate": {
        "cmap": "YlGnBu_r",
        "d_cmap_params": ("coolwarm", "white", 0.05, 0.05),
    },
    "Shortwave Heating Rate": {
        "cmap": "YlOrRd",
        "d_cmap_params": ("coolwarm", "white", 0.05, 0.05),
    },
    "Latent Heat Flux": {
        "d_cmap_params": ("coolwarm", "white", 0.05, 0.05)
    },
    "Sensible Heat Flux": {
        "d_cmap_params": ("coolwarm", "white", 0.05, 0.05)
    },
    "Column Temperature": {"cmap": "YlOrRd", "d_cmap": "RdYlBu_r"},
    "Column Water Vapor": {"cmap": "YlGnBu", "d_cmap": "BrBG"},
    "Column Longwave Heating": {
        "cmap": "YlGnBu_r",
        "d_cmap_params": ("coolwarm", "white", 0.05, 0.05),
    },
    "Column Shortwave Heating": {
        "cmap": "YlOrRd",
        "d_cmap_params": ("coolwarm", "white", 0.05, 0.05),
    },
    "Column Moist Static Energy": {
        "cmap": "coolwarm",
        "d_cmap_params": ("coolwarm", "white", 0.05, 0.05),
    },
    "Potential Temperature": {"cmap": "YlOrRd", "d_cmap": "RdYlBu_r"},
    "Saturation Specific Humidity": {"cmap": "YlGnBu", "d_cmap": "BrBG"},
    "Column Relative Humidity": {"cmap": "YlGnBu", "d_cmap": "BrBG"},
    "Chikira alpha": {
        "cmap": "coolwarm",
        "d_cmap_params": ("coolwarm", "white", 0.05, 0.05),
        "norm_center": 1,
    },
    "Net Longwave Flux": {
        "cmap": "YlGnBu_r",
        "d_cmap_params": ("coolwarm", "white", 0.05, 0.05),
    },
    "Net Shortwave Flux": {
        "cmap": "YlOrRd",
        "d_cmap_params": ("coolwarm", "white", 0.05, 0.05),
    },
}

# ============================
# Subset bounds (constants only)
# ============================
LATITUDE_SOUTH = -30
LATITUDE_NORTH = 30
CENTRAL_LONGITUDE = 0
LONGITUDE_MIN = 0
LONGITUDE_MAX = 360

INTRASEASONAL_LOWCUT = 100
INTRASEASONAL_HIGHCUT = 20

LARGE_SCALE_CUTOFF = 1
SMALL_SCALE_CUTOFF = 3

LOWER_LEVEL_PRESSURE = 950
UPPER_LEVEL_PRESSURE = 0

# ============================
# Experiment metadata
# ============================
EXPERIMENTS = ["-4K", "0K", "4K"]
EXPERIMENT_DISPLAY_NAMES = {"-4K": "−4K", "0K": "CTRL", "4K": "+4K"}
EXPERIMENT_CMAPS = {"-4K": "Blues", "0K": "Reds", "4K": "Purples"}
