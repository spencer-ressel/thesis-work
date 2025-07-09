import sys
sys.path.insert(0, "/glade/u/home/sressel/thesis-work/python/auxiliary_functions/")
import mjo_mean_state_diagnostics as mjo
import matplotlib.colors as mcolors
import cftime
import xarray as xr

data_directory = rf"/glade/campaign/univ/uwas0152/post_processed_data"
aquaplanet_output_directory = f"/glade/u/home/sressel/thesis-work/python/aquaplanet_analysis/output/"

plotting_attributes = {
    'Precipitation': dict(
        cmap='YlGnBu',
        # d_cmap='BrBG'
        d_cmap=mjo.modified_colormap("BrBG", "white", 0.05, 0.05)
    ),
    'Outgoing Longwave Radiation': dict(d_cmap='gray_r'),
    'Zonal Wind': dict(
        cmap=mjo.modified_colormap("coolwarm", "white", 0.05, 0.05),
        norm=mcolors.CenteredNorm(vcenter=0),
        d_cmap=mjo.modified_colormap("coolwarm", "white", 0.05, 0.05)
    ),
    'Meridional Wind': dict(
        cmap='coolwarm',
        norm=mcolors.CenteredNorm(vcenter=0),
        d_cmap='coolwarm'
    ),
    'Vertical Wind': dict(
        cmap=mjo.modified_colormap("coolwarm", "white", 0.05, 0.05),
        norm=mcolors.CenteredNorm(vcenter=0),
        d_cmap='coolwarm'
    ),
    'Temperature': dict(
        cmap='YlOrRd',
        d_cmap='RdYlBu_r'
    ),
    'Moisture': dict(
        cmap='YlGnBu',
        d_cmap='BrBG'
    ),
    'Moist Static Energy': dict(
        d_cmap=mjo.modified_colormap("coolwarm", "white", 0.05, 0.05)
    ),
    'Geopotential Height': dict(
        cmap='YlOrBr',
        d_cmap='PuOr'
    ),
    'Longwave Heating Rate': dict(
        cmap='YlGnBu_r',
        d_cmap=mjo.modified_colormap("coolwarm", "white", 0.05, 0.05)
    ),
    'Shortwave Heating Rate': dict(
        cmap='YlOrRd',
        d_cmap=mjo.modified_colormap("coolwarm", "white", 0.05, 0.05)
    ),
    'Latent Heat Flux': dict(
        d_cmap=mjo.modified_colormap("coolwarm", "white", 0.05, 0.05)
    ),
    'Sensible Heat Flux': dict(
        d_cmap=mjo.modified_colormap("coolwarm", "white", 0.05, 0.05)
    ),
    'Column Temperature': dict(
        cmap='YlOrRd',
        d_cmap='RdYlBu_r'
    ),
    'Column Water Vapor': dict(
        cmap='YlGnBu',
        d_cmap='BrBG'
    ),
    'Column Longwave Heating': dict(
        cmap='YlGnBu_r',
        d_cmap=mjo.modified_colormap("coolwarm", "white", 0.05, 0.05)
    ),
    'Column Shortwave Heating': dict(
        cmap='YlOrRd',
        d_cmap=mjo.modified_colormap("coolwarm", "white", 0.05, 0.05)
    ),
    'Column Moist Static Energy': dict(
        cmap='coolwarm',
        d_cmap=mjo.modified_colormap("coolwarm", "white", 0.05, 0.05)
    ),
    'Potential Temperature': dict(
        cmap='YlOrRd',
        d_cmap='RdYlBu_r'
    ),
    'Saturation Specific Humidity': dict(
        cmap='YlGnBu',
        d_cmap='BrBG'
    ),
    'Column Relative Humidity': dict(
        cmap='YlGnBu',
        d_cmap='BrBG'
    ),
    'Chikira alpha': dict(
        cmap='coolwarm',
        d_cmap=mjo.modified_colormap("coolwarm", "white", 0.05, 0.05),
        norm=mcolors.CenteredNorm(vcenter=1)
    ),
    'Net Longwave Flux': dict(
        cmap='YlGnBu_r',
        d_cmap=mjo.modified_colormap("coolwarm", "white", 0.05, 0.05)
    ),
    'Net Shortwave Flux': dict(
        cmap='YlOrRd',
        d_cmap=mjo.modified_colormap("coolwarm", "white", 0.05, 0.05)
    ),
}

experiments_list = ['-4K', '0K', '4K']
experiments_array = xr.DataArray(
    data=experiments_list,
    dims=["experiment"],
    coords={
        'experiment': experiments_list,
        'name': ('experiment', ['−4K', 'CTRL', '+4K'])
    }
)

# Specify parameters
# Physical constants
# Seconds per day
SECONDS_PER_DAY = 24 * 3600
EARTH_RADIUS = 6378137               # m
GRAVITY = 9.81                       # m s^-2
HEAT_OF_VAPORIZATION = 2.26*10**6    # J kg^-1
HEAT_OF_FUSION = 334*10**3           # J kg^-1
SPECIFIC_HEAT = 1005                 # J kg^-1 K^-1
DRY_AIR_GAS_CONSTANT = 287           # J kg^-1 K^-1
WATER_VAPOR_GAS_CONSTANT = 461       # J kg^-1 K^-1
LIQUID_WATER_DENSITY = 1000          # kg m^-3

# Print statement string width
str_width = 40

# Standard plotting grid kwargs
grid_kwargs = {"linewidth": 1, "linestyle": (0, (5, 10)), "color": "gray"}

# Identify the days that are missing data
from datetime import timedelta
missing_days = [
    cftime.DatetimeNoLeap(7, 2, 4, 0, 0, 0, 0, has_year_zero=True),
    cftime.DatetimeNoLeap(7, 2, 5, 0, 0, 0, 0, has_year_zero=True),
    cftime.DatetimeNoLeap(7, 2, 6, 0, 0, 0, 0, has_year_zero=True),
    cftime.DatetimeNoLeap(7, 2, 7, 0, 0, 0, 0, has_year_zero=True),
    cftime.DatetimeNoLeap(7, 2, 8, 0, 0, 0, 0, has_year_zero=True),
]
missing_timesteps = [missing_days[0] + timedelta(hours=x) for x in range(0, 24*3, 3)]