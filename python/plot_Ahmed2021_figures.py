#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# plot_Ahmed2021_figures.py                                                   #
# Spencer Ressel                                                              #
# 2021.12.10                                                                  #
###############################################################################
"""
This script recreates Figures 1a and 1b from Ahmed 2021 using ERA5 column 
water vapor data. 
"""

# %% Imports
import os
import numpy as np
import xarray as xr
import matplotlib.ticker as mticker
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import seaborn as sns
import cartopy.crs as ccrs
import cartopy as cart

os.chdir(r"C:\Users\resse\OneDrive\UW\Research\Recreation of Ahmed 2021")

#%% Constants
# Physical Constants
METERS_PER_DEGREE = 110.574 * 1e3
DENSITY_WATER = 997
GRAVITY = 9.8
LATENT_HEAT = 2.26 * 10 ** 6
SPECIFIC_HEAT_DRY_AIR = 1004
GROSS_DRY_STABILITY = 3.12e4

# Related to vertical structures
DELTA_PRESSURE = 999 * 1e2
INTEGRATED_MOISTURE_STRUCTURE = 54.6
INTEGRATED_VELOCITY_STRUCTURE = -0.35531135531135527

# sigma_y value from Ahmed 2021
A21_MERIDIONAL_MOISTURE_PARAMETER = 9e-9
# %% Load Data & Pre-process
# Load and process the data
data_directory = r"C:\Users\resse\Desktop\Data\era5/"
file_name = r"monthly_reanalysis_CWV_CIT_2002_2014.nc"

# DataArray
data = xr.open_dataset(data_directory + file_name, engine="netcdf4")
data = data.reindex(latitude=list(reversed(data.latitude)))
data = data.rename({"p54.162": "column temperature"})

# Coordinates
time = data["time"]
latitude = data["latitude"]
longitude = data["longitude"]

# Total Column Water Vapour
column_water_vapor = data["tcwv"]
# column_water_vapor = 1000 * (
#     column_water_vapor / DENSITY_WATER
# )  # Convert from kg/m^2 to mm

# Column-Integrated Temperature
total_column_temperature = data["column temperature"]

# Average Column Temperature
average_column_temperature = total_column_temperature / (
    DELTA_PRESSURE / GRAVITY
)  # Convert from K kg m^-2 to K

# Time Mean Column Water Vapor
time_mean_column_water_vapor_global = column_water_vapor.mean(dim="time")
time_mean_column_water_vapor_anomalies = (
    time_mean_column_water_vapor_global
    - time_mean_column_water_vapor_global.mean(dim=["latitude", "longitude"])
)

# Time Mean Average Column Temperature
time_mean_column_temperature = average_column_temperature.mean(dim="time")
time_mean_column_temperature_anomalies = (
    time_mean_column_temperature - time_mean_column_temperature.mean()
)

# %% Plotting Maps
#### Column Water Vapor
[fig, ax] = plt.subplots(
    1,
    figsize=(32, 16),
    subplot_kw={"projection": ccrs.PlateCarree(central_longitude=100)},
)
plt.rcParams.update({"font.size": 24})
ax.spines[["bottom", "top", "left", "right"]].set_color("0")

# Specify colormap
column_water_vapor_colormap = sns.color_palette("GnBu", as_cmap=True)

# Plot the CWV data as a function of longitude and latitude
im = ax.contourf(
    longitude,
    latitude,
    time_mean_column_water_vapor_global,
    transform=ccrs.PlateCarree(),
    cmap=column_water_vapor_colormap,
    levels=20,
)
ax.set_title("November-April mean CWV")

# Create a colorbar
cbar = fig.colorbar(im, location="bottom", fraction=0.1, aspect=50, shrink=1, pad=0.1)
cbar.set_label("mm")

# Add the land overlay
# ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k')
gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=2,
    color="black",
    alpha=0.0,
    linestyle="-",
)
gl.right_labels = False
gl.top_labels = False
ax.coastlines()

# Plot it
fig.tight_layout()
plt.show()

#### Column Average Temperature Temperature
[fig, ax] = plt.subplots(
    1,
    figsize=(32, 16),
    subplot_kw={"projection": ccrs.PlateCarree(central_longitude=100)},
)
plt.rcParams.update({"font.size": 24})
ax.spines[["bottom", "top", "left", "right"]].set_color("0")

# Specify colormap
column_temperature_colormap = sns.color_palette("Spectral_r", as_cmap=True)

temperature_vmin = int(np.around(time_mean_column_temperature.min()))
temperature_vmax = int(np.around(time_mean_column_temperature.max()))

# Plot the CWV data as a function of latitude and longitude
im = ax.contourf(
    longitude,
    latitude,
    time_mean_column_temperature,
    transform=ccrs.PlateCarree(),
    cmap=column_temperature_colormap,
    levels=20,
    vmin=temperature_vmin,
    vmax=temperature_vmax,
)
ax.set_title("November-April mean Column Average Temperature")

# Colorbar
cbar = fig.colorbar(
    ScalarMappable(norm=im.norm, cmap=im.cmap),
    location="bottom",
    fraction=0.1,
    aspect=50,
    shrink=1,
    pad=0.1,
    ticks=range(temperature_vmin, temperature_vmax, 10),
)
cbar.set_label("K")

# Add the land overlay
ax.add_feature(cart.feature.LAND, zorder=100, edgecolor="k")
gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=2,
    color="black",
    alpha=0.0,
    linestyle="-",
)
gl.right_labels = False
gl.top_labels = False
ax.coastlines()

# Plot it
fig.tight_layout()
plt.show()

# %% Parabolic Fits
def parabolic_fit_function(
    meridional_distance, zonal_time_mean_column_water_vapor, evaluation_points
):
    """
    Fit a meridionally-centered parabola to a zonally and time averaged 
    column water vapor field.

    Parameters
    ----------
    meridional_distance : xarray.core.dataarray.DataArray
        Array containing the physical distance (in m) from the equator of each 
        latitude point.
    zonal_time_mean_column_water_vapor : dict
        Dictionary containing gridded zonal and time mean CWV values for 
        each specified region.
    evaluation_points : numpy.ndarray
        Array containing the physical distances from the equator (in m) at 
        which to evaluate the parabolic fit.

    Returns
    -------
    parabolic_fit : dict
        Dictionary containing values of a best-fit parabola for each specified
        region.
    shifted_evaluation_points : dict
        Dictionary containing distance from the equator (in m) locations 
        corresponding to the best-fit parabolas, shifted to be centered 
        around the equator, for each specified region.
    meridional_moisture_gradient_parameter : dict
        Dictionary containing values for the parameter 
        σ_y (as defined in Ahmed 2021), corresponding to each specified 
        region. 

    """

    # Calculate the coefficients specifying the parabolic fit
    parabolic_fit_coefficients = np.polyfit(
        meridional_distance,
        zonal_time_mean_column_water_vapor / INTEGRATED_MOISTURE_STRUCTURE,
        2,
    )

    # Evaluate the parabolic fit on the chosen domain
    parabolic_fit = INTEGRATED_MOISTURE_STRUCTURE * np.polyval(
        parabolic_fit_coefficients, evaluation_points
    )

    # Shift the evaluation points so that the parabolic fit is centered at the equator
    shifted_evaluation_points = (
        evaluation_points - evaluation_points[np.argmax(parabolic_fit)]
    )

    # Calculate the strength of the basic state meridional moisture gradient
    meridional_moisture_gradient_strength = 2 * parabolic_fit_coefficients[0]

    # Convert to match σ_y from Ahmed 2021 with units K kg m^-4
    meridional_moisture_gradient_parameter = (
        meridional_moisture_gradient_strength
        * (LATENT_HEAT / SPECIFIC_HEAT_DRY_AIR)
        * INTEGRATED_MOISTURE_STRUCTURE
        * INTEGRATED_VELOCITY_STRUCTURE
    )

    return (
        parabolic_fit,
        shifted_evaluation_points,
        meridional_moisture_gradient_parameter,
    )


#### Specify regions of interest
LAT_MIN = -18
LAT_MAX = 18
regions = ["Warm Pool", "Indian Ocean", "Maritime Continent", "Western Pacific"]

#### Specify longitudes of each region
LON_MIN = {}
LON_MAX = {}

# Warm Pool
LON_MIN["Warm Pool"] = 60
LON_MAX["Warm Pool"] = 180

# Indian Ocean
LON_MIN["Indian Ocean"] = 60
LON_MAX["Indian Ocean"] = 95

# Maritime Continent
LON_MIN["Maritime Continent"] = 95
LON_MAX["Maritime Continent"] = 145

# Western Pacific
LON_MIN["Western Pacific"] = 145
LON_MAX["Western Pacific"] = 180

# Subset latitudes to the tropics
latitude_tropics = latitude.sel(latitude=slice(LAT_MIN, LAT_MAX))

# Convert from degrees of latitude to meridional distance from the equator
meridional_distance = METERS_PER_DEGREE * latitude_tropics
meridional_distance.name = "Meridional Distance"
meridional_distance.attrs["units"] = "m"

# Specify the domain over which to evaluate the parabolic fit
evaluation_points = np.linspace(-3e6, 3e6, len(latitude_tropics))

time_mean_column_water_vapor = {}
zonal_time_mean_column_water_vapor = {}

parabolic_fit = {}
shifted_evaluation_points = {}
meridional_moisture_gradient_parameter = {}

for region in regions:
    time_mean_column_water_vapor[region] = time_mean_column_water_vapor_global.sel(
        latitude=slice(LAT_MIN, LAT_MAX),
        longitude=slice(LON_MIN[region], LON_MAX[region]),
    )

    zonal_time_mean_column_water_vapor[region] = time_mean_column_water_vapor[
        region
    ].mean(dim="longitude")

    [
        parabolic_fit[region],
        shifted_evaluation_points[region],
        meridional_moisture_gradient_parameter[region],
    ] = parabolic_fit_function(
        meridional_distance,
        zonal_time_mean_column_water_vapor[region],
        evaluation_points,
    )
# Calculate the number to multiply A21's sigma_y value by
meridional_moisture_gradient_multiplier = {}
for region in regions:
    meridional_moisture_gradient_multiplier[region] = (
        meridional_moisture_gradient_parameter[region]
        / A21_MERIDIONAL_MOISTURE_PARAMETER
    )
#### Plotting
plt.style.use("bmh")

# By sub-region
[fig, ax] = plt.subplots(1, 3, figsize=(16, 16))
plt.rcParams.update({"font.size": 24})
plt.suptitle("Zonal and time mean CWV over specified regions", fontsize=42)

regional_title = {
    "Indian Ocean": "Indian Ocean (60°E - 95°E)",
    "Maritime Continent": "Maritime Continent (95°E - 145°E)",
    "Western Pacific": "Western Pacific (145°E - 180)",
}


scale = 1e6

for i, region in enumerate(regions[1:]):
    ax[i].set_title(regional_title[region])
    ax[i].set_xlabel(r" x10$^{3}$ km")
    ax[i].set_ylabel("mm")

    # Raw data
    ax[i].plot(
        meridional_distance / scale,
        zonal_time_mean_column_water_vapor[region],
        label="ERA5",
        color="red",
    )

    # Parabolic Fit
    ax[i].plot(
        evaluation_points / scale,
        parabolic_fit[region],
        label="Parabolic Fit",
        color="black",
        ls="--",
    )

    # Parabolic fit shifted to be symmetric about the equator
    ax[i].plot(
        shifted_evaluation_points[region] / scale,
        parabolic_fit[region],
        label="Shifted Fit",
        color="black",
        ls="-",
    )

    ax[i].legend()
    ax[i].set_xlim(evaluation_points[0] / scale, evaluation_points[-1] / scale)
    ax[i].set_ylim(25, 60)
    ax[i].axvline(x=0, color="darkgray", ls="--", lw=2, alpha=0.75)
    ax[i].xaxis.set_major_locator(MaxNLocator(prune="lower"))
# Plot warm pool
plt.figure(figsize=(16, 16))
plt.rcParams.update({"font.size": 24})
plt.title("Zonal and time mean CWV over the Warm Pool")
plt.xlabel("km")
plt.ylabel("mm")

# Raw data
plt.plot(
    1e-3 * meridional_distance,
    zonal_time_mean_column_water_vapor["Warm Pool"],
    label="ERA5",
    color="red",
)

# Parabolic Fit
plt.plot(
    1e-3 * evaluation_points,
    parabolic_fit["Warm Pool"],
    label="Parabolic Fit",
    color="black",
    ls="--",
)

# Parabolic fit shifted to be symmetric about the equator
plt.plot(
    1e-3 * shifted_evaluation_points["Warm Pool"],
    parabolic_fit["Warm Pool"],
    label="Shifted Fit",
    color="black",
    ls="-",
)

plt.legend()
plt.xlim(1e-3 * evaluation_points[0], 1e-3 * evaluation_points[-1])
plt.ylim(25, 55)
plt.axvline(x=0, color="darkgray", ls="--", lw=2, alpha=0.75)
plt.gca().xaxis.set_major_locator(MaxNLocator(prune="lower"))
plt.gca().set_aspect(6000 / (55 - 25))

#%% Symmetric and Antisymmetric Anomalies

latitude_size = int(len(latitude_tropics))
latitude_size_half = int((len(latitude_tropics) + 1) / 2)

zonal_time_mean_column_water_vapor_anomalies = {}
for _, key in enumerate(zonal_time_mean_column_water_vapor):
    zonal_time_mean_column_water_vapor_anomalies[
        key
    ] = zonal_time_mean_column_water_vapor[key] - zonal_time_mean_column_water_vapor[
        key
    ].mean(
        dim="latitude"
    )
sym_zonal_time_mean_column_water_vapor_anomalies = {}
sym_zonal_time_mean_column_water_vapor_anomalies["Indian Ocean"] = np.empty(
    (latitude_size)
)
sym_zonal_time_mean_column_water_vapor_anomalies["Maritime Continent"] = np.empty(
    (latitude_size)
)
sym_zonal_time_mean_column_water_vapor_anomalies["Western Pacific"] = np.empty(
    (latitude_size)
)

asym_zonal_time_mean_column_water_vapor_anomalies = {}
asym_zonal_time_mean_column_water_vapor_anomalies["Indian Ocean"] = np.empty(
    (latitude_size)
)
asym_zonal_time_mean_column_water_vapor_anomalies["Maritime Continent"] = np.empty(
    (latitude_size)
)
asym_zonal_time_mean_column_water_vapor_anomalies["Western Pacific"] = np.empty(
    (latitude_size)
)


for _, key in enumerate(sym_zonal_time_mean_column_water_vapor_anomalies):
    for i in range(0, latitude_size_half):
        # Symmetric Component - NH
        sym_zonal_time_mean_column_water_vapor_anomalies[key][i] = (
            zonal_time_mean_column_water_vapor_anomalies[key][i]
            + zonal_time_mean_column_water_vapor_anomalies[key][latitude_size - i - 1]
        ) / 2

        # SH
        sym_zonal_time_mean_column_water_vapor_anomalies[key][latitude_size - i - 1] = (
            zonal_time_mean_column_water_vapor_anomalies[key][i]
            + zonal_time_mean_column_water_vapor_anomalies[key][latitude_size - i - 1]
        ) / 2

        # Antisymmetric Component - NH
        asym_zonal_time_mean_column_water_vapor_anomalies[key][i] = (
            zonal_time_mean_column_water_vapor_anomalies[key][i]
            - zonal_time_mean_column_water_vapor_anomalies[key][latitude_size - i - 1]
        ) / 2

        # SH
        asym_zonal_time_mean_column_water_vapor_anomalies[key][
            latitude_size - i - 1
        ] = -(
            (
                zonal_time_mean_column_water_vapor_anomalies[key][i]
                - zonal_time_mean_column_water_vapor_anomalies[key][
                    latitude_size - i - 1
                ]
            )
            / 2
        )
plt.style.use("bmh")
[fig, ax] = plt.subplots(1, 3, figsize=(16, 16))
plt.rcParams.update({"font.size": 24})
plt.suptitle("Symmetric and Anti-Symmetric Components of CWV anomalies", fontsize=42)

regional_title = {}
regional_title["Indian Ocean"] = "Indian Ocean (60°E - 95°E)"
regional_title["Maritime Continent"] = "Maritime Continent (95°E - 145°E)"
regional_title["Western Pacific"] = "Western Pacific (145°E - 180)"

for i, key in enumerate(sym_zonal_time_mean_column_water_vapor_anomalies):
    ax[i].set_title(regional_title[key])
    ax[i].set_xlabel(r" x10$^{3}$ km")
    ax[i].set_ylabel("mm")

    # Raw data
    ax[i].plot(
        meridional_distance / scale,
        zonal_time_mean_column_water_vapor_anomalies[key],
        label="ERA5",
        color="black",
    )

    # Symmetric component
    ax[i].plot(
        meridional_distance / scale,
        sym_zonal_time_mean_column_water_vapor_anomalies[key],
        label="Symmetric",
        color="blue",
        ls="--",
    )

    # Anti-symmetric component
    ax[i].plot(
        meridional_distance / scale,
        asym_zonal_time_mean_column_water_vapor_anomalies[key],
        label="Asymmetric",
        color="red",
        ls="-",
    )

    ax[i].legend()
    ax[i].set_xlim(meridional_distance[0] / scale, meridional_distance[-1] / scale)
    ax[i].set_ylim(-15, 15)
    ax[i].axvline(x=0, color="darkgray", ls="--", lw=2, alpha=0.75)
    ax[i].xaxis.set_major_locator(MaxNLocator(prune="lower"))
#%% Symmetric and Antisymmetric
# Symmetric and Antisymmetric components

latitude_size = int(len(latitude_tropics))
latitude_size_half = int((len(latitude_tropics) + 1) / 2)

sym_zonal_time_mean_column_water_vapor = {}
sym_zonal_time_mean_column_water_vapor["Indian Ocean"] = np.empty((latitude_size))
sym_zonal_time_mean_column_water_vapor["Maritime Continent"] = np.empty((latitude_size))
sym_zonal_time_mean_column_water_vapor["Western Pacific"] = np.empty((latitude_size))

asym_zonal_time_mean_column_water_vapor = {}
asym_zonal_time_mean_column_water_vapor["Indian Ocean"] = np.empty((latitude_size))
asym_zonal_time_mean_column_water_vapor["Maritime Continent"] = np.empty(
    (latitude_size)
)
asym_zonal_time_mean_column_water_vapor["Western Pacific"] = np.empty((latitude_size))


for _, key in enumerate(sym_zonal_time_mean_column_water_vapor):
    for i in range(0, latitude_size_half):
        # Symmetric Component - NH
        sym_zonal_time_mean_column_water_vapor[key][i] = (
            zonal_time_mean_column_water_vapor[key][i]
            + zonal_time_mean_column_water_vapor[key][latitude_size - i - 1]
        ) / 2

        # SH
        sym_zonal_time_mean_column_water_vapor[key][latitude_size - i - 1] = (
            zonal_time_mean_column_water_vapor[key][i]
            + zonal_time_mean_column_water_vapor[key][latitude_size - i - 1]
        ) / 2

        # Antisymmetric Component - NH
        asym_zonal_time_mean_column_water_vapor[key][i] = (
            zonal_time_mean_column_water_vapor[key][i]
            - zonal_time_mean_column_water_vapor[key][latitude_size - i - 1]
        ) / 2

        # SH
        asym_zonal_time_mean_column_water_vapor[key][latitude_size - i - 1] = -(
            (
                zonal_time_mean_column_water_vapor[key][i]
                - zonal_time_mean_column_water_vapor[key][latitude_size - i - 1]
            )
            / 2
        )
plt.style.use("bmh")
[fig, ax] = plt.subplots(1, 3, figsize=(16, 16))
plt.rcParams.update({"font.size": 24})
plt.suptitle("Symmetric and Anti-Symmetric Components of CWV", fontsize=42)

regional_title = {}
regional_title["Indian Ocean"] = "Indian Ocean (60°E - 95°E)"
regional_title["Maritime Continent"] = "Maritime Continent (95°E - 145°E)"
regional_title["Western Pacific"] = "Western Pacific (145°E - 180)"

for i, key in enumerate(sym_zonal_time_mean_column_water_vapor):
    ax[i].set_title(regional_title[key])
    ax[i].set_xlabel(r" x10$^{3}$ km")
    ax[i].set_ylabel("mm")

    # Raw data
    ax[i].plot(
        meridional_distance / scale,
        zonal_time_mean_column_water_vapor[key],
        label="ERA5",
        color="black",
    )

    # Symmetric component
    ax[i].plot(
        meridional_distance / scale,
        sym_zonal_time_mean_column_water_vapor[key],
        label="Symmetric",
        color="blue",
        ls="--",
    )

    # Anti-symmetric component
    ax[i].plot(
        meridional_distance / scale,
        asym_zonal_time_mean_column_water_vapor[key],
        label="Asymmetric",
        color="red",
        ls="-",
    )

    ax[i].legend()
    ax[i].set_xlim(meridional_distance[0] / scale, meridional_distance[-1] / scale)
    ax[i].axvline(x=0, color="darkgray", ls="--", lw=2, alpha=0.75)
    ax[i].xaxis.set_major_locator(MaxNLocator(prune="lower"))
