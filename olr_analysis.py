# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 20:47:19 2021

@author: resse
"""

#%% Imports
# Data Analysis Tools
import numpy as np
from scipy import interpolate
import scipy.fft as fft
import pandas as pd
# Plotting tools
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.colors as colors
import cartopy as cart
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
# Modules to read ERA5 data
import netCDF4 as nc
from scipy.io import netcdf

#%% Wind Vector ERA5 Data
#### Wind Vector data from ERA5
# Specify the file location
fname = 'uv_200_850_monthly_mean.nc'

# Load in the data and scale it according to the specified values
dataset = netcdf.netcdf_file(fname, maskandscale=True, mmap=False)
u = dataset.variables['u'][:] # zonal wind vectors
v = dataset.variables['v'][:] # meridional wind vectors
wind_lat = dataset.variables['latitude'][:] # wind vector latitude data
wind_lon = dataset.variables['longitude'][:] # wind vector longitude data
# Create a two dimensional meshgrid of the longitude-latitude data
[WIND_LON,WIND_LAT] = np.meshgrid(wind_lon, wind_lat)
# How many kilometers are in a degree of latitude
km_per_deg_lat = 110.574

down_sample = 15 # Not every data point needs to be plotted - 1 will plot all

# Take the temporal mean of the u wind vector and down sample it
u_850_time_avg = np.mean(u, axis=0)[1,:,:]
u_850_time_avg = u_850_time_avg[::down_sample, ::down_sample]

# Take the temporal mean of the v wind vector and down sample it
v_850_time_avg = np.mean(v, axis=0)[1,:,:]
v_850_time_avg = v_850_time_avg[::down_sample, ::down_sample]

# Down sample the geographic grid data
WIND_LON = WIND_LON[::down_sample, ::down_sample]
WIND_LAT = WIND_LAT[::down_sample, ::down_sample]

#%% OLR Data from Liebmann and Smith (1996)
#### Read in Data
olr_ds = nc.Dataset("olr.day.mean.nc") # Load in the OLR data
olr_full = olr_ds['olr'][:] 
olr_lon_full = olr_ds['lon'][:] # OLR longitude sampling data
olr_lat_full = olr_ds['lat'][:] # OLR latitude sampling data
olr_time_full = olr_ds['time'][:] # OLR temporal sampling data

df = pd.read_csv('time_date_data.csv')
dates_full = df['date']
hours_full = df['hours']

# Reduce the temporal domain of the data based on two dates
# t_start = np.where(dates=='1/1/1979')[0][0]
# t_end = np.where(dates=='1/1/1993')[0][0]
t_start = np.where(dates_full=='6/1/1996')[0][0]
t_end = np.where(dates_full=='6/1/1997')[0][0]
olr_time = olr_time_full[t_start:t_end]
olr_time_adjusted = np.array((olr_time - olr_time[0])/24, dtype=int)

# Reduce the spatial domain by finding out the indices corresponding to the 
# latitude and longitude that we care about
# w_edge = np.where(olr_lon_full == 0)[0][0] 
# e_edge = np.where(olr_lon_full == 360)[0][0] + 1
w_edge = 0
e_edge = -1
n_edge = np.where(olr_lat_full == 15)[0][0]
s_edge = np.where(olr_lat_full == -15)[0][0] + 1
olr_lon = olr_lon_full[w_edge:e_edge]
olr_lat = olr_lat_full[n_edge:s_edge]

# Reduce the full OLR array to the domain specified 
olr = olr_full[t_start:t_end, n_edge:s_edge, w_edge:e_edge]

# Calculate the OLR anomaly from the temporal mean at each location
olr_anomaly = olr - np.mean(olr, axis=0)[np.newaxis, :, :]

#### Plotting
# Define the colorbar
cmap = plt.cm.coolwarm_r
divnorm = colors.TwoSlopeNorm(vmin=-125, vcenter=-40, vmax=50)
# norm = matplotlib.colors.Normalize(vmin=-125.,vmax=50.,clip=False)

# Plot the data on a map projection centered on the international dateline
plt.rcParams.update({'font.size':24})
[fig, ax] = plt.subplots(1, figsize=(24,18),
            subplot_kw={'projection':ccrs.PlateCarree(central_longitude=180)})
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
ax.set_title('OLR Anomalies')
#ax.set_xlabel('Longitude')
#ax.set_ylabel('Latitude')

# Add coastlines and gridlines to the map
ax.coastlines()
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='black', alpha=0.0, linestyle='-')
gl.xlocator = mticker.FixedLocator([80, 120, 160, -160, -120, -80])
gl.ylocator = mticker.FixedLocator([40, 20, 0, -20, -40])
gl.right_labels = False
gl.top_labels = False

vmin = np.min(olr_anomaly)
vmax=np.max(olr_anomaly)
im = ax.contourf(olr_lon, olr_lat, olr_anomaly[0], 
                 transform=ccrs.PlateCarree(), 
                 vmin=vmin, vmax=vmax, extend='both', cmap=cmap, norm=divnorm)
cbar = fig.colorbar(im, ax=ax, location = 'bottom', fraction=0.1,
                    aspect=30, shrink=1, pad = 0.1)
cbar.set_label(r'W m$^{-2}$')

for i in range(1, 151, 5):
    im = ax.contourf(olr_lon, olr_lat, olr_anomaly[i], 
            transform=ccrs.PlateCarree(), 
            vmin=vmin, vmax=vmax, extend='both', cmap=cmap, norm=divnorm)
    text = ax.text(75, 35, 
        "Date: {}".format(dates_full[1675+i]),
        bbox=dict(facecolor='white'))
    plt.pause(0.01)
    text.remove()

plt.show()

# # Plot the wind vectors, scaled and with the chosen colormap
# ax.quiver(WIND_LON, WIND_LAT, u_850_time_avg, v_850_time_avg, 
#           cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), scale=500, 
#           width=0.001)

# plt.show()

#%% Filtered MJO Data
#### Spatially Filtered MJO OLR Data
olr_anomaly = olr - np.mean(olr, axis=0)[np.newaxis, :, :]
olr_anomaly_f = fft.fft2(olr_anomaly, axes=[1,2])
olr_anomaly_f_mjo = olr_anomaly_f.copy()
olr_anomaly_f_mjo[:, :, 10:] = 0
olr_anomaly_mjo = fft.ifft2(olr_anomaly_f_mjo, axes=[1,2])
vmin = np.min(olr_anomaly)
vmax=np.max(olr_anomaly)

#### Plotting
# Define the colorbar
cmap = plt.cm.coolwarm_r
divnorm = colors.TwoSlopeNorm(vmin=-125, vcenter=-40, vmax=50)
# norm = matplotlib.colors.Normalize(vmin=-125.,vmax=50.,clip=False)

# Plot the data on a map projection centered on the international dateline
plt.rcParams.update({'font.size':24})
[fig, axs] = plt.subplots(2, 1, figsize=(24,18),
            subplot_kw={'projection':ccrs.PlateCarree(central_longitude=180)})
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)

axs[0].set_title('OLR Anomalies')

# Add coastlines and gridlines to the map
axs[0].coastlines()
gl0 = axs[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='black', alpha=0.0, linestyle='-')
gl0.xlocator = mticker.FixedLocator([80, 120, 160, -160, -120, -80])
gl0.ylocator = mticker.FixedLocator([40, 20, 0, -20, -40])
gl0.right_labels = False
gl0.top_labels = False

im0 = axs[0].contourf(olr_lon, olr_lat, olr_anomaly[0], 
                 transform=ccrs.PlateCarree(), 
                 vmin=vmin, vmax=vmax, extend='both', cmap=cmap, norm=divnorm)

axs[1].set_title('MJO Filtered OLR Anomalies')
#ax.set_xlabel('Longitude')
#ax.set_ylabel('Latitude')

# Add coastlines and gridlines to the map
axs[1].coastlines()
gl1 = axs[1].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='black', alpha=0.0, linestyle='-')
gl1.xlocator = mticker.FixedLocator([80, 120, 160, -160, -120, -80])
gl1.ylocator = mticker.FixedLocator([40, 20, 0, -20, -40])
gl1.right_labels = False
gl1.top_labels = False

im1 = axs[1].contourf(olr_lon, olr_lat, olr_anomaly_mjo[0], 
                 transform=ccrs.PlateCarree(), 
                 vmin=vmin, vmax=vmax, extend='both', cmap=cmap, norm=divnorm)
cbar = fig.colorbar(im1, ax=axs[:], location = 'right', fraction=0.1,
                    aspect=30, shrink=1, pad = 0.1)
cbar.set_label(r'W m$^{-2}$')

for i in range(1, 151, 5):
    im0 = axs[0].contourf(olr_lon, olr_lat, olr_anomaly[i], 
            transform=ccrs.PlateCarree(), 
            vmin=vmin, vmax=vmax, extend='both', cmap=cmap, norm=divnorm)
    
    im1 = axs[1].contourf(olr_lon, olr_lat, olr_anomaly_mjo[i], 
            transform=ccrs.PlateCarree(), 
            vmin=vmin, vmax=vmax, extend='both', cmap=cmap, norm=divnorm)
    text = ax.text(75, 35, 
        "Date: {}".format(dates_full[1675+i]),
        bbox=dict(facecolor='white'))
    plt.pause(0.01)
    text.remove()

plt.show()



#### Temporally Filtered MJO Time Series Data
# # Define the time series data at 0 deg N, 97.7 deg E
# time_data_16_7 = olr[:,16,7]
# # Convert the time series data to Fourier domain
# freq_data_16_7 = fft.fft(time_data_16_7)

# # Adjust the time axis to be in days since Jan. 1 1979 at 12 pm
# # instead of hours since Jan. 1 1800 at 12 pm
# olr_time_adjusted = olr_time - olr_time[0]
# olr_time_days = olr_time_adjusted/24
# # Define the frequency axis in Fourier space
# omega = fft.fftfreq(freq_data_16_7.size, 1.0)

# # Find the index corresponding to a 60 day period
# low_time_index = np.where(omega > 1/96)[0][0]

# # Find the index corresponding to a 30 day period
# high_time_index = np.where(omega > 1/30)[0][0]

# # Set all values outside of a 30-60 day period equal to zero
# freq_filt = freq_data_16_7.copy()
# freq_filt[0:low_time_index] = 0
# freq_filt[high_time_index:-1-high_time_index] = 0
# freq_filt[-1-low_time_index:-1] = 0

# # Convert the Fourier data back to the time domain
# time_data_filt_16_7 = fft.ifft(freq_filt)

# # Plot time series data and filtered time series data
# plt.figure()
# plt.plot(olr_time_days, time_data_16_7 - np.mean(time_data_16_7))
# plt.plot(olr_time_days, time_data_filt_16_7,'r')
