######################
# Calculate CAPE
# 2026.4.27
# Mu-Ting Chien
######################
import numpy as np
import xarray as xr
import os
from metpy.calc import cape_cin, dewpoint_from_relative_humidity, parcel_profile
from metpy.units import units

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    # filename="aquaplanet_logs/cape_calculation.log",      # <— write logs here
    # filemode="a",               # "a" = append (default), "w" = overwrite
)

logger = logging.getLogger(__name__)

##############
# 1. Load data
#################
dir_in = '/glade/campaign/univ/uwas0114/processed_pcoord/'
expname_list = list(['-4K','CTL','4K'])
fout_list = list(['m4K','ctl','p4K'])

Ptop_all = np.array([75, 100, 125])*100 # Find tropoppause for each exp!

# for iexp in range(0, 3):#3): # 0, 3
iexp = 2
    
dir_in_sub = dir_in + fout_list[iexp]+'/'

# for iyr in range(3, 6):#6): # 3, 6
iyr = 3

yy_str = f"{iyr:04d}"
logger.info('Start calculating CAPE for :'+ expname_list[iexp]+','+yy_str)

# Load T
fname = dir_in_sub + fout_list[iexp]+'_T_yr'+yy_str+'.nc'
ds    = xr.open_dataset(fname).sel(lat=slice(-15,15))
T = ds['T']
time = ds['time']
lat  = ds['lat']
lon  = ds['lon']
plev = ds['plev']
logger.info('T loaded')

# Load q
fname = dir_in_sub + fout_list[iexp]+'_Q_yr'+yy_str+'.nc'
ds    = xr.open_dataset(fname).sel(lat=slice(-15,15))
q = ds['Q']
logger.info('Q loaded')

# Define CC equation
epsilon = 0.622
e = 6.1094*np.exp( 17.625*(T-273.15)/(T-273.15+243.04) ) # hPa
plev_large = np.tile(plev, (np.size(T,0), np.size(T, 1), np.size(T, 2), 1))
print(np.shape(plev_large))
#for ilev in range(0, np.size(plev)):
#    if ilev == 0:
#        RH = np.empty([np.size(T,0), np.size(T, 1), np.size(T, 2), np.size(T, 3)])
RH = q/( epsilon*e/plev_large )
logger.info("RH calculated")

del q
# Calculate CAPE
RH_r = RH[:,:,:,::-1] * units.dimensionless
del RH
T_C = (T[:,:,:,::-1] - 273.15) * units.degC # unit C
plev_r = plev[::-1] * units.hPa
plev_large_r = plev_large[:,:,:,::-1] * units.hPa

del T

# calculate dewpoint
Td = dewpoint_from_relative_humidity(T_C, RH_r)
del RH_r
logger.info("Dewpoint calculated")

CAPE, CIN = xr.zeros_like(Td).isel(plev=-1, drop=True), xr.zeros_like(Td).isel(plev=-1, drop=True)

# compute parcel temperature
logger.info("Computing parcel temperatures...")
for it in range(0, np.size(Td,0)):
    if (it+1) % 40 == 0:
        logger.info(f"Day {3*(it+1)/24}/{int(3*np.size(Td,0)/24)}")

    for ilat in range(0, np.size(Td, 1)):
        for ilon in range(0, np.size(Td, 2)):
            T_parc = parcel_profile(plev_r, T_C[it,ilat,ilon,0], Td[it,ilat,ilon,0])#.to('degC')

            # calculate surface-based CAPE/CIN
            T_parc_C = (T_parc.values - 273.15)* units.degC
            CAPE_col, CIN_col = cape_cin(plev_r, T_C[it, ilat, ilon], Td[it, ilat, ilon], T_parc_C)
            CAPE[it,ilat,ilon] = CAPE_col.magnitude
            CIN[it,ilat,ilon] = CIN_col.magnitude

#T_parc = parcel_profile(plev_r, T_C[:,:,:,0], Td[:,:,:,0]).to('degC')
logger.info("Parcel temperatures calculated")


logger.info("CAPE, CIN calculated")

####################
# Save CAPE CIN
# dir_out = '/glade/work/muting/KW/output_data/Paper4_revision/CAPE_CIN/'
dir_out = '/glade/u/home/sressel/spencer-scratch/CAPE_CIN/'
os.makedirs(dir_out, exist_ok=True)
fname_out = dir_out + 'CAPE_CIN_'+fout_list[iexp]+'_yr'+yy_str+'_15SN.nc'
da_CAPE = xr.DataArray(CAPE,\
    coords={"time": time, "lat": lat, "lon": lon},\
    dims=("time", "lat", "lon"), name="CAPE")

da_CIN = xr.DataArray(CIN,\
    coords={"time": time, "lat": lat, "lon": lon},\
    dims=("time", "lat", "lon"), name="CIN")

# ds_out = xr.Dataset({"CAPE": (['time', 'lat', 'lon'], da_CAPE.data)},{"CIN": (['time', 'lat', 'lon'], da_CIN.data)},\
#     coords={'time': ds.time.values, 'lat': ds.lat.values, 'lon': ds.lon.values})
ds_out = xr.Dataset({"CAPE": (['time', 'lat', 'lon'], da_CAPE.data), "CIN": (['time', 'lat', 'lon'], da_CIN.data)},coords={'time': ds.time.values, 'lat': ds.lat.values, 'lon': ds.lon.values})
ds_out.to_netcdf(fname_out)

logger.info('Finish calculating CAPE_CIN for :'+ expname_list[iexp]+','+yy_str)