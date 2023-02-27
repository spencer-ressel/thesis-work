################
# Calculate KW vertical structure at every lat-lon grid point
# Input: (1)KW precip timeseries from each RA from 01.kw_filter_RA.py
#        (2)Other raw variable to be regressed: T,u,geopotential,q,\
#             Q (from 02.Q1calculation_byDSE.py)
# Output: CCKW composite vertical structure of T,u,gp,q,Q
# !!! Caution: This code takes a lot of time to run!
# 2021.11.14
# Mu-Ting Chien
###############
import sys
sys.path.append('/home/disk/eos9/muting/function/python/')
import mjo_mean_state_diagnostics as MJO
import numpy as np
from numpy import dtype
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import statsmodels.api as sm


######################
# Cautiion: Chage this if diff data
dir_out = '/home/disk/eos9/muting/KW/'    
DIR = dir_out+'output_data/publication/' #mean state
#######################

t_prev_or_later = 1 #0
if t_prev_or_later == 1: #later
    model_list = list(['era5','merra2','CFSRv2','JRA55'])
    new = list(['','','','_new'])
    tmin = 20110102
    tmax = 20181230
    trange = '2011_2018'
nmod = np.size(model_list)

ilevmin = 1 #Exclude 1000hPa
g = 9.8
lmax = 15


NS_10 = 1
if NS_10 == 1:
    latmax = 10
    SN = '_10SN'
    SN2 = '_10SN'
elif NS_10 == 0:
    latmax = 15
    SN = ''
    SN2 = '_15SN'
latmin = -latmax
    
WP = 1
if WP == 0:
    lonmax = 100
    WP2 = '_IO'
    lonmax_s = '100'
    lonmin = 40
else:
    lonmax = 180 
    WP2 = ''
    lonmax_s = '180'
    lonmin = 40 


# Load land data
fland = dir_out+'output_data/publication/merra2/PC_merra2_QbyDSE_15NS_ocn_new.nc' # Caution!!!
data = Dataset(fland, "r", format="NETCDF4")
land = np.array(data.variables['land'][:].squeeze()) 
lat = data.variables['lat'][:]
lon = data.variables['lon'][:]
nlon = np.size(lon)
nlat = np.size(lat)
del data

ilatmin = np.argwhere(lat==latmin).squeeze()
ilatmax = np.argwhere(lat==latmax).squeeze()
ilonmin = np.argwhere(lon==lonmin).squeeze()
ilonmax = np.argwhere(lon==lonmax).squeeze()

lat_s = lat[ilatmin:ilatmax+1]
lon_s = lon[ilonmin:ilonmax+1]
nlat_s = np.size(lat_s)
nlon_s = np.size(lon_s)

imod = 0 
for im in range(imod,nmod):
    model = model_list[im]
    print(model)
    model_name = model+'_'

    #########################
    # Caution: Change this if diff data
    # Load KW timeseries 
    file_out = dir_out+'output_data/publication/All_RA/pr_15NS_CCKW_filt_'+trange+'.nc'
    data = Dataset( file_out, 'r', format='NETCDF4') 
    time = data.variables['time'][:] #19800101-20181231
    
    itmin2 = np.argwhere(time==tmin).squeeze()
    itmax2 = np.argwhere(time==tmax).squeeze()
    pr_kw = data.variables['pr_kw'][itmin2:itmax2+1,:,:,im]
    time = time[itmin2:itmax2+1]
    nt = np.size(time)
    print(nt)
    del data

    ######################
    # KW composite (regress multiple fields on KW time series)
    #     using reanalysis 
    #     regressing u, v, w, T, Q, H upon KW timeseries
    ####################
    # Caution: Change this if diff data
    # load variables to be regressed (with vertical structure)    
    dire1 = '/home/disk/eos9/muting/data/combine_reanalysis/'+model+'/'
    T_nc = model_name+'T.nc' #1980~2018
    u_nc = model_name+'u.nc'
    if model == 'era5':
        w_nc = model_name+'w.nc'
        q_nc = model_name+'q.nc'
        gz_nc = model_name+'geopotential.nc'
        Tname = 'x'
        uname = 'x'
        hname = 'x'
        wname = 'x'
        qname = 'x'
    else:
        w_nc = model_name+'omega.nc'
        q_nc = model_name+'qv.nc'
        gz_nc = model_name+'gph.nc'      
        Tname = 'T'
        uname = 'u'
        hname = 'gph'
        wname = 'omega'
        qname = 'qv'

    # Load time for each RA
    if model=='era5':
        dir_in = '/home/disk/eos9/muting/from_muting_laptop/combined_eof/input_data/'
        ncname = 'merra2_u850_u200_small.nc'
        data = Dataset(dir_in+ncname, "r", format="NETCDF4")
        time_org = data.variables['time'][:]
    else:
        data = Dataset(dire1+T_nc, "r", format="NETCDF4")
        time_org = data.variables['time'][:]
    itmin = np.argwhere(time_org==tmin).squeeze()
    itmax = np.argwhere(time_org==tmax).squeeze()
    
    lag = 4 #only consider lead/lag 4 days (-4,-3,....+3,+4)
    lags = np.arange(-lag, lag+1,1)
    
    for ilat in range(ilatmin,ilatmax+1):
        for ilon in range(ilonmin,ilonmax+1):
            if ilat==ilatmin and ilon==ilonmin:     
                iref = -1

            if land[ilat,ilon]==1:
                continue
            else:
                iref = iref + 1 #iref = 0 represents the start
            #print(ilat,ilon)
            
            # Find reference point
            pr_kw_ref = pr_kw[:,ilat,ilon] 

            data = Dataset(dire1+T_nc, "r", format="NETCDF4")
            lon_o = data.variables['lon'][:]
            lat_o = data.variables['lat'][:] #-30~30
            if model == 'era5':
                plev = data.variables['lev'][ilevmin:] #1000~100hPa
            else:
                plev = data.variables['plev'][ilevmin:]
            nlev = np.size(plev)
            ilatmin_o = np.argwhere(lat_o==-lmax).squeeze()
            if ilat==ilatmin and ilon==ilonmin:     
                print(ilatmin_o)
            TT = data.variables[Tname][itmin:itmax+1,ilevmin:,ilatmin_o+ilat,ilon] 
            del data

            data = Dataset(dire1+u_nc, "r", format="NETCDF4")
            uu = data.variables[uname][itmin:itmax+1,ilevmin:,ilatmin_o+ilat,ilon]
            del data
            data = Dataset(dire1+q_nc, "r", format="NETCDF4")
            qq = data.variables[qname][itmin:itmax+1,ilevmin:,ilatmin_o+ilat,ilon]
            del data
            data = Dataset(dire1+gz_nc, "r", format="NETCDF4")
            # Caution: make sure the unit of geopotential is geopotential, not geopotential height
            if model!='era5':
                hh = data.variables[hname][itmin:itmax+1,ilevmin:,ilatmin_o+ilat,ilon]*g #Need to change unit!!! (gpm-->gp)
            else:
                hh = data.variables[hname][itmin:itmax+1,ilevmin:,ilatmin_o+ilat,ilon]
            del data

            # Initialize
            Q_KW = np.zeros([2*lag+1,nlev-1])
            T_KW = np.zeros([2*lag+1,nlev-1])
            q_KW = np.zeros([2*lag+1,nlev-1])
            U_KW = np.zeros([2*lag+1,nlev-1])
            H_KW = np.zeros([2*lag+1,nlev-1]) 

            # Calculate mid-point value (to be comparable with vertical levels of Q1)
            Tmid = np.zeros([nt,nlev-1])
            umid = np.zeros([nt,nlev-1])
            qmid = np.zeros([nt,nlev-1])
            hmid = np.zeros([nt,nlev-1])
            for ilev in range(0,nlev-1):
                Tmid[:,ilev] = (TT[:,ilev]+TT[:,ilev+1])/2
                umid[:,ilev] = (uu[:,ilev]+uu[:,ilev+1])/2
                qmid[:,ilev] = (qq[:,ilev]+qq[:,ilev+1])/2
                hmid[:,ilev] = (hh[:,ilev]+hh[:,ilev+1])/2


            # load Q1 to be regressed 
            output = dir_out+'output_data/publication/'+model+'/Q1_1000_100_'+model_name+'byDSE_15S_15N'+new[im]+'.nc'
            data = Dataset(output, "r", format="NETCDF4")
            plev = data.variables['plev'][ilevmin:]
            time_Q = data.variables['time'][:]
            nlev = np.size(plev)
            itminQ = np.argwhere(time_Q==tmin).squeeze()
            itmaxQ = np.argwhere(time_Q==tmax).squeeze()
            Q1 = data.variables['Q1'][itminQ:itmaxQ+1,ilevmin:,ilat,ilon] 
            del data
            
            if iref == 0: 
                # Lag-level regression
                ntk = np.zeros([nt*2-1])
                ntk[0:nt] = np.arange(1,nt+1)
                ntk[nt:] = np.arange(nt-1,0,-1)
                ntk = ntk[nt-1-lag:nt-1+lag+1]

            pr_kw_std = pr_kw_ref.std() # normalize kw std at each location
            #print(pr_kw_std)
            c = ntk*pr_kw_std #normalization factor

            # calculate anomaly
            Q_ref,cyc = MJO.remove_anncycle_2d(Q1,time,plev)
            u_ref,cyc = MJO.remove_anncycle_2d(umid,time,plev)
            q_ref,cyc = MJO.remove_anncycle_2d(qmid,time,plev)
            T_ref,cyc = MJO.remove_anncycle_2d(Tmid,time,plev) 
            h_ref,cyc = MJO.remove_anncycle_2d(hmid,time,plev) 
            
            for ilev in range(0,nlev):

                Q_kw = np.correlate( Q_ref[:,ilev], pr_kw_ref, mode='full')
                T_kw = np.correlate( T_ref[:,ilev], pr_kw_ref, mode='full')
                q_kw = np.correlate( q_ref[:,ilev], pr_kw_ref, mode='full')
                u_kw = np.correlate( u_ref[:,ilev], pr_kw_ref, mode='full')
                H_kw = np.correlate( h_ref[:,ilev], pr_kw_ref, mode='full')

                Q_KW[:,ilev] = Q_kw[nt-1-lag:nt-1+lag+1]/c# this is what you want   
                T_KW[:,ilev] = T_kw[nt-1-lag:nt-1+lag+1]/c # this is what you want 
                q_KW[:,ilev] = q_kw[nt-1-lag:nt-1+lag+1]/c # this is what you want 
                U_KW[:,ilev] = u_kw[nt-1-lag:nt-1+lag+1]/c # this is what you want 
                H_KW[:,ilev] = H_kw[nt-1-lag:nt-1+lag+1]/c # this is what you want  

            ####################
            # Caution: Remember to change this if using diff data
            # Make the vertical levels of MERRA2 and MERRA1 the same as other dataset so that you can save KW vert composite from all RA in one nc file
            # new plev
            p = 0
            plev_new = np.empty([22])
            nlev_new = np.size(plev_new)
            for i in range(0,nlev_new):
                if model == 'merra1' or model == 'merra2':
                    if i==9:
                        print(plev[p])
                        plev_new[i] = 1/2*(plev[p]+plev[p+1])
                        p = p+2
                    else:
                        plev_new[i] = plev[p]
                        p = p+1
                else:
                    if i==19 or i==20 or i==21:
                        plev_new[i] = 1/2*(plev[p]+plev[p+1])
                        p = p+2
                    else:
                        plev_new[i] = plev[p]
                        p = p+1

            plev_goal = np.array([962.5, 937.5, 912.5, 887.5, 862.5,\
                                  837.5, 812.5, 787.5, 762.5, 725,\
                                  675, 625, 575, 525, 475,\
                                  425, 375, 325, 275, 225, \
                                  175, 125])
            ############################
            
            if iref==0 and im == imod: # This is what you are saving in ncfile
                print('initialize')
                Q_KW_new = np.zeros([2*lag+1,nlev_new,nlat,nlon,nmod])
                T_KW_new = np.zeros([2*lag+1,nlev_new,nlat,nlon,nmod])
                q_KW_new = np.zeros([2*lag+1,nlev_new,nlat,nlon,nmod])
                U_KW_new = np.zeros([2*lag+1,nlev_new,nlat,nlon,nmod])
                H_KW_new = np.zeros([2*lag+1,nlev_new,nlat,nlon,nmod])  
            
            ###################
            # Regrid in the vertical to make sure each RA has the same vertical level
            # Caution: Remember to change this if using diff data
            p = 0
            for i in range(0,nlev_new): 
                if model == 'merra1' or model == 'merra2':
                    if i==9:
                        case = 0
                    else:
                        case = 1
                else:
                    if i==19 or i==20 or i==21:
                        case = 0
                    else:
                        case = 1
                        
                if case == 0:
                    Q_KW_new[:,i,ilat,ilon,im] = 1/2*(Q_KW[:,p]+Q_KW[:,p+1])
                    T_KW_new[:,i,ilat,ilon,im] = 1/2*(T_KW[:,p]+T_KW[:,p+1])
                    q_KW_new[:,i,ilat,ilon,im] = 1/2*(q_KW[:,p]+q_KW[:,p+1])
                    U_KW_new[:,i,ilat,ilon,im] = 1/2*(U_KW[:,p]+U_KW[:,p+1])
                    H_KW_new[:,i,ilat,ilon,im] = 1/2*(H_KW[:,p]+H_KW[:,p+1])
                    p = p+2 
                elif case == 1:
                    Q_KW_new[:,i,ilat,ilon,im] = Q_KW[:,p]
                    T_KW_new[:,i,ilat,ilon,im] = T_KW[:,p]
                    q_KW_new[:,i,ilat,ilon,im] = q_KW[:,p]
                    U_KW_new[:,i,ilat,ilon,im] = U_KW[:,p]
                    H_KW_new[:,i,ilat,ilon,im] = H_KW[:,p]
                    p = p+1
            print(Q_KW_new[4,0,ilat,ilon,im])
            print(T_KW_new[4,0,ilat,ilon,im])
            #############################################
                    
for ilat in range(0,nlat):      
    for ilon in range(0,nlon):
        if land[ilat,ilon]==1:
            Q_KW_new[:,:,ilat,ilon,:] = np.nan
            T_KW_new[:,:,ilat,ilon,:] = np.nan
            q_KW_new[:,:,ilat,ilon,:] = np.nan
            U_KW_new[:,:,ilat,ilon,:] = np.nan
            H_KW_new[:,:,ilat,ilon,:] = np.nan

# Only select IO_WP domain (40-180E, 10S-10N)
Q_KW_new2 = Q_KW_new[:,:,ilatmin:ilatmax+1,ilonmin:ilonmax+1,:]
T_KW_new2 = T_KW_new[:,:,ilatmin:ilatmax+1,ilonmin:ilonmax+1,:]
q_KW_new2 = q_KW_new[:,:,ilatmin:ilatmax+1,ilonmin:ilonmax+1,:]
U_KW_new2 = U_KW_new[:,:,ilatmin:ilatmax+1,ilonmin:ilonmax+1,:]
H_KW_new2 = H_KW_new[:,:,ilatmin:ilatmax+1,ilonmin:ilonmax+1,:]
#print(np.shape(Q_KW_new2))

############################################################
# Save kw vertical structure composite  
file_out = dir_out+'output_data/publication/All_RA/IO_WP_SST_not_normalized_RApr/kw_vertical_structure_'+trange+'_lat_lon_RApr_WP.nc' 
ncout = Dataset(file_out, 'w', format='NETCDF4')
# define axis size
ncout.createDimension('plev', nlev_new)
ncout.createDimension('lags', lag*2+1)
ncout.createDimension('lat', nlat_s)
ncout.createDimension('lon', nlon_s)
ncout.createDimension('time', nt)
ncout.createDimension('mod',nmod)
#print(lag*2+1,nlev_new,nlat_s,nlon_s,nmod)

# create plev axis
plev2 = ncout.createVariable('plev', dtype('double').char, ('plev'))
plev2.standard_name = 'plev'
plev2.long_name = 'pressure level (mid point)'    
plev2.units = 'hPa'
plev2.axis = 'Y'
# create lag axis
lag2 = ncout.createVariable('lags', dtype('double').char, ('lags'))
lag2.long_name = 'lag days compared to kw precip timeseries'
lag2.units = 'day'
lag2.axis = 'X'
# create loc axis
lat2 = ncout.createVariable('lat', dtype('double').char, ('lat'))
lat2.long_name = 'latitude'
lat2.units = 'deg'
lat2.axis = 'lat'
# create loc axis
lon2 = ncout.createVariable('lon', dtype('double').char, ('lon'))
lon2.long_name = 'longitude'
lon2.units = 'deg'
lon2.axis = 'lon'
# create time axis
time2 = ncout.createVariable('time', dtype('double').char, ('time'))
time2.long_name = 'time'
time2.units = 'yyyymmdd'
time2.calendar = 'standard'
time2.axis = 'T'


# create model axis
mod2 = ncout.createVariable('mod', dtype('double').char, ('mod'))
mod2.long_name = 'Reanalysis product (ERA5-MERRA2-CFSRv2-JRA55)' 
mod2.units = 'none'
mod2.axis = 'mod'
    
Qout = ncout.createVariable('Q_KW', dtype('double').char, ('lags', 'plev','lat','lon','mod'))
Qout.long_name = 'kw composite vertical structure (lag-lev regression upon kw pr): diabatic heating rate Q1'
Qout.units = '1/day'    
Tout = ncout.createVariable('T_KW', dtype('double').char, ('lags', 'plev','lat','lon','mod'))
Tout.long_name = 'kw composite vertical structure (lag-lev regression upon kw pr): Temperature'
Tout.units = 'K'  
qout = ncout.createVariable('q_KW', dtype('double').char, ('lags', 'plev','lat','lon','mod'))
qout.long_name = 'kw composite vertical structure (lag-lev regression upon kw pr): Specific humidity'
qout.units = 'kg/kg'     
Uout = ncout.createVariable('U_KW', dtype('double').char, ('lags', 'plev','lat','lon','mod'))
Uout.long_name = 'kw composite vertical structure (lag-lev regression upon kw pr): Zonal wind'
Uout.units = 'm/s'     
Hout = ncout.createVariable('H_KW', dtype('double').char, ('lags', 'plev','lat','lon','mod'))
Hout.long_name = 'kw composite vertical structure (lag-lev regression upon kw pr): Geopotential'
Hout.units = 'm^2/s^2'   

# save
Qout[:] = Q_KW_new2[:]
Tout[:] = T_KW_new2[:]
qout[:] = q_KW_new2[:]
Uout[:] = U_KW_new2[:]
Hout[:] = H_KW_new2[:]
lat2[:] = lat_s[:]
lon2[:] = lon_s[:]
lag2[:] = lags[:]
plev2[:] = plev_new[:]
time2[:] = time[:]
mod2[:] = np.arange(0,nmod)
    
print('finish saving KW vertical structure')


