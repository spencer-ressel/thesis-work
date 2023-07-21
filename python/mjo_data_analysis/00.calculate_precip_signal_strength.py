##########################
# Plot wk1999 from TRMM and reanalysis products. Use precipitation
# Save data: raw power/background power/signal strength in wavenumber-frequency domain
# Input: precip timeseries
# Mu-Ting Chien
# 2021.8.2
##################

import sys
sys.path.append('/home/disk/eos9/muting/function/python/')
import mjo_mean_state_diagnostics as MJO
import nclcmaps
import create_my_colormap as mycolor
RWB = mycolor.red_white_blue()
WYGB = mycolor.white_yellow_green_blue()
import numpy as np
from numpy import dtype
import numpy.ma as ma
import numpy.matlib
import math
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.cm import get_cmap
import os
os.environ["PROJ_LIB"] = r'/home/disk/p/muting/anaconda3/share/proj/'#' (location of epsg-->You need to change your own directory, this may vary among machines)'    
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import scipy.signal as signal
from wrf import (getvar, to_np, vertcross, smooth2d,CoordPair, get_basemap, latlon_coords)

######################################
# Caution: Remember to change these if using different data
dir_out = '/home/disk/eos9/muting/KW/'
dir_in = '/home/disk/eos9/muting/data/'
nc = list(['na','data_trmm_daily_2018.nc']) #1998-2018, daily data, 7670 days
plot_raw_spectrum = 0 # Change to 1 if you want to plot
plot_background_spectrum = 0 # Change to 1 if you want to plot
plot_signal_strength = 0 # Change to 1 if you want to plot
Fs_lon = 1/2.5 # longitudinal resolution is 2.5 deg. Change this if diff data
Fs_t = 1 # 1 means 1 day (daily data). Change this if diff data
###################################

t_prev_or_later = 1
if t_prev_or_later == 1: 
    model_list = list(['era5','eraI','merra2','CFSRv2','JRA55'])
    exp = list(['noaa','TRMM','era5','eraI','merra2','CFSRv2','JRA55']) 
    VNAME = list(['OLR','PR','PR','PR','PR','PR','PR'])
    vname = list(['na','prec','pr','pr','pr','pr','pr'])
    new = list(['','_new','','','_new'])
    tmin = 20110102
    tmax = 20181230
    trange = '2011_2018'
nmod = np.size(model_list)


# remember to check whether precip is in mm/day, if in kg/m^2/s, needs to multiply s2d to have the same unit as TRMM
latmax_spec = 15 #max latitude for pr spectrum (wk1999)
lmax = str(latmax_spec)
s2d = 86400

pi = np.pi
spi = '\u03C0'
g = 9.8 #m/s^2
re = 6371*1000 #earth radius (m)
d = np.array([3,6,20]) # mark 3, 6, 20 day in WK1999
he = np.array([2.5,10,25,100,250]) # mark equivalent depth in WK1999
he2 = np.array([8,25,90]) #muting's definition
dname = list(['3d','6d','20d'])
hname = list(['2.5m','10m','25m','100m','250m'])
hname2 = list(['8m','25m','90m'])
m = list(['','_muting'])
pr_spectrum = 1


#######################
# Load land data
dir_in_landsea = '/home/disk/eos9/muting/data/topo/'
ncname = 'NCEP_land_2.5deg.nc'
data = Dataset(dir_in_landsea+ncname, "r", format="NETCDF4")
lat_land = data.variables['lat'][:]
lon_land = data.variables['lon'][:]
land = np.array(data.variables['land'][:].squeeze()) #3d (1, lat, lon), ocean=0, land=1, time doesn't matter
imin2 = np.squeeze( np.argwhere(lat_land==  latmax_spec) )
imax2 = np.squeeze( np.argwhere(lat_land== -latmax_spec) )
land = land[imin2:imax2+1,:]
land = np.flip(land,0) #because lat_land starts from 90~-90 (reverse direction as era5_w.nc: -90~90

for e in range(1,2): #e=1: TRMM precipitation, e=0: NOAA OLR
    '''
    1. Calculate space-time spectrum: pr
    '''
    print(exp[e])
    if pr_spectrum == 1:
        if t_prev_or_later ==1 : 
            clevr = np.arange(1.1,2.1,0.1) #pr
        clev = np.arange(-1,0.1,0.1)
        cticksr = clevr
        fig_dir = dir_out+'figure/publication/All_RA/'+trange+'/pr_spectrum_'+exp[e]+'/'
        if exp[e]=='TRMM':
            nan_value = np.array([-9999.8]) #Change if diff data
            nan_big_small = np.array([0]) #Change if diff data
        elif e>1:
            nan_value = np.array([10*14]) #Change if diff data
            nan_big_small = np.array([1]) #Change if diff data
        if e < 2:
            data = Dataset( dir_in+nc[e], "r", format="NETCDF4") #Change if diff data
        else:
            data = Dataset( dir_in+'combine_reanalysis/'+exp[e]+'/'+exp[e]+'_pr.nc', "r", format="NETCDF4") #Change if diff data
        time = data.variables['time'][:]
        cticks = clev
        lat = data.variables['lat'][:]
        lon = data.variables['lon'][:]
        itmin = np.argwhere(time==tmin).squeeze()
        itmax = np.argwhere(time==tmax).squeeze()
        time = time[itmin:itmax+1]

        # trimm narrower tropical band for space-time spectrum calculation
        latmin_spec = -latmax_spec
        imin = np.argwhere(lat==latmin_spec).squeeze()
        imax = np.argwhere(lat==latmax_spec).squeeze()
        lat_tropics = lat[imin:imax+1]
        
        pr = data.variables[vname[e]][itmin:itmax+1,imin:imax+1,:]             
            
        # replace nan
        pr_nan = MJO.filled_to_nan(pr,nan_value,nan_big_small)
        if e >1: #original unit is kg/m^2/s
            pr_nan = pr_nan*s2d
        if np.sum(np.isnan(pr_nan))==0:
            pr_tropics_ano,cyc = MJO.remove_anncycle_3d(pr_nan,time,lat_tropics,lon) 
        else:
            pr_tropics_ano,cyc = MJO.remove_anncycle_3d(pr_nan,time,lat_tropics,lon)
            print('nan exists!!! can not remove ann cycle')
        nlon = np.size(lon)
        nt = np.size(time)
        nlat_tropics = np.size(lat_tropics)
        
        # mask only ocean data
        V1 = np.empty([nt,nlat_tropics,nlon])
        for IT in range(0,nt): 
            V1[IT,:,:] = np.where(land==1,np.nan,pr_tropics_ano[IT,:,:]) 
        print('finish masking land')

        # separate into symmetric/antisymmetric component
        nlat_half = int((nlat_tropics+1)/2) #include equator
        V1_sym = np.zeros([nt,nlat_half,nlon])
        V1_asy = np.zeros([nt,nlat_half,nlon])
        for ilat in range(0,nlat_half):
            V1_sym[:,ilat,:] = (V1[:,ilat,:]+V1[:,nlat_tropics-ilat-1,:])/2
            V1_asy[:,ilat,:] = -(V1[:,ilat,:]-V1[:,nlat_tropics-ilat-1,:])/2

        # make sure nan becomes zero
        if np.sum(np.isnan(V1_sym))!=0:
            print('has nan, replace by zero ! nannum=')
            print(np.sum(np.isnan(V1_sym[0,:,:])))
            V1_sym = np.where(np.isnan(V1_sym)==1,0,V1_sym)
            V1_asy = np.where(np.isnan(V1_asy)==1,0,V1_asy)   

        # subset into segments in time (96 days, overlap 60 days)
        seglen = 96  
        overlap = 60  
        Hw = 5   #width of Hann window
        n = int(seglen-overlap) #average seglen (not counting the overlap part)
        nseg = math.floor((nt-seglen)/n)+1 
        V1_sym_seg = np.zeros([nseg,seglen,nlat_half,nlon])
        V1_asy_seg = np.zeros([nseg,seglen,nlat_half,nlon])
        HANN = np.concatenate((np.hanning(Hw),np.ones(seglen-Hw*2),np.hanning(Hw)),axis=0)
        HANN = np.tile(HANN,(nlon,nlat_half,1))
        HANN = HANN.transpose(2, 1, 0)
        for iseg in range(0,nseg):
            iseg_n = int(iseg*n)
            V1_sym_seg[iseg,:,:,:] = signal.detrend(V1_sym[iseg*n:iseg*n+seglen,:,:],axis=0)*HANN
            V1_asy_seg[iseg,:,:,:] = signal.detrend(V1_asy[iseg*n:iseg*n+seglen,:,:],axis=0)*HANN

        # calculate space-time spectrum
        FFT_V1_sym = np.zeros([nseg,seglen,nlon,nlat_half],dtype=complex)
        FFT_V1_asy = np.zeros([nseg,seglen,nlon,nlat_half],dtype=complex)
        for iseg in range(0,nseg):
            for ilat in range(0,nlat_half):
                FFT_V1_sym[iseg,:,:,ilat] = np.fft.fft2(V1_sym_seg[iseg,:,ilat,:])/(nlon*seglen)*4 
                FFT_V1_asy[iseg,:,:,ilat] = np.fft.fft2(V1_asy_seg[iseg,:,ilat,:])/(nlon*seglen)*4

        B_sym = FFT_V1_sym*np.conj(FFT_V1_sym)
        B_asy = FFT_V1_asy*np.conj(FFT_V1_asy)
        B_sym = np.real(B_sym)
        B_asy = np.real(B_asy)

        # average over lat and between different segment
        Bm_sym = np.nanmean(np.nanmean(B_sym,3),0)
        Bm_asy = np.nanmean(np.nanmean(B_asy,3),0)
        Bm_sym_shift = np.fft.fftshift( np.fft.fftshift(Bm_sym,axes=1),axes=0 )
        Bm_asy_shift = np.fft.fftshift( np.fft.fftshift(Bm_asy,axes=1),axes=0 )

        freq = np.arange(-seglen/2,seglen/2)*Fs_t/seglen
        zonalwnum = np.arange(-nlon/2,nlon/2)*Fs_lon/nlon*360
        x,y = np.meshgrid(zonalwnum,-freq)

        if plot_raw_spectrum == 1:
            # Plot sym spectrum
            fig = plt.figure(figsize=(12,9))  
            plt.contourf(x,y,np.log10(Bm_sym_shift),cmap=get_cmap('hot_r'),levels = clev,extend='both') 
            plt.rcParams.update({'font.size': 18})
            cb = plt.colorbar(orientation = 'vertical',shrink=.9)
            cb.set_ticks(cticks)
            plt.title(VNAME[e]+' sym (log): '+exp[e]+' '+lmax+'NS '+trange)
            plt.ylabel('freq')
            plt.xlabel('zonal wavenum')
            plt.axis([-15,15,0,0.5])
            plt.xticks([-15,-10,-5,0,5,10,15])
            plt.yticks(np.arange(0,0.55,0.05))
            plt.savefig(fig_dir+VNAME[e]+'_sym_'+exp[e]+'_'+lmax+'NS.png') 
            plt.close()

            # Plot asym spectrum
            fig = plt.figure(figsize=(12,9))
            plt.contourf(x,y,np.log10(Bm_asy_shift),cmap=get_cmap('hot_r'),levels = clev,extend='both') 
            plt.rcParams.update({'font.size': 18})
            cb = plt.colorbar(orientation = 'vertical',shrink=.9)
            cb.set_ticks(cticks)
            plt.title(VNAME[e]+' asy (log): '+exp[e]+' '+lmax+'NS '+trange)
            plt.ylabel('freq')
            plt.xlabel('zonal wavenum')
            plt.axis([-15,15,0,0.5])
            plt.xticks([-15,-10,-5,0,5,10,15])
            plt.yticks(np.arange(0,0.55,0.05))
            plt.savefig(fig_dir+VNAME[e]+'_asy_'+exp[e]+'_'+lmax+'NS.png') 
            plt.close()
            print('finish raw spectrum')

        # 1-2-1 filter in frequency for many times
        fcyc = 15 #How many times applying filters
        fcycs = np.str(fcyc)
        Bm_shift_s = (Bm_sym_shift+Bm_asy_shift)/2
        for k in range(0,fcyc):   
            for i in range(1,seglen-1):
                    Bm_shift_s[i,:] = 1/4*Bm_shift_s[i-1,:] + 1/2*Bm_shift_s[i,:] + 1/4*Bm_shift_s[i+1,:] 
        #
        # 1-2-1 filter in wavenum 
        for k in range(0,fcyc):
            for i in range(1,nlon-1):
                Bm_shift_s[:,i] = 1/4*Bm_shift_s[:,i-1] + 1/2*Bm_shift_s[:,i] + 1/4*Bm_shift_s[:,i+1]   
        
        if plot_background_spectrum == 1:
            # Plot background spectrum-smooth (sym-asy mixed)
            fig = plt.figure(figsize=(12,9))
            plt.contourf(x,y,np.log10(Bm_shift_s),cmap=get_cmap('hot_r'),levels = clev,extend='both') 
            cb = plt.colorbar(orientation = 'vertical',shrink=.9)
            cb.set_ticks(cticks)
            plt.title(VNAME[e]+' (sym+asy)/2: smooth (log) -'+exp[e]+' '+lmax+'NS '+trange)
            plt.ylabel('freq')
            plt.xlabel('zonal wavenum')
            plt.axis([-15,15,0,0.5])
            plt.xticks([-15,-10,-5,0,5,10,15])
            plt.yticks(np.arange(0,0.55,0.05))
            plt.savefig(fig_dir+VNAME[e]+'_background_'+fcycs+'cyc_smooth_'+lmax+'NS.png') 
            plt.close()
            print('finish background spectrum')

        # Calculate signal strength = raw/smooth
        r_sym = Bm_sym_shift/Bm_shift_s
        r_asy =  Bm_asy_shift/Bm_shift_s
        
        # remove artificial signal from satellite: only for olr, not precip
        if e==0:
            aa = np.array([1,-1])
            for a in range(0,2):#2):
                iymin = np.argwhere(zonalwnum==14).squeeze()
                iymax = np.argwhere(zonalwnum==15).squeeze()
                fmin = 0.1*aa[a]
                fmax = 0.15*aa[a] 
                dmin = np.abs(freq-fmin)
                dmax = np.abs(freq-fmax)
                ixmin = np.argwhere(dmin==np.min(dmin)).squeeze()
                ixmax = np.argwhere(dmax==np.min(dmax)).squeeze()
                if a==0:
                    r_sym[ixmin:ixmax+1,iymin:iymax+1]=0
                elif a==1:
                    r_sym[ixmax:ixmin+1,iymin:iymax+1]=0      
        
        # Plot signal strength  
        if plot_signal_strength == 1:
            color = 'WhiteBlueGreenYellowRed'
            cmap2 = nclcmaps.cmap(color)
            for i in range(1,2): # plot color bar hor(0)/vertical(1)
                for mm in range(1,2): # plot he by muting's definition(1) or daehyun's definition(0)
                    # Plot sym signal strength
                    fig = plt.figure(figsize=(12,9))  
                    cf = plt.contourf(x,y,r_sym,cmap=cmap2,levels = clevr,extend='max') 
                    plt.rcParams.update({'font.size': 18})
                    if i==1:
                        cb = plt.colorbar(orientation = 'vertical',shrink=.9)
                    else:
                        cb = plt.colorbar(orientation = 'horizontal',shrink=.9)
                    cb.set_ticks(cticksr)
                    plt.contour(cf, colors='g',linewidths=0.5)
                    # Mark 3, 6, 20 day period:
                    for dd in range(0,np.size(d)):
                        plt.plot([-15,15], [1/d[dd],1/d[dd]], 'k',linewidth=1, linestyle=':')#'dashed')
                        plt.text(-14.8,1/d[dd]+0.01,dname[dd], fontsize=15)
                    # Mark CCKW dispersion relationship:
                    xloc = np.array([12,12,12,4.9,2.2])
                    yloc = np.array([0.145,0.29,0.47,0.47,0.47])
                    xloc2 = np.array([12,12,4.9])
                    yloc2 = np.array([0.29,0.47,0.47])
                    if mm==1:
                        cp = (g*he2)**0.5
                        zwnum_goal = 0.5/s2d/cp*2*np.pi*re
                        for hh in range(0,np.size(he2)):
                            plt.plot([0,zwnum_goal[hh]],[0,0.5],'k',linewidth=1,linestyle=(0,(5,5)))
                            plt.text(xloc2[hh],yloc2[hh],hname2[hh], fontsize=15)               
                    # Mark zwnum == 0:
                    plt.plot([0,0],[0,0.5],'k',linewidth=1,linestyle=':')#'dashed')
                    # Mark CCKW band:
                    # y=s*x
                    s_8 = (g*8)**0.5/(2*np.pi*re)*s2d #slope of he = 8m
                    s_90 = (g*90)**0.5/(2*np.pi*re)*s2d #slope of he = 90m
                    kw_x = np.array([1, 0.05/s_8,     14,  14, 0.4/s_90, 1, 1 ])
                    kw_y = np.array([0.05, 0.05,  14*s_8, 0.4,      0.4,  s_90, 0.05])
                    for kk in range(0,np.size(kw_x)):
                        plt.plot(kw_x,kw_y,'purple',linewidth=1.5,linestyle='solid')
                    plt.title(VNAME[e]+': raw_sym/background -'+exp[e]+' '+lmax+'NS '+trange)
                    plt.ylabel('freq')
                    plt.xlabel('zonal wavenum')
                    plt.axis([-15,15,0,0.5])
                    plt.xticks([-15,-10,-5,0,5,10,15])
                    plt.yticks(np.arange(0,0.6,0.1))
                    plt.tick_params(bottom=True,top=True,left=True,right=True)
                    plt.tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False)
                    plt.tick_params(direction="in")
                    plt.savefig(fig_dir+VNAME[e]+'_ratio_sym_'+lmax+'NS_purplebox.png')
                    plt.close()

            # Plot asy signal strength
            fig = plt.figure(figsize=(12,9))
            cf = plt.contourf(x,y,r_asy,cmap=get_cmap(cmap2),levels = clevr,extend='max') 
            plt.rcParams.update({'font.size': 18})
            cb = plt.colorbar(orientation = 'vertical',shrink=.9)
            cb.set_ticks(cticksr)
            plt.contour(cf, colors='g',linewidths=0.5)
            # Mark 3, 6, 20 day period:
            for dd in range(0,np.size(d)):
                plt.plot([-15,15], [1/d[dd],1/d[dd]], 'k',linewidth=1, linestyle=':')#'dashed')
                plt.text(-14.8,1/d[dd]+0.01,dname[dd], fontsize=15)
            # Mark zwnum == 0:
            plt.plot([0,0],[0,0.5],'k',linewidth=1,linestyle=':')#'dashed')
            # Mark MRG dispersion relationship:
            xloc2 = np.array([12,12,4.9])
            yloc2 = np.array([0.29,0.47,0.47])
            zwnum = np.array([-15,15])/(2*np.pi*re)
            sign = np.array([-1,1])
            cp = (g*he)**0.5
            freq_min = cp*( zwnum[0]/2+((zwnum[0]/2)**2+1)**0.5*sign[0] )*s2d
            freq_max = cp*( zwnum[1]/2+((zwnum[1]/2)**2+1)**0.5*sign[1] )*s2d


            # Mark MRG,EIG band:
            cp8 = (g*8)**0.5
            cp90 = (g*90)**0.5
            cp = np.array([cp8, cp8, cp90, cp90])
            mrg_x = np.array([-10, -1, -1, -10])/(2*np.pi*re)
            mrg_y = cp*( mrg_x/2+((mrg_x/2)**2+1)**0.5*sign[0] )*s2d
            eig_x = np.array([0,14,14,0])/(2*np.pi*re)
            eig_y = cp*( eig_x/2+((eig_x/2)**2+1)**0.5*sign[1] )*s2d
            for kk in range(0,np.size(cp)):
                plt.plot(mrg_x,mrg_y,'grey',linewidth=1.5,linestyle='solid')
                plt.plot(eig_x,eig_y,'grey',linewidth=1.5,linestyle='solid')
            plt.title(VNAME[e]+': raw_asy/background -'+exp[e]+' '+lmax+'NS '+trange)
            plt.ylabel('freq')
            plt.xlabel('zonal wavenum')
            plt.axis([-15,15,0,0.5])
            plt.xticks([-15,-10,-5,0,5,10,15])
            plt.yticks(np.arange(0,0.55,0.05))
            plt.savefig(fig_dir+VNAME[e]+'_ratio_asy_'+lmax+'NS.png')                      
            plt.close()
        
        # initialize
        if e == 1:
            nx = np.size(freq)
            ny = np.size(zonalwnum)
            r_sym_all = np.empty([nx,ny,nmod+1])
            r_asy_all = np.empty([nx,ny,nmod+1])
            Bm_shift_s_all = np.empty([nx,ny,nmod+1])
            Bm_sym_shift_all = np.empty([nx,ny,nmod+1])
            Bm_asy_shift_all= np.empty([nx,ny,nmod+1])
        
        if e == 1: # put TRMM data the last
            r_sym_all[:,:,nmod] = r_sym
            r_asy_all[:,:,nmod] = r_asy
            Bm_shift_s_all[:,:,nmod] = Bm_shift_s
            Bm_sym_shift_all[:,:,nmod] = Bm_sym_shift
            Bm_asy_shift_all[:,:,nmod] = Bm_asy_shift
        else: # put RA data in the front
            r_sym_all[:,:,e-2] = r_sym
            r_asy_all[:,:,e-2] = r_asy
            Bm_shift_s_all[:,:,e-2] = Bm_shift_s
            Bm_sym_shift_all[:,:,e-2] = Bm_sym_shift
            Bm_asy_shift_all[:,:,e-2] = Bm_asy_shift            

# save output   
output = dir_out+'output_data/publication/All_RA/pr_spectrum_all_96_60_H5_'+lmax+'NS_ocn_'+trange+'.nc'
ncout = Dataset(output, 'w', format='NETCDF4')
# define axis size
ncout.createDimension('freq', np.size(freq)) 
ncout.createDimension('zwnum', np.size(zonalwnum))
ncout.createDimension('mod',nmod+1) #include trmm

# create freq axis
freq2 = ncout.createVariable('freq', dtype('double').char, ('freq'))
freq2.long_name = 'frequency'
freq2.units = 'cyc/day'
freq2.axis = 'f'

# create zwnum axis
zwnum2 = ncout.createVariable('zwnum', dtype('double').char, ('zwnum'))
zwnum2.long_name = 'zonal wavenumber'
zwnum2.units = 'cyc/zonal band'
zwnum2.axis = 'k'

# create model axis
mod2 = ncout.createVariable('mod', dtype('double').char, ('mod'))
mod2.long_name = 'Reanalysis and satellite product (ERA5-ERAI-MERRA2-CFSRv2-JRA55-TRMM)' 
mod2.units = 'none'
mod2.axis = 'mod'

# create variable array
r_sym2= ncout.createVariable('r_sym', dtype('double').char, ('freq','zwnum','mod'))
r_sym2.long_name = 'signal strength-symmetric component: ocean only'
r_sym2.units = 'none'  

r_asy2= ncout.createVariable('r_asy', dtype('double').char, ('freq','zwnum','mod'))
r_asy2.long_name = 'signal strength-asymmetric component: ocean only'
r_asy2.units = 'none'  

bg2= ncout.createVariable('background', dtype('double').char, ('freq','zwnum','mod'))
bg2.long_name = 'background spectrum (15 cyc 1-2-1 filter): ocean only'
bg2.units = 'none'  

raw_sym2= ncout.createVariable('raw_sym', dtype('double').char, ('freq','zwnum','mod'))
raw_sym2.long_name = 'raw spectrum-symmetric component: ocean only'
raw_sym2.units = 'mm/day'         
        
raw_asy2= ncout.createVariable('raw_asy', dtype('double').char, ('freq','zwnum','mod'))
raw_asy2.long_name = 'raw spectrum-asymmetric component: ocean only'
raw_asy2.units = 'mm/day'   

x2 = ncout.createVariable('x', dtype('double').char,('freq','zwnum'))
x2.long_name = 'x for plotting cross spectrum results'
x2.units = 'none'  

y2 = ncout.createVariable('y', dtype('double').char,('freq','zwnum'))
y2.long_name = 'y for plotting cross spectrum results'
y2.units = 'none'  

# copy axis from original dataset
freq2[:] = -freq[:]
zwnum2[:] = zonalwnum[:]
r_sym2[:] = r_sym_all[:]
r_asy2[:] = r_asy_all[:]
bg2[:] = Bm_shift_s_all[:]
raw_sym2[:] = Bm_sym_shift_all[:]
raw_asy2[:] = Bm_asy_shift_all[:]      
x2[:] = x[:]
y2[:] = y[:]        
mod2[:] = np.arange(0,nmod+1)
print('finish signal strength')

