;;==================================================
;
; 2023-4-14
; calculate horizontal advaction, vertical advection
; then integrate them 
; (time, lev, lat, lon) -> (time, lat, lon)
; by HUANG Yuanyuan
; @HKUST
;
;;==================================================
;;This file still has to be loaded manually    
  load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/diagnostics_cam.ncl"    
;;==================================================

;;discription of the advection function 
;; 0.0174533 = pi/180 = 2pi/360 (angle->radian)
;; lon(2)-lon(1) = delta
;; 6378388 = R (earth radius)
;; for ceter_finite_diff_n, scalar: 
;; dX assumed to be the distance between adjacent points.
;; so we dont need to *2

  function  Advection(T,u0,v0,tau,lev,lat,lon) 
  begin
       ntau   = dimsizes(tau)   
       nlon   = dimsizes(lon)
       nlat   = dimsizes(lat)
       nlev   = dimsizes(lev)
       Tadv   = T

       ; dlon = (lon(2)-lon(1))*0.0174533 
       ; dTdX = new((/ntau,nlev,nlat,nlon/), typeof(T), T@_FillValue)
       ; do nl=0,nlat-1        
       ;   dX = 6378388.*cos(0.0174533*lat(nl))*dlon  
       ;   dTdX(:,:,nl:nl,:) = center_finite_diff_n(T(:,:,nl:nl,:),dX,False,0,3)
       ; end do

       dlat = (lat(2)-lat(1))*0.0174533 
       dTdY = new((/ntau,nlev,nlat,nlon/), typeof(T), T@_FillValue)
       do ml=0,nlon-1
          dY = 6378388.*dlat
          dTdY(:,:,:,ml:ml) = center_finite_diff_n(T(:,:,:,ml:ml),dY,False,0,2)
       end do

       ;Tadv  =  (/ u0*dTdX + v0*dTdY /)
       Tadv_m =  v0*dTdY
       ;Tadv_z =  u0*dTdX
       adv = new((/2,ntau,nlev,nlat,nlon/), typeof(T), T@_FillValue)
       adv(0,:,:,:,:) = Tadv_m
       ;adv(1,:,:,:,:) = Tadv_z

       return(adv)
  end

  latN     = 40
  latS     = -latN   ; symmetric about the equator

;;input nc data
  diri1    = "/mnt/data1/dataset/aqua_case_200/" 
  fils1    = systemfunc("ls " + diri1 + "*.nc ")
  f1       = addfiles(fils1, "r") 
  uwnd     = f1[:]->U(2001:2919,:,{latS:latN},:)
  vwnd     = f1[:]->V(2001:2919,:,{latS:latN},:)

  time     = dble2flt(uwnd&time)
  lat      = dble2flt(uwnd&lat)
  lon      = dble2flt(uwnd&lon)
  lev      = dble2flt(uwnd&lev)

;;input MSE data (J/kg) 3.4*10^5
  input      = addfiles("/mnt/data1/mind-envr_code/NCL/aqua_case_10yrs/MSE/mse_200.nc", "r")
  mse        = input[:]->MSE(2001:2919,:,:,:)
  mse&time   = time 
  mse&lev    = lev
  mse&lat    = lat 
  mse&lon    = lon
  mse@_FillValue = 9.96921e+36  
  printVarSummary(mse)
  print(mse(0,14,3,4))

;;horizontal advection
; 1 function - Use "highly accurate" spherical harmonics 

  ; gridType  = 1   ; 0 means gaussian grid; 1 means regular or fixed grid.
  opt_adv   = 0   ; return only the advected variable
  long_name = "advection of mse"
  units     = "W/m2"

  ; hadv = advect_variable(uwnd,vwnd,mse,gridType,long_name,units,opt_adv)
  ; printVarSummary(hadv)

  ; hadv_ave = dim_avg_n_Wrap(hadv(:,25,:,:), 0)

; 2 function - Using centered-finite_differences (cfd) 

  ; cyclic    = True   ; Grid is global
  ; hadv_cfd   = advect_variable_cfd(uwnd,vwnd,mse,mse&lat,mse&lon,cyclic,long_name,units,opt_adv)
  ; printVarSummary(hadv_cfd)

  ; hadv_cfd_ave = dim_avg_n_Wrap(hadv_cfd(:,25,:,:), 0)

; 3 define a function 

  hadv_def = new((/2,919,32,84,288/),"float")
  hadv_def = Advection(mse,uwnd,vwnd,time,lev,lat,lon) 
  hadv_m   = hadv_def(0,:,:,:,:)
; hadv_z   = hadv_def(1,:,:,:,:)
  copy_VarMeta(mse,hadv_m)
; copy_VarMeta(mse,hadv_z)
  printVarSummary(hadv_def)

  ; hadv_def_ave = dim_avg_n_Wrap(hadv_def(:,25,:,:), 0)  


;;column-integrated MSE
  psfc     = f1[:]->PS(2001:2919,{latS:latN},:)       ; Pa
  hyam     = f1[:]->hyam(0:31)      ; get a coefficiants
  hybm     = f1[:]->hybm(0:31)      ; get b coefficiants
  p0       = 100000.

  linlog = 1

  pm   = hadv_m(:,:,:,:)       ; simple way to create a variable with appropriate metadata
  pm   = pres_hybrid_ccm(psfc, p0, hyam, hybm)    ; pm(ntim,klvl,nlat,mlon) top-to-bottom
  pm@long_name = "pressure"
  pm@units     = "Pa"
  printVarSummary(pm)

  delete(hyam)
  delete(hybm)

  pbot   = 100000.
  ptop   = 1000.

  ; vint = hadv_z(time|:,lat|:,lon|:,lev|0)  ; create a variable with metadata
  ;                                     ; reverse pressure order via ::-1 syntax
  ;                                     ; pm and x will span bottom-to-top
  ; vint = vibeta (pm(time|:,lat|:,lon|:,lev|::-1), \
  ;                 hadv_z(time|:,lat|:,lon|:,lev|::-1),linlog,psfc,pbot,ptop)/9.8
  ; printVarSummary(vint) ;mse(time,lat,lon)

  ; fbindirwrite("/localdisk/v050.b/export/yhuangfh/NCL/aqua_case_10yrs/MSE/projecting/hadv_zonal_200.dat", vint)

  ; delete(vint)


  vint1 = hadv_m(time|:,lat|:,lon|:,lev|0)  ; create a variable with metadata
                                      ; reverse pressure order via ::-1 syntax
                                      ; pm and x will span bottom-to-top
  vint1 = vibeta (pm(time|:,lat|:,lon|:,lev|::-1), \
                  hadv_m(time|:,lat|:,lon|:,lev|::-1),linlog,psfc,pbot,ptop)/9.8
  printVarSummary(vint1) ;mse(time,lat,lon)

  fbindirwrite("/mnt/data1/mind-envr_code/NCL/aqua_case_10yrs/MSE/projecting/hadv_meridional_200_2001_2919.dat", vint1)




