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

print_clock("Horizontal & Vertical Advection")

;;discription of the advection function
;; 0.0174533 = pi/180 = 2pi/360 (angle->radian)
;; lon(2)-lon(1) = delta
;; 6378388 = R (earth radius)
;; for ceter_finite_diff_n, scalar:
;; dX assumed to be the distance between adjacent points.
;; so we dont need to *2

  ; function  Advection(T,u0,v0,tau,lev,lat,lon)
  ; begin
  ;      ntau   = dimsizes(tau)
  ;      nlon   = dimsizes(lon)
  ;      nlat   = dimsizes(lat)
  ;      nlev   = dimsizes(lev)
  ;      Tadv   = T

  ;      dlon = (lon(2)-lon(1))*0.0174533
  ;      dTdX = new((/ntau,nlev,nlat,nlon/), typeof(T), T@_FillValue)
  ;      do nl=0,nlat-1
  ;        dX = 6378388.*cos(0.0174533*lat(nl))*dlon
  ;        dTdX(:,:,nl:nl,:) = center_finite_diff_n(T(:,:,nl:nl,:),dX,False,0,3)
  ;      end do

  ;      dlat = (lat(2)-lat(1))*0.0174533
  ;      dTdY = new((/ntau,nlev,nlat,nlon/), typeof(T), T@_FillValue)
  ;      do ml=0,nlon-1
  ;         dY = 6378388.*dlat
  ;         dTdY(:,:,:,ml:ml) = center_finite_diff_n(T(:,:,:,ml:ml),dY,False,0,2)
  ;      end do

  ;      Tadv  =  (/ u0*dTdX + v0*dTdY /)

  ;      return(Tadv)
  ; end

experiment_SST = "-4K"
; input nc data
print_clock("Loading data...")

; -4K
if (experiment_SST .eq. "-4K") then
    diri1 = "/glade/campaign/univ/uwas0114/SST_AQP3_Qobs_27_-4K_3h_10y/atm/hist/"
    fils1    = systemfunc("ls " + diri1 + "*h1*.nc ")
    print(experiment_SST + "; " + diri1)
    f1       = addfiles(fils1(188:1130), "r")
end if

; 0K
if (experiment_SST .eq. "0K") then
    diri1 = "/glade/campaign/univ/uwas0114/SST_AQP3_Qobs_27_3h_20y/atm/hist/"
    fils1    = systemfunc("ls " + diri1 + "*h1*.nc ")
    print(experiment_SST + "; " + diri1)
    f1       = addfiles(fils1(0:1035), "r")
end if

; 4K
if (experiment_SST .eq. "4K") then
    diri1 = "/glade/campaign/univ/uwas0114/SST_AQP3_Qobs_27_4K_3h_10y/atm/hist/"
    fils1    = systemfunc("ls " + diri1 + "*h1*.nc ")
    print(experiment_SST + "; " + diri1)
    f1       = addfiles(fils1, "r")                ; 4K
end if

if (experiment_SST .eq. "test_4K") then
    diri1 = "/glade/campaign/univ/uwas0114/SST_AQP3_Qobs_27_4K_3h_10y/atm/hist/"
    fils1    = systemfunc("ls " + diri1 + "*h1*.nc ")
    print(experiment_SST + "; " + diri1)
    f1       = addfiles(fils1(0:4), "r")        ; test
end if

latN     = 40
latS     = -latN   ; symmetric about the equator


uwnd     = calculate_daily_values(f1[:]->U(:,:,{latS:latN},:), "avg", 0, False)
vwnd     = calculate_daily_values(f1[:]->V(:,:,{latS:latN},:), "avg", 0, False)
omega    = calculate_daily_values(f1[:]->OMEGA(:,:,{latS:latN},:), "avg", 0, False) ; vertical velocity (pa/s)

time     = dble2flt(uwnd&time)
lat      = dble2flt(uwnd&lat)
lon      = dble2flt(uwnd&lon)
lev      = dble2flt(uwnd&lev)

;;input Moisture data (kg m^-2 s^-1) 3.4*10^5
input      = addfiles("/glade/derecho/scratch/sressel/" + experiment_SST + "_moisture.nc", "r")
moisture        = input[:]->moisture(:,:,:,:)
; moisture&time   = time
; moisture&lev    = lev
; moisture&lat    = lat
; moisture&lon    = lon
; moisture@_FillValue = 9.96921e+36
; printVarSummary(moisture)
; print(moisture(0,14,3,4))

print_clock("Calculating advection...")
;;horizontal advection
; 1 function - Use "highly accurate" spherical harmonics

; gridType  = 1   ; 0 means gaussian grid; 1 means regular or fixed grid.
opt_adv   = 0   ; return only the advected variable
long_name = "advection of moisture"
; units     = "W/m2"
units     = "kg kg^-1 s^-1"

; hadv = advect_variable(uwnd,vwnd,moisture,gridType,long_name,units,opt_adv)
; printVarSummary(hadv)

; hadv_ave = dim_avg_n_Wrap(hadv(:,25,:,:), 0)

; 2 function - Using centered-finite_differences (cfd)

cyclic    = True   ; Grid is global
hadv_cfd   = advect_variable_cfd(uwnd, vwnd, moisture, moisture&lat, moisture&lon, cyclic, long_name, units, opt_adv)
; printVarSummary(hadv_cfd)

; hadv_cfd_ave = dim_avg_n_Wrap(hadv_cfd(:,25,:,:), 0)

; 3 define a function

; hadv_def = new((/2920,32,84,288/),"float")
; hadv_def = Advection(moisture,uwnd,vwnd,time,lev,lat,lon)
; copy_VarMeta(moisture,hadv_def)
; printVarSummary(hadv_def)

; hadv_def_ave = dim_avg_n_Wrap(hadv_def(:,25,:,:), 0)

;;vertical advection
dqdp = center_finite_diff_n(moisture,lev*100.,False,0,1)  ;lev:Pa
vadv = omega*dqdp
vadv@long_name = "Vertical Advection of moisture"
vadv@units = "kg kg^-1 s^-1"
copy_VarCoords(hadv_cfd, vadv)
; printVarSummary(vadv)
delete(dqdp)

print_clock("Column integrating...")
;;column-integrated Moisture
; psfc     = f1[:]->PS(:,{latS:latN},:)       ; Pa
psfc     = calculate_daily_values(f1[:]->PS(:,{latS:latN},:), "avg", 0, False)       ; Pa
hyam     = f1[:]->hyam(0:31)      ; get a coefficiants
hybm     = f1[:]->hybm(0:31)      ; get b coefficiants
p0       = 100000.

linlog = 1

pm   = hadv_cfd(:,:,:,:)       ; simple way to create a variable with appropriate metadata
pm   = pres_hybrid_ccm (psfc, p0, hyam, hybm)    ; pm(ntim,klvl,nlat,mlon) top-to-bottom
pm@long_name = "pressure"
pm@units     = "Pa"

delete(hyam)
delete(hybm)

pbot   = 100000.
ptop   = 1000.

; Horizontal Advection
vint_hadv = hadv_cfd(time|:,lat|:,lon|:,lev|0)  ; create a variable with metadata
                                  ; reverse pressure order via ::-1 syntax
                                  ; pm and x will span bottom-to-top
vint_hadv = vibeta (pm(time|:,lat|:,lon|:,lev|::-1), \
              hadv_cfd(time|:,lat|:,lon|:,lev|::-1),linlog,psfc,pbot,ptop)/9.8
vint_hadv@units     = "W m^-2"
;printVarSummary(vint_hadv) ;moisture(time,lat,lon)

; Vertical Advection
vint_vadv = hadv_cfd(time|:,lat|:,lon|:,lev|0)  ; create a variable with metadata
                                  ; reverse pressure order via ::-1 syntax
                                  ; pm and x will span bottom-to-top
vint_vadv = vibeta (pm(time|:,lat|:,lon|:,lev|::-1), \
              vadv(time|:,lat|:,lon|:,lev|::-1),linlog,psfc,pbot,ptop)/9.8
vint_vadv@units = "W m^-2"
; printVarSummary(vint_vadv) ;moisture(time,lat,lon)

print_clock("Saving output...")
output_directory = "/glade/derecho/scratch/sressel/"
setfileoption("nc", "Format", "NetCDF4Classic")

; Horizontal Advection
hadv_filename = experiment_SST + "_horizontal_moisture_advection.nc"
system("/bin/rm -f " + output_directory + hadv_filename)
fout_hadv = addfile(output_directory + hadv_filename, "c")
print("Output file: " + output_directory + hadv_filename)

setfileoption(fout_hadv, "DefineMode", True)
dimNames = (/"time", "lat", "lon"/)
dimSizes = (/ dimsizes(vint_hadv&time),  dimsizes(vint_hadv&lat), dimsizes(vint_hadv&lon) /)
dimUnlim = (/ False, False , False /)
filedimdef(fout_hadv, dimNames, dimSizes, dimUnlim)
filevardef(fout_hadv, "hadv", typeof(vint_hadv), getvardims(vint_hadv))
filevarattdef(fout_hadv, "hadv", vint_hadv)
filevardef(fout_hadv, "time", typeof(vint_hadv&time), getvardims(vint_hadv&time))
filevarattdef(fout_hadv, "time", vint_hadv&time)
filevardef(fout_hadv, "lat", typeof(vint_hadv&lat), getvardims(vint_hadv&lat))
filevarattdef(fout_hadv, "lat", vint_hadv&lat)
filevardef(fout_hadv, "lon", typeof(vint_hadv&lon), getvardims(vint_hadv&lon))
filevarattdef(fout_hadv, "lon", vint_hadv&lon)
setfileoption(fout_hadv, "DefineMode", False)

fout_hadv->time = (/ vint_hadv&time /)
fout_hadv->lat =  (/ vint_hadv&lat /)
fout_hadv->lon =  (/ vint_hadv&lon /)
fout_hadv->hadv = vint_hadv

; Vertical Advection
vadv_filename = experiment_SST + "_vertical_moisture_advection.nc"
system("/bin/rm -f " + output_directory + vadv_filename)
fout_vadv = addfile(output_directory + vadv_filename, "c")
print("Output file: " + output_directory + vadv_filename)

setfileoption(fout_vadv, "DefineMode", True)
dimNames = (/"time", "lat", "lon"/)
dimSizes = (/ dimsizes(vint_vadv&time),  dimsizes(vint_vadv&lat), dimsizes(vint_vadv&lon) /)
dimUnlim = (/ False, False , False /)
filedimdef(fout_vadv, dimNames, dimSizes, dimUnlim)
filevardef(fout_vadv, "vadv", typeof(vint_vadv), getvardims(vint_vadv))
filevarattdef(fout_vadv, "vadv", vint_vadv)
filevardef(fout_vadv, "time", typeof(vint_vadv&time), getvardims(vint_vadv&time))
filevarattdef(fout_vadv, "time", vint_vadv&time)
filevardef(fout_vadv, "lat", typeof(vint_vadv&lat), getvardims(vint_vadv&lat))
filevarattdef(fout_vadv, "lat", vint_vadv&lat)
filevardef(fout_vadv, "lon", typeof(vint_vadv&lon), getvardims(vint_vadv&lon))
filevarattdef(fout_vadv, "lon", vint_vadv&lon)
setfileoption(fout_vadv, "DefineMode", False)

fout_vadv->time = (/ vint_vadv&time /)
fout_vadv->lat =  (/ vint_vadv&lat /)
fout_vadv->lon =  (/ vint_vadv&lon /)
fout_vadv->vadv = vint_vadv

print_clock("Finished")