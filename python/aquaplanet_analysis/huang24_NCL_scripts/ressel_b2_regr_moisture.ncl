;;==================================================
;
; 2023-4-14
; Regression map: evaporation, precipitation, <LW>, <SW>,
; horizontal advaction, vertical advection
; by HUANG Yuanyuan
; @HKUST
;
;;==================================================
;;This file still has to be loaded manually
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/diagnostics_cam.ncl"
;;==================================================

print_clock("Budget Term Regressions")
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

; latN     = 40
; latS     = -latN   ; symmetric about the equator

; precipitation rate (time,lat,lon) for calculate the MJO index
PRECC  = calculate_daily_values(f1[:]->PRECC(:,{-40:40},:), "avg", 0, False)   ; (time,lat,lon)
PRECL  = calculate_daily_values(f1[:]->PRECL(:,{-40:40},:), "avg", 0, False)   ; (time,lat,lon)
preci = 1000.*3600.*(PRECC + PRECL)
preci@long_name = "Total Precipitation (Convective + Large Scale)"
preci@units = "mm/hr"
copy_VarCoords(PRECC, preci)

time     = preci&time
lat      = preci&lat
lon      = preci&lon

; precipitation
precipitation = 998.*(PRECC + PRECL)
precipitation@long_name = "Precipitation"
precipitation@units = "kg m^-2 s^-1"
copy_VarCoords(PRECC, precipitation)

; evaporation
heat_of_vaporization = 2.501e6
evaporation       = (1./heat_of_vaporization)*calculate_daily_values(f1[:]->LHFLX(:,{-40:40},:), "avg", 0, False) ; surface evaporation (kg m^-2 s^-1)
evaporation@units = "kg m^-2 s^-1"

hadv_input  = addfiles("/glade/derecho/scratch/sressel/" + experiment_SST + "_horizontal_moisture_advection.nc", "r")
hadv        = hadv_input[:]->hadv(:,:,:)

vadv_input  = addfiles("/glade/derecho/scratch/sressel/" + experiment_SST + "_vertical_moisture_advection.nc", "r")
vadv        = vadv_input[:]->vadv(:,:,:)

;;======================================================
;
; define an index for regression:
; the 20-100 day band-pass filtered precipitation (Butterworth filter)
; region: a 10° × 10° square region centered at the equator
;
;;======================================================
print_clock("Defining MJO index...")
latS    = -5.       ; specific region
latN    =  5.
lonL    = 175.
lonR    = 185.

ca      = 100       ; band width in days
cb      = 20
fca     = 1.0/ca    ;frequency
fcb     = 1.0/cb
opt     = False

preci_region = preci(:,{latS:latN},{lonL:lonR})
preci_ave    = dim_avg_n_Wrap(preci_region, (/1,2/))

preci_bf     = bw_bandpass_filter(preci_ave,fca,fcb,opt,0)
copy_VarMeta(preci_ave,preci_bf)
preci_bf@long_name = "BW Bandpass: "+cb+"-"+ca+" day"

preci_std    = dim_standardize_n_Wrap(preci_bf,1,0)
copy_VarMeta(preci_ave,preci_std)
preci_std@long_name = "BW Bandpass: "+cb+"-"+ca+" day"

;;=================================================
;
; regression
;
;;=================================================

print_clock("Regressing data...")
; index    = conform(msf800, preci_std, 0)
index    = conform(hadv, preci_std, 0)

; Evaporation
evap_reg  = index*evaporation
evap_reg := dim_avg_n_Wrap(evap_reg, 0)
evap_reg@long_name = "Evaporation"
copy_VarCoords(hadv(0,:,:), evap_reg)
evap_reg@units = "kg m^-2 s^-1"

; Precipitation
precip_reg  = index*precipitation
precip_reg := dim_avg_n_Wrap(precip_reg,0)
copy_VarCoords(hadv(0,:,:), precip_reg)
precip_reg@long_name = "Precipitation"
precip_reg@units = "kg m^-2 s^-1"

; Horizontal Advection
hadv_reg  = index*hadv
hadv_reg := dim_avg_n_Wrap(hadv_reg,0)
copy_VarCoords(hadv(0,:,:), hadv_reg)
hadv_reg@long_name = "Horizontal Advection"
hadv_reg@units = "kg m^-2 s^-1"

; Vertical Advection
vadv_reg  = index*vadv
vadv_reg := dim_avg_n_Wrap(vadv_reg,0)
copy_VarCoords(hadv(0,:,:), vadv_reg)
vadv_reg@long_name = "Vertical Advection"
vadv_reg@units = "kg m^-2 s^-1"

print_clock("Saving outputs...")
output_directory = "/glade/derecho/scratch/sressel/"
setfileoption("nc", "Format", "NetCDF4Classic")

; Evaporation
system("/bin/rm -f " + output_directory + experiment_SST + "_regressed_evaporation.nc")
fout = addfile(output_directory + experiment_SST + "_regressed_evaporation.nc", "c")
setfileoption(fout, "DefineMode", True)
dimNames = (/"lat", "lon"/)
dimSizes = (/ dimsizes(evap_reg&lat), dimsizes(evap_reg&lon) /)
dimUnlim = (/False , False /)
filedimdef(fout, dimNames, dimSizes, dimUnlim)
filevardef(fout, "evaporation", typeof(evap_reg), getvardims(evap_reg))
filevarattdef(fout, "evaporation", evap_reg)
filevardef(fout, "lat", typeof(evap_reg&lat), getvardims(evap_reg&lat))
filevarattdef(fout, "lat", evap_reg&lat)
filevardef(fout, "lon", typeof(evap_reg&lon), getvardims(evap_reg&lon))
filevarattdef(fout, "lon", evap_reg&lon)
setfileoption(fout, "DefineMode", False)

fout->lat =  (/ evap_reg&lat /)
fout->lon =  (/ evap_reg&lon /)
fout->evaporation = evap_reg

; Precipitation
system("/bin/rm -f " + output_directory + experiment_SST + "_regressed_precipitation.nc")
fout := addfile(output_directory + experiment_SST + "_regressed_precipitation.nc", "c")
setfileoption(fout, "DefineMode", True)
dimNames := (/"lat", "lon"/)
dimSizes := (/ dimsizes(precip_reg&lat), dimsizes(precip_reg&lon) /)
dimUnlim := (/False , False /)
filedimdef(fout, dimNames, dimSizes, dimUnlim)
filevardef(fout, "precipitation", typeof(precip_reg), getvardims(precip_reg))
filevarattdef(fout, "precipitation", precip_reg)
filevardef(fout, "lat", typeof(precip_reg&lat), getvardims(precip_reg&lat))
filevarattdef(fout, "lat", precip_reg&lat)
filevardef(fout, "lon", typeof(precip_reg&lon), getvardims(precip_reg&lon))
filevarattdef(fout, "lon", precip_reg&lon)
setfileoption(fout, "DefineMode", False)

fout->lat =  (/ precip_reg&lat /)
fout->lon =  (/ precip_reg&lon /)
fout->precipitation = precip_reg

; Horizontal Advection
system("/bin/rm -f " + output_directory + experiment_SST + "_regressed_horizontal_moisture_advection.nc")
fout := addfile(output_directory + experiment_SST + "_regressed_horizontal_moisture_advection.nc", "c")
setfileoption(fout, "DefineMode", True)
dimNames := (/"lat", "lon"/)
dimSizes := (/ dimsizes(hadv_reg&lat), dimsizes(hadv_reg&lon) /)
dimUnlim := (/False , False /)
filedimdef(fout, dimNames, dimSizes, dimUnlim)
filevardef(fout, "hadv", typeof(hadv_reg), getvardims(hadv_reg))
filevarattdef(fout, "hadv", hadv_reg)
filevardef(fout, "lat", typeof(hadv_reg&lat), getvardims(hadv_reg&lat))
filevarattdef(fout, "lat", hadv_reg&lat)
filevardef(fout, "lon", typeof(hadv_reg&lon), getvardims(hadv_reg&lon))
filevarattdef(fout, "lon", hadv_reg&lon)
setfileoption(fout, "DefineMode", False)

fout->lat =  (/ hadv_reg&lat /)
fout->lon =  (/ hadv_reg&lon /)
fout->hadv = hadv_reg

; Vertical Advection
system("/bin/rm -f " + output_directory + experiment_SST + "_regressed_vertical_moisture_advection.nc")
fout := addfile(output_directory + experiment_SST + "_regressed_vertical_moisture_advection.nc", "c")
setfileoption(fout, "DefineMode", True)
dimNames := (/"lat", "lon"/)
dimSizes := (/ dimsizes(vadv_reg&lat), dimsizes(vadv_reg&lon) /)
dimUnlim := (/False , False /)
filedimdef(fout, dimNames, dimSizes, dimUnlim)
filevardef(fout, "vadv", typeof(vadv_reg), getvardims(vadv_reg))
filevarattdef(fout, "vadv", vadv_reg)
filevardef(fout, "lat", typeof(vadv_reg&lat), getvardims(vadv_reg&lat))
filevarattdef(fout, "lat", vadv_reg&lat)
filevardef(fout, "lon", typeof(vadv_reg&lon), getvardims(vadv_reg&lon))
filevarattdef(fout, "lon", vadv_reg&lon)
setfileoption(fout, "DefineMode", False)

fout->lat =  (/ vadv_reg&lat /)
fout->lon =  (/ vadv_reg&lon /)
fout->vadv = vadv_reg

print_clock("Finished")