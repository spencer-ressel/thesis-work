;;==================================================
;
; 2023-4-14
; Regression map: LH, SH, <LW>, <SW>,
; horizontal advaction, vertical advection
; by HUANG Yuanyuan
; @HKUST
;
;;==================================================
;;This file still has to be loaded manually
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/diagnostics_cam.ncl"
;;==================================================

print_clock("Budget Term Regressions")
; experiment_SST = "4K"
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
    f1       = addfiles(fils1(0:949), "r")
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

start_time = 733
end_time = 4409.875

; latN     = 40
; latS     = -latN   ; symmetric about the equator

; precipitation rate (time,lat,lon) for calculate the MJO index
PRECC  = calculate_daily_values(f1[:]->PRECC({start_time:end_time},{-40:40},:), "avg", 0, False)   ; (time,lat,lon)
PRECL  = calculate_daily_values(f1[:]->PRECL({start_time:end_time},{-40:40},:), "avg", 0, False)   ; (time,lat,lon)
preci = 1000.*3600.*(PRECC + PRECL)
preci@long_name = "Total Precipitation (Convective + Large Scale)"
preci@units = "mm/hr"
copy_VarCoords(PRECC, preci)

time     = preci&time
lat      = preci&lat
lon      = preci&lon

; latent heat and sensible heat flux
lh       = calculate_daily_values(f1[:]->LHFLX({start_time:end_time},{-40:40},:), "avg", 0, False) ; surface latent heat flux (W/m2)
sh       = calculate_daily_values(f1[:]->SHFLX({start_time:end_time},{-40:40},:), "avg", 0, False) ; surface sensible heat flux (W/m2)

; radiation flux
lw_sfc   = calculate_daily_values(f1[:]->FLNS({start_time:end_time},{-40:40},:), "avg", 0, False)   ; Net longwave flux at surface
lw_tom   = calculate_daily_values(f1[:]->FLNT({start_time:end_time},{-40:40},:), "avg", 0, False)   ; Net longwave flux at top of model
sw_sfc   = calculate_daily_values(f1[:]->FSNS({start_time:end_time},{-40:40},:), "avg", 0, False)   ; Net solar flux at surface
sw_tom   = calculate_daily_values(f1[:]->FSNT({start_time:end_time},{-40:40},:), "avg", 0, False)   ; Net solar flux at top of model
lw_net   = lw_sfc - lw_tom ; sfc-tom: positive: heating
sw_net   = sw_tom - sw_sfc ; tom-sfc: positive: heating
copy_VarMeta(lw_sfc,lw_net)
copy_VarMeta(lw_sfc,sw_net)

hadv_input  = addfiles("/glade/derecho/scratch/sressel/" + experiment_SST + "_horizontal_MSE_advection.nc", "r")
hadv        = hadv_input[:]->hadv(:,:,:)

vadv_input  = addfiles("/glade/derecho/scratch/sressel/" + experiment_SST + "_vertical_MSE_advection.nc", "r")
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

; Latent Heating
lh_reg  = index*lh
lh_reg := dim_avg_n_Wrap(lh_reg, 0)
lh_reg@long_name = "Latent Heating"
copy_VarCoords(hadv(0,:,:), lh_reg)
lh_reg@units = "W m^-2"

; Sensible Heating
sh_reg  = index*sh
sh_reg := dim_avg_n_Wrap(sh_reg,0)
copy_VarCoords(hadv(0,:,:), sh_reg)
sh_reg@long_name = "Sensible Heating"
sh_reg@units = "W m^-2"

; Longwave Heating
lw_reg  = index*lw_net
lw_reg := dim_avg_n_Wrap(lw_reg,0)
copy_VarCoords(hadv(0,:,:), lw_reg)
lw_reg@long_name = "Longwave Heating"
lw_reg@units = "W m^-2"

; Shortwave Heating
sw_reg  = index*sw_net
sw_reg := dim_avg_n_Wrap(sw_reg,0)
copy_VarCoords(hadv(0,:,:), sw_reg)
sw_reg@long_name = "Shortwave Heating"
sw_reg@units = "W m^-2"

; Horizontal Advection
hadv_reg  = index*hadv
hadv_reg := dim_avg_n_Wrap(hadv_reg,0)
copy_VarCoords(hadv(0,:,:), hadv_reg)
hadv_reg@long_name = "Horizontal Advection"
hadv_reg@units = "W m^-2"

; Vertical Advection
vadv_reg  = index*vadv
vadv_reg := dim_avg_n_Wrap(vadv_reg,0)
copy_VarCoords(hadv(0,:,:), vadv_reg)
vadv_reg@long_name = "Vertical Advection"
vadv_reg@units = "W m^-2"

print_clock("Saving outputs...")
output_directory = "/glade/derecho/scratch/sressel/"
setfileoption("nc", "Format", "NetCDF4Classic")

; Latent Heating
system("/bin/rm -f " + output_directory + experiment_SST + "_regressed_latent_heating.nc")
fout = addfile(output_directory + experiment_SST + "_regressed_latent_heating.nc", "c")
setfileoption(fout, "DefineMode", True)
dimNames = (/"lat", "lon"/)
dimSizes = (/ dimsizes(lh_reg&lat), dimsizes(lh_reg&lon) /)
dimUnlim = (/False , False /)
filedimdef(fout, dimNames, dimSizes, dimUnlim)
filevardef(fout, "LH", typeof(lh_reg), getvardims(lh_reg))
filevarattdef(fout, "LH", lh_reg)
filevardef(fout, "lat", typeof(lh_reg&lat), getvardims(lh_reg&lat))
filevarattdef(fout, "lat", lh_reg&lat)
filevardef(fout, "lon", typeof(lh_reg&lon), getvardims(lh_reg&lon))
filevarattdef(fout, "lon", lh_reg&lon)
setfileoption(fout, "DefineMode", False)

fout->lat =  (/ lh_reg&lat /)
fout->lon =  (/ lh_reg&lon /)
fout->LH = lh_reg

; Sensible Heating
system("/bin/rm -f " + output_directory + experiment_SST + "_regressed_sensible_heating.nc")
fout := addfile(output_directory + experiment_SST + "_regressed_sensible_heating.nc", "c")
setfileoption(fout, "DefineMode", True)
dimNames := (/"lat", "lon"/)
dimSizes := (/ dimsizes(sh_reg&lat), dimsizes(sh_reg&lon) /)
dimUnlim := (/False , False /)
filedimdef(fout, dimNames, dimSizes, dimUnlim)
filevardef(fout, "SH", typeof(sh_reg), getvardims(sh_reg))
filevarattdef(fout, "SH", sh_reg)
filevardef(fout, "lat", typeof(sh_reg&lat), getvardims(sh_reg&lat))
filevarattdef(fout, "lat", sh_reg&lat)
filevardef(fout, "lon", typeof(sh_reg&lon), getvardims(sh_reg&lon))
filevarattdef(fout, "lon", sh_reg&lon)
setfileoption(fout, "DefineMode", False)

fout->lat =  (/ sh_reg&lat /)
fout->lon =  (/ sh_reg&lon /)
fout->SH = sh_reg

; Longwave Heating
system("/bin/rm -f " + output_directory + experiment_SST + "_regressed_longwave_heating.nc")
fout := addfile(output_directory + experiment_SST + "_regressed_longwave_heating.nc", "c")
setfileoption(fout, "DefineMode", True)
dimNames := (/"lat", "lon"/)
dimSizes := (/ dimsizes(lw_reg&lat), dimsizes(lw_reg&lon) /)
dimUnlim := (/False , False /)
filedimdef(fout, dimNames, dimSizes, dimUnlim)
filevardef(fout, "LW", typeof(lw_reg), getvardims(lw_reg))
filevarattdef(fout, "LW", lw_reg)
filevardef(fout, "lat", typeof(lw_reg&lat), getvardims(lw_reg&lat))
filevarattdef(fout, "lat", lw_reg&lat)
filevardef(fout, "lon", typeof(lw_reg&lon), getvardims(lw_reg&lon))
filevarattdef(fout, "lon", lw_reg&lon)
setfileoption(fout, "DefineMode", False)

fout->lat =  (/ lw_reg&lat /)
fout->lon =  (/ lw_reg&lon /)
fout->LW = lw_reg

; Shortwave Heating
system("/bin/rm -f " + output_directory + experiment_SST + "_regressed_shortwave_heating.nc")
fout := addfile(output_directory + experiment_SST + "_regressed_shortwave_heating.nc", "c")
setfileoption(fout, "DefineMode", True)
dimNames := (/"lat", "lon"/)
dimSizes := (/ dimsizes(sw_reg&lat), dimsizes(sw_reg&lon) /)
dimUnlim := (/False , False /)
filedimdef(fout, dimNames, dimSizes, dimUnlim)
filevardef(fout, "SW", typeof(sw_reg), getvardims(sw_reg))
filevarattdef(fout, "SW", sw_reg)
filevardef(fout, "lat", typeof(sw_reg&lat), getvardims(sw_reg&lat))
filevarattdef(fout, "lat", sw_reg&lat)
filevardef(fout, "lon", typeof(sw_reg&lon), getvardims(sw_reg&lon))
filevarattdef(fout, "lon", sw_reg&lon)
setfileoption(fout, "DefineMode", False)

fout->lat =  (/ sw_reg&lat /)
fout->lon =  (/ sw_reg&lon /)
fout->SW = sw_reg

; Horizontal Advection
system("/bin/rm -f " + output_directory + experiment_SST + "_regressed_horizontal_advection.nc")
fout := addfile(output_directory + experiment_SST + "_regressed_horizontal_advection.nc", "c")
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
system("/bin/rm -f " + output_directory + experiment_SST + "_regressed_vertical_advection.nc")
fout := addfile(output_directory + experiment_SST + "_regressed_vertical_advection.nc", "c")
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