;;==================================================
;
; 2023-4-13
; Regression map: Column MSE tendency
; by HUANG Yuanyuan
; @HKUST
;
;;==================================================
;;This file still has to be loaded manually
  load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/diagnostics_cam.ncl"
;;==================================================

print_clock("MSE & MSE Tendency Regressions")
; experiment_SST = "-4K"
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

PRECC  = calculate_daily_values(f1[:]->PRECC({start_time:end_time},{-40:40},:), "avg", 0, False)   ; (time,lat,lon)
PRECL  = calculate_daily_values(f1[:]->PRECL({start_time:end_time},{-40:40},:), "avg", 0, False)   ; (time,lat,lon)
preci = 1000.*3600.*(PRECC + PRECL)
preci@long_name = "Total Precipitation (Convective + Large Scale)"
preci@units = "mm/hr"
; preci    = 1000.*3600.*preci_o
copy_VarCoords(PRECC,preci)
; printVarSummary(preci)

time     = preci&time
lat      = preci&lat
lon      = preci&lon
/;
 mse = fbindirread("/localdisk/v050.b/export/yhuangfh/NCL/aqua_case_10yrs/MSE/dhdt_int_200.dat", 0, (/2920,84,288/),"double") ; pay attention to the type
 mse!0 = "time"
 mse&time = time
 mse!1 = "lat"
 mse&lat = lat
 mse!2 = "lon"
 mse&lon = lon
 print(mse(80,50:60,80:90))
;/

input      = addfiles("/glade/derecho/scratch/sressel/" + experiment_SST + "_column_MSE.nc", "r")
MSE        = input[:]->MSE(:,:,:)

input      := addfiles("/glade/derecho/scratch/sressel/" + experiment_SST + "_column_MSE_tendency.nc", "r")
dhdt        = input[:]->dhdt(:,:,:)
; MSE&time   = time
; MSE&lev    = lev
; MSE&lat    = lat
; MSE&lon    = lon
; time     = MSE&time
; lat      = MSE&lat
; lon      = MSE&lon
; lev      = MSE&lev

;;======================================================
;
; define an index for regression:
; the 20-100 day band-pass filtered precipitation (Butterworth filter)
; region: a 10° × 10° square region centered at the equator
;
;;======================================================

print_clock("Filter precipitation...")
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

print_clock("Regress values...")
index    = conform(dhdt, preci_std, 0)

mse_reg  = index*MSE
mse_reg := dim_avg_n_Wrap(mse_reg,0)
copy_VarMeta(dhdt(0,:,:), mse_reg)

dhdt_reg  = index*dhdt
dhdt_reg := dim_avg_n_Wrap(dhdt_reg,0)
copy_VarMeta(dhdt(0,:,:), dhdt_reg)

output_directory = "/glade/derecho/scratch/sressel/"
output_filename = experiment_SST + "_regressed_column_MSE.nc"
print("Output file: " + output_directory + output_filename)
system("/bin/rm -f " + output_directory + output_filename)
setfileoption("nc", "Format", "NetCDF4Classic")
fout = addfile(output_directory + output_filename, "c")

setfileoption(fout, "DefineMode", True)
dimNames = (/"lat", "lon"/)
dimSizes = (/ dimsizes(mse_reg&lat), dimsizes(mse_reg&lon) /)
dimUnlim = (/False , False /)
filedimdef(fout, dimNames, dimSizes, dimUnlim)
filevardef(fout, "MSE", typeof(mse_reg), getvardims(mse_reg))
filevarattdef(fout, "MSE", mse_reg)
filevardef(fout, "lat", typeof(mse_reg&lat), getvardims(mse_reg&lat))
filevarattdef(fout, "lat", mse_reg&lat)
filevardef(fout, "lon", typeof(mse_reg&lon), getvardims(mse_reg&lon))
filevarattdef(fout, "lon", mse_reg&lon)
setfileoption(fout, "DefineMode", False)

fout->lat =  (/ mse_reg&lat /)
fout->lon =  (/ mse_reg&lon /)
fout->MSE = mse_reg

output_filename := experiment_SST + "_regressed_column_MSE_tendency.nc"
system("/bin/rm -f " + output_directory + output_filename)
print("Output file: " + output_directory + output_filename)
setfileoption("nc", "Format", "NetCDF4Classic")
fout := addfile(output_directory + output_filename, "c")

setfileoption(fout, "DefineMode", True)
dimNames := (/"lat", "lon"/)
dimSizes := (/ dimsizes(dhdt_reg&lat), dimsizes(dhdt_reg&lon) /)
dimUnlim := (/False , False /)
filedimdef(fout, dimNames, dimSizes, dimUnlim)
filevardef(fout, "dhdt", typeof(dhdt_reg), getvardims(dhdt_reg))
filevarattdef(fout, "dhdt", dhdt_reg)
filevardef(fout, "lat", typeof(dhdt_reg&lat), getvardims(dhdt_reg&lat))
filevarattdef(fout, "lat", dhdt_reg&lat)
filevardef(fout, "lon", typeof(dhdt_reg&lon), getvardims(dhdt_reg&lon))
filevarattdef(fout, "lon", dhdt_reg&lon)
setfileoption(fout, "DefineMode", False)

fout->lat =  (/ dhdt_reg&lat /)
fout->lon =  (/ dhdt_reg&lon /)
fout->dhdt = dhdt_reg

print_clock("Finished")

