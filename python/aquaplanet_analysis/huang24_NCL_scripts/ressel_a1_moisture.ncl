;==================================================
; Calculate column-integrated Moisture
; 2023-04-13
; by HUANG Yuanyuan
; @HKUST
; Modified:
; 2025-04-09
; by Spencer Ressel
; @UW
;==================================================
print_clock("Moisture & Column Moisture")
latN     = 40
latS     = -latN   ; symmetric about the equator

experiment_SST = "test_0K"
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
    f1       = addfiles(fils1(0:10), "r")        ; test
end if

if (experiment_SST .eq. "test_0K") then
    diri1 = "/glade/campaign/univ/uwas0114/SST_AQP3_Qobs_27_3h_20y/atm/hist/"
    fils1    = systemfunc("ls " + diri1 + "*h1*.nc ")
    print(experiment_SST + "; " + diri1)
    f1       = addfiles(fils1(0:10), "r")        ; test
end if

if (experiment_SST .eq. "test_-4K") then
    diri1 = "/glade/campaign/univ/uwas0114/SST_AQP3_Qobs_27_-4K_3h_10y/atm/hist/"
    fils1    = systemfunc("ls " + diri1 + "*h1*.nc ")
    print(experiment_SST + "; " + diri1)
    f1       = addfiles(fils1(0:10), "r")        ; test
end if

start_time = 733
end_time = 4409.875

moisture         = calculate_daily_values(f1[:]->Q({start_time:end_time},:,{latS:latN},:), "avg", 0, False) ;specific humidity
moisture@long_name = "Specific Humidity"
moisture@units = "kg/kg"

print_clock("Column-integrating Moisture...")
;column-integrated moisture
psfc     = calculate_daily_values(f1[:]->PS({start_time:end_time},{latS:latN},:), "avg", 0, False)       ; Pa
; psfc     = f1[:]->PS(:,{latS:latN},:)
hyam     = f1[:]->hyam(0:31)      ; get a coefficiants
hybm     = f1[:]->hybm(0:31)      ; get b coefficiants
p0       = 100000.

linlog = 1

pm   = moisture(:,:,:,:)       ; simple way to create a variable with appropriate metadata
pm   = pres_hybrid_ccm (psfc, p0, hyam, hybm)    ; pm(ntim,klvl,nlat,mlon) top-to-bottom
pm@long_name = "pressure"
pm@units     = "Pa"

delete(hyam)
delete(hybm)

; pbot   = 100000.
; ptop   = 1000.
pbot   = 110000.
ptop   = 0.

vint = moisture(time|:,lat|:,lon|:,lev|0)  ; create a variable with metadata
                                  ; reverse pressure order via ::-1 syntax
                                  ; pm and x will span bottom-to-top
vint = vibeta (pm(time|:,lat|:,lon|:,lev|::-1), \
                  moisture(time|:,lat|:,lon|:,lev|::-1),linlog,psfc,pbot,ptop)/9.8
vint@long_name = "Column-integrated Moisture"
vint@units = "kg/m^2"

delete(psfc)
delete(pm)
print_clock("Writing output data...")
output_directory = "/glade/derecho/scratch/sressel/"

setfileoption("nc", "Format", "NetCDF4Classic")
moisture_filename = experiment_SST + "_moisture.nc"
system("/bin/rm -f " + output_directory + moisture_filename)
fout = addfile(output_directory + moisture_filename, "c")

setfileoption(fout, "DefineMode", True)
dimNames = (/"time", "lev", "lat", "lon"/)
dimSizes = (/ dimsizes(moisture&time), dimsizes(moisture&lev), dimsizes(moisture&lat), dimsizes(moisture&lon) /)
dimUnlim = (/ False , False,  False , False /)
filedimdef(fout, dimNames, dimSizes, dimUnlim)
filevardef(fout, "moisture", typeof(moisture), getvardims(moisture))
filevarattdef(fout, "moisture", moisture)
filevardef(fout, "time", typeof(moisture&time), getvardims(moisture&time))
filevarattdef(fout, "time", moisture&time)
filevardef(fout, "lev", typeof(moisture&lev), getvardims(moisture&lev))
filevarattdef(fout, "lev", moisture&lev)
filevardef(fout, "lat", typeof(moisture&lat), getvardims(moisture&lat))
filevarattdef(fout, "lat", moisture&lat)
filevardef(fout, "lon", typeof(moisture&lon), getvardims(moisture&lon))
filevarattdef(fout, "lon", moisture&lon)
setfileoption(fout, "DefineMode", False)

fout->time = (/ moisture&time /)
fout->lev =  (/ moisture&lev /)
fout->lat =  (/ moisture&lat /)
fout->lon =  (/ moisture&lon /)
fout->moisture = moisture

column_moisture_filename = experiment_SST + "_column_moisture.nc"
system("/bin/rm -f " + output_directory + column_moisture_filename)
fout_vint = addfile(output_directory + column_moisture_filename, "c")
print("Output file: " + output_directory + column_moisture_filename)

setfileoption(fout_vint, "DefineMode", True)
dimNames_vint = (/"time", "lat", "lon"/)
dimSizes_vint = (/ dimsizes(vint&time), dimsizes(vint&lat), dimsizes(vint&lon) /)
dimUnlim_vint = (/ False , False , False /)
filedimdef(fout_vint, dimNames_vint, dimSizes_vint, dimUnlim_vint)
filevardef(fout_vint, "moisture", typeof(vint), getvardims(vint))
filevarattdef(fout_vint, "moisture", vint)
filevardef(fout_vint, "time", typeof(vint&time), getvardims(vint&time))
filevarattdef(fout_vint, "time", vint&time)
filevardef(fout_vint, "lat", typeof(vint&lat), getvardims(vint&lat))
filevarattdef(fout_vint, "lat", vint&lat)
filevardef(fout_vint, "lon", typeof(vint&lon), getvardims(vint&lon))
filevarattdef(fout_vint, "lon", vint&lon)
setfileoption(fout_vint, "DefineMode", False)

fout_vint->time = (/ vint&time /)
fout_vint->lat =  (/ vint&lat /)
fout_vint->lon =  (/ vint&lon /)
fout_vint->moisture = vint

print_clock("Finished")
