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
print_clock("Moisture Tendency & Column Moisture Tendency")
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

latS = -40
latN = 40

; hgt      = calculate_daily_values(f1[:]->Z3(:,:,{latS:latN},:), "avg", 0, False)
; time     = hgt&time
; lat      = hgt&lat
; lon      = hgt&lon
; lev      = hgt&lev

input      = addfiles("/glade/derecho/scratch/sressel/" + experiment_SST + "_moisture.nc", "r")
moisture        = input[:]->moisture(:,:,:,:)
; moisture&time   = time
; moisture&lev    = lev
; moisture&lat    = lat
; moisture&lon    = lon
time     = moisture&time
lat      = moisture&lat
lon      = moisture&lon
lev      = moisture&lev
; printVarSummary(moisture)

print_clock("Calculating moisture tendency...")
; moisture tendency
dqdt = (1./86400.)*center_finite_diff_n(moisture,time,False,0,0) ;kg/kg/s
copy_VarMeta(moisture, dqdt)
dqdt@units = "kg kg^-1 s^-1"

; printVarSummary(dqdt)

;column-integrated moisture (tendency)
; psfc     = f1[:]->PS(:,{latS:latN},:)       ; Pa
psfc     = calculate_daily_values(f1[:]->PS(:,{latS:latN},:), "avg", 0, False)       ; Pa
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

pbot   = 100000.
ptop   = 1000.

vint_dqdt = dqdt(time|:,lat|:,lon|:,lev|0)  ; create a variable with metadata
                             ; reverse pressure order via ::-1 syntax
                             ; pm and x will span bottom-to-top
vint_dqdt = vibeta (pm(time|:,lat|:,lon|:,lev|::-1), \
          dqdt(time|:,lat|:,lon|:,lev|::-1),linlog,psfc,pbot,ptop)/9.8

vint_dqdt@long_name = "Column-integrated Moisture Tendency"
vint_dqdt@units = "kg m^-2 s^-1"

delete(psfc)
delete(pm)

print_clock("Saving data...")
;output
output_directory = "/glade/derecho/scratch/sressel/"
setfileoption("nc", "Format", "NetCDF4Classic")

; fbindirwrite(output_directory + experiment_SST + "_moisture_tendency.dat", vint)

; Moist Static Energy Tendency
dqdt_filename = experiment_SST + "_moisture_tendency.nc"
system("/bin/rm -f " + output_directory + dqdt_filename)
fout_dqdt = addfile(output_directory + dqdt_filename, "c")

setfileoption(fout_dqdt,"DefineMode",True)
dimNames = (/"time","lev","lat","lon"/)
dimSizes = (/ dimsizes(dqdt&time), dimsizes(dqdt&lev), dimsizes(dqdt&lat), dimsizes(dqdt&lon) /)
dimUnlim = (/ False , False,  False , False /)
filedimdef(fout_dqdt, dimNames, dimSizes, dimUnlim)
filevardef(fout_dqdt, "dqdt", typeof(dqdt), getvardims(dqdt))
filevarattdef(fout_dqdt, "dqdt", dqdt)
filevardef(fout_dqdt, "time", typeof(dqdt&time), getvardims(dqdt&time))
filevarattdef(fout_dqdt, "time", dqdt&time)
filevardef(fout_dqdt, "lev", typeof(dqdt&lev), getvardims(dqdt&lev))
filevarattdef(fout_dqdt, "lev", dqdt&lev)
filevardef(fout_dqdt, "lat", typeof(dqdt&lat), getvardims(dqdt&lat))
filevarattdef(fout_dqdt, "lat", dqdt&lat)
filevardef(fout_dqdt, "lon", typeof(dqdt&lon), getvardims(dqdt&lon))
filevarattdef(fout_dqdt, "lon", dqdt&lon)
setfileoption(fout_dqdt,"DefineMode",False)

fout_dqdt->time = (/ dqdt&time /)
fout_dqdt->lev =  (/ dqdt&lev /)
fout_dqdt->lat =  (/ dqdt&lat /)
fout_dqdt->lon =  (/ dqdt&lon /)
fout_dqdt->dqdt = dqdt

; Vertically-integrated moisture Tendency
dqdt_int_filename = experiment_SST + "_column_moisture_tendency.nc"
system("/bin/rm -f " + output_directory + dqdt_int_filename)
fout_dqdt_int = addfile(output_directory + dqdt_int_filename, "c")
print("Output file: " + output_directory + dqdt_int_filename)

setfileoption(fout_dqdt_int,"DefineMode",True)
dimNames_vint = (/"time", "lat", "lon"/)
dimSizes_vint = (/ dimsizes(vint_dqdt&time), dimsizes(vint_dqdt&lat), dimsizes(vint_dqdt&lon) /)
dimUnlim_vint = (/ False , False , False /)
filedimdef(fout_dqdt_int, dimNames_vint, dimSizes_vint, dimUnlim_vint)
filevardef(fout_dqdt_int, "dqdt", typeof(vint_dqdt), getvardims(vint_dqdt))
filevarattdef(fout_dqdt_int, "dqdt", vint_dqdt)
filevardef(fout_dqdt_int, "time", typeof(vint_dqdt&time), getvardims(vint_dqdt&time))
filevarattdef(fout_dqdt_int, "time", vint_dqdt&time)
filevardef(fout_dqdt_int, "lat", typeof(vint_dqdt&lat), getvardims(vint_dqdt&lat))
filevarattdef(fout_dqdt_int, "lat", vint_dqdt&lat)
filevardef(fout_dqdt_int, "lon", typeof(vint_dqdt&lon), getvardims(vint_dqdt&lon))
filevarattdef(fout_dqdt_int, "lon", vint_dqdt&lon)
setfileoption(fout_dqdt_int,"DefineMode",False)

fout_dqdt_int->time = (/ vint_dqdt&time /)
fout_dqdt_int->lat =  (/ vint_dqdt&lat /)
fout_dqdt_int->lon =  (/ vint_dqdt&lon /)
fout_dqdt_int->dqdt = vint_dqdt

print_clock("Finished")