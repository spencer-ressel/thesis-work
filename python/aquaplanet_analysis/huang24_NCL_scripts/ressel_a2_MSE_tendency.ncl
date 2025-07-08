;==================================================
; Calculate column-integrated MSE
; 2023-04-13
; by HUANG Yuanyuan
; @HKUST
;==================================================
print_clock("MSE Tendency & Column MSE Tendency")
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

latS = -40
latN = 40

start_time = 733
end_time = 4409.875

input      = addfiles("/glade/derecho/scratch/sressel/" + experiment_SST + "_MSE.nc", "r")
MSE        = input[:]->MSE(:,:,:,:)
; MSE&time   = time
; MSE&lev    = lev
; MSE&lat    = lat
; MSE&lon    = lon
time     = MSE&time
lat      = MSE&lat
lon      = MSE&lon
lev      = MSE&lev
; printVarSummary(MSE)

print_clock("Calculating MSE tendency...")
; MSE tendency
dhdt = (1./86400.)*center_finite_diff_n(MSE,time,False,0,0) ;J/kg/s
copy_VarMeta(MSE, dhdt)
dhdt@units = "J kg^-1 s^-1"

; printVarSummary(dhdt)

;column-integrated MSE (tendency)
; psfc     = f1[:]->PS(:,{latS:latN},:)       ; Pa
psfc     = calculate_daily_values(f1[:]->PS({start_time:end_time},{latS:latN},:), "avg", 0, False)       ; Pa
hyam     = f1[:]->hyam(0:31)      ; get a coefficiants
hybm     = f1[:]->hybm(0:31)      ; get b coefficiants
p0       = 100000.

linlog = 1

pm   = MSE(:,:,:,:)       ; simple way to create a variable with appropriate metadata
pm   = pres_hybrid_ccm (psfc, p0, hyam, hybm)    ; pm(ntim,klvl,nlat,mlon) top-to-bottom
pm@long_name = "pressure"
pm@units     = "Pa"

delete(hyam)
delete(hybm)

pbot   = 100000.
ptop   = 1000.

vint_dhdt = dhdt(time|:,lat|:,lon|:,lev|0)  ; create a variable with metadata
                             ; reverse pressure order via ::-1 syntax
                             ; pm and x will span bottom-to-top
vint_dhdt = vibeta (pm(time|:,lat|:,lon|:,lev|::-1), \
          dhdt(time|:,lat|:,lon|:,lev|::-1),linlog,psfc,pbot,ptop)/9.8
; printVarSummary(vint_dhdt) ;MSE(time,lat,lon)
vint_dhdt@long_name = "Column-integrated MSE Tendency"
vint_dhdt@units = "W m^-2"

delete(psfc)
delete(pm)

print_clock("Saving data...")
;output
output_directory = "/glade/derecho/scratch/sressel/"
setfileoption("nc", "Format", "NetCDF4Classic")

; fbindirwrite(output_directory + experiment_SST + "_MSE_tendency.dat", vint)

; Moist Static Energy Tendency
dhdt_filename = experiment_SST + "_MSE_tendency.nc"
system("/bin/rm -f " + output_directory + dhdt_filename)
fout_dhdt = addfile(output_directory + dhdt_filename, "c")

setfileoption(fout_dhdt,"DefineMode",True)
dimNames = (/"time","lev","lat","lon"/)
dimSizes = (/ dimsizes(dhdt&time), dimsizes(dhdt&lev), dimsizes(dhdt&lat), dimsizes(dhdt&lon) /)
dimUnlim = (/ False , False,  False , False /)
filedimdef(fout_dhdt, dimNames, dimSizes, dimUnlim)
filevardef(fout_dhdt, "dhdt", typeof(dhdt), getvardims(dhdt))
filevarattdef(fout_dhdt, "dhdt", dhdt)
filevardef(fout_dhdt, "time", typeof(dhdt&time), getvardims(dhdt&time))
filevarattdef(fout_dhdt, "time", dhdt&time)
filevardef(fout_dhdt, "lev", typeof(dhdt&lev), getvardims(dhdt&lev))
filevarattdef(fout_dhdt, "lev", dhdt&lev)
filevardef(fout_dhdt, "lat", typeof(dhdt&lat), getvardims(dhdt&lat))
filevarattdef(fout_dhdt, "lat", dhdt&lat)
filevardef(fout_dhdt, "lon", typeof(dhdt&lon), getvardims(dhdt&lon))
filevarattdef(fout_dhdt, "lon", dhdt&lon)
setfileoption(fout_dhdt,"DefineMode",False)

fout_dhdt->time = (/ dhdt&time /)
fout_dhdt->lev =  (/ dhdt&lev /)
fout_dhdt->lat =  (/ dhdt&lat /)
fout_dhdt->lon =  (/ dhdt&lon /)
fout_dhdt->dhdt = dhdt

; Vertically-integrated MSE Tendency
dhdt_int_filename = experiment_SST + "_column_MSE_tendency.nc"
system("/bin/rm -f " + output_directory + dhdt_int_filename)
fout_dhdt_int = addfile(output_directory + dhdt_int_filename, "c")
print("Output file: " + output_directory + dhdt_int_filename)

setfileoption(fout_dhdt_int,"DefineMode",True)
dimNames_vint = (/"time", "lat", "lon"/)
dimSizes_vint = (/ dimsizes(vint_dhdt&time), dimsizes(vint_dhdt&lat), dimsizes(vint_dhdt&lon) /)
dimUnlim_vint = (/ False , False , False /)
filedimdef(fout_dhdt_int, dimNames_vint, dimSizes_vint, dimUnlim_vint)
filevardef(fout_dhdt_int, "dhdt", typeof(vint_dhdt), getvardims(vint_dhdt))
filevarattdef(fout_dhdt_int, "dhdt", vint_dhdt)
filevardef(fout_dhdt_int, "time", typeof(vint_dhdt&time), getvardims(vint_dhdt&time))
filevarattdef(fout_dhdt_int, "time", vint_dhdt&time)
filevardef(fout_dhdt_int, "lat", typeof(vint_dhdt&lat), getvardims(vint_dhdt&lat))
filevarattdef(fout_dhdt_int, "lat", vint_dhdt&lat)
filevardef(fout_dhdt_int, "lon", typeof(vint_dhdt&lon), getvardims(vint_dhdt&lon))
filevarattdef(fout_dhdt_int, "lon", vint_dhdt&lon)
setfileoption(fout_dhdt_int,"DefineMode",False)

fout_dhdt_int->time = (/ vint_dhdt&time /)
fout_dhdt_int->lat =  (/ vint_dhdt&lat /)
fout_dhdt_int->lon =  (/ vint_dhdt&lon /)
fout_dhdt_int->dhdt = vint_dhdt

print_clock("Finished")