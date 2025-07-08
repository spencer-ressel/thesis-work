;==================================================
; Calculate column-integrated MSE
; 2023-04-13
; by HUANG Yuanyuan
; @HKUST
;==================================================
print_clock("MSE & Column MSE")

experiment_SST = "test"
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

if (experiment_SST .eq. "test") then
    diri1 = "/glade/campaign/univ/uwas0114/SST_AQP3_Qobs_27_4K_3h_10y/atm/hist/"
    fils1    = systemfunc("ls " + diri1 + "*h1*.nc ")
    print(experiment_SST + "; " + diri1)
    f1       = addfiles(fils1(0:30), "r")        ; test
end if

latN     = 40
latS     = -latN   ; symmetric about the equator

start_time = 733
end_time = 4409.875


;calculate height above the surface z (m)
; z       = calculate_daily_values(f1[:]->Z3({start_time:end_time},:,{latS:latN},:), "avg", 0, False)
; T       = calculate_daily_values(f1[:]->T({start_time:end_time},:,{latS:latN},:), "avg", 0, False)
; q       = calculate_daily_values(f1[:]->Q({start_time:end_time},:,{latS:latN},:), "avg", 0, False) ;specific humidity
; qi      = calculate_daily_values(f1[:]->CLDICE({start_time:end_time},:,{latS:latN},:), "avg", 0, False) ;grid box averaged cloud ice amount (kg/kg)
z       = calculate_daily_values(f1[:]->Z3(:,:,{latS:latN},:), "avg", 0, False)
T       = calculate_daily_values(f1[:]->T(:,:,{latS:latN},:), "avg", 0, False)
q       = calculate_daily_values(f1[:]->Q(:,:,{latS:latN},:), "avg", 0, False) ;specific humidity
qi      = calculate_daily_values(f1[:]->CLDICE(:,:,{latS:latN},:), "avg", 0, False) ;grid box averaged cloud ice amount (kg/kg)
T@_Fillvalue = 9.96921e+36
q@_Fillvalue = 9.96921e+36
qi@_Fillvalue = 9.96921e+36

g    = 9.8             ; m/s^2
Cp   = 1005.7          ; J/Kg*K ; specific heat at constant pressure for dry air
Lv   = 2.501e6         ; [J/kg]=[m2/s2]  Latent Heat of Vaporization at 0
Lf   = 3.337e5         ; [J/kg]=[m2/s2]  Latent Heat of Sublimation at 0
t0   = 273.15

print_clock("Calculating MSE...")
; MSE  = Cp*T + g*z + Lv*q + Lf*qi    Using -q_i as defined in Huang24 paper
MSE  = Cp*T + g*z + Lv*q - Lf*qi
MSE@long_name = "Frozen Moist Static Energy"
MSE@units     = "J/kg"         ; "m^2/s^2"
; MSE@info      = "FMSE = (Cp*t + g*z) + Lv*q + Lf*qi"
MSE@info      = "FMSE = (Cp*t + g*z) + Lv*q - Lf*qi"
MSE@short_name= "FMSE"
copy_VarCoords(T, MSE)
; printVarSummary(MSE) ;mse(time,lev,lat,lon)

delete(T)
delete(z)
delete(q)
delete(qi)

print_clock("Column-integrating MSE...")
;column-integrated MSE
psfc     = calculate_daily_values(f1[:]->PS({start_time:end_time},{latS:latN},:), "avg", 0, False)       ; Pa
; psfc     = f1[:]->PS(:,{latS:latN},:)
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

vint = MSE(time|:,lat|:,lon|:,lev|0)  ; create a variable with metadata
                                  ; reverse pressure order via ::-1 syntax
                                  ; pm and x will span bottom-to-top
vint = vibeta (pm(time|:,lat|:,lon|:,lev|::-1), \
                  MSE(time|:,lat|:,lon|:,lev|::-1),linlog,psfc,pbot,ptop)/9.8
vint@long_name = "Column-integrated MSE"
vint@units = "J/m^2"
; printVarSummary(vint) ;mse(time,lat,lon)

delete(psfc)
delete(pm)
print_clock("Writing output data...")
output_directory = "/glade/derecho/scratch/sressel/"

setfileoption("nc", "Format", "NetCDF4Classic")
MSE_filename = experiment_SST + "_MSE.nc"
system("/bin/rm -f " + output_directory + MSE_filename)
fout = addfile(output_directory + MSE_filename, "c")

setfileoption(fout, "DefineMode", True)
dimNames = (/"time", "lev", "lat", "lon"/)
dimSizes = (/ dimsizes(MSE&time), dimsizes(MSE&lev), dimsizes(MSE&lat), dimsizes(MSE&lon) /)
dimUnlim = (/ False , False,  False , False /)
filedimdef(fout, dimNames, dimSizes, dimUnlim)
filevardef(fout, "MSE", typeof(MSE), getvardims(MSE))
filevarattdef(fout, "MSE", MSE)
filevardef(fout, "time", typeof(MSE&time), getvardims(MSE&time))
filevarattdef(fout, "time", MSE&time)
filevardef(fout, "lev", typeof(MSE&lev), getvardims(MSE&lev))
filevarattdef(fout, "lev", MSE&lev)
filevardef(fout, "lat", typeof(MSE&lat), getvardims(MSE&lat))
filevarattdef(fout, "lat", MSE&lat)
filevardef(fout, "lon", typeof(MSE&lon), getvardims(MSE&lon))
filevarattdef(fout, "lon", MSE&lon)
setfileoption(fout, "DefineMode", False)

fout->time = (/ MSE&time /)
fout->lev =  (/ MSE&lev /)
fout->lat =  (/ MSE&lat /)
fout->lon =  (/ MSE&lon /)
fout->MSE = MSE

column_MSE_filename = experiment_SST + "_column_MSE.nc"
system("/bin/rm -f " + output_directory + column_MSE_filename)
fout_vint = addfile(output_directory + column_MSE_filename, "c")
print("Output file: " + output_directory + column_MSE_filename)

setfileoption(fout_vint, "DefineMode", True)
dimNames_vint = (/"time", "lat", "lon"/)
dimSizes_vint = (/ dimsizes(vint&time), dimsizes(vint&lat), dimsizes(vint&lon) /)
dimUnlim_vint = (/ False , False , False /)
filedimdef(fout_vint, dimNames_vint, dimSizes_vint, dimUnlim_vint)
filevardef(fout_vint, "MSE", typeof(vint), getvardims(vint))
filevarattdef(fout_vint, "MSE", vint)
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
fout_vint->MSE = vint

print_clock("Finished")
