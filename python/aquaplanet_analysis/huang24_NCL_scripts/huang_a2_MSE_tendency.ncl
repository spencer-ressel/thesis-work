;==================================================
; Calculate column-integrated MSE
; 2023-04-13
; by HUANG Yuanyuan
; @HKUST
;==================================================

  latN     = 40
  latS     = -latN   ; symmetric about the equator

  diri1    = "/localdisk/v050.b/export/yhuangfh/aqua_case_200/" 
  fils1    = systemfunc("ls " + diri1 + "*.nc ")
  f1       = addfiles(fils1, "r")
  hgt      = f1[:]->Z3(:,:,{latS:latN},:)
  time     = hgt&time
  lat      = hgt&lat
  lon      = hgt&lon
  lev      = hgt&lev

  input      = addfiles("/localdisk/v050.b/export/yhuangfh/NCL/aqua_case_10yrs/MSE/mse_800.nc", "r")
  mse        = input[:]->MSE(:,:,:,:)
  mse&time   = time 
  mse&lev    = lev
  mse&lat    = lat 
  mse&lon    = lon  
  printVarSummary(mse)

  ; mse800 = fbindirread("/localdisk/v050.b/export/yhuangfh/NCL/aqua_case_10yrs/MSE/mse_int_800.dat", 0, (/2920,84,288/),"float")
  ; mse800@long_name = "Frozen Moist Static Energy"
  ; mse800@units     = "J/kg"         ; "m^2/s^2"
  ; mse800@info      = "FMSE = (Cp*t + g*z) + Lv*q + Lf*qi"
  ; mse800@short_name= "FMSE"
  ; mse800!0 = "time"
  ; mse800&time = time 
  ; ; mse800!1 = "level"
  ; ; mse800&level = lev
  ; mse800!1 = "lat"
  ; mse800&lat = lat
  ; mse800!2 = "lon"
  ; mse800&lon = lon
  ; printVarSummary(mse800)

;;MSE tendency
  dhdt = center_finite_diff_n (mse,time,False,0,0) ;J/kg/day
  copy_VarMeta(mse, dhdt)
  printVarSummary(dhdt)

;;output  
  ; fbindirwrite("/localdisk/v050.b/export/yhuangfh/NCL/aqua_case_10yrs/MSE/dh_int_dt_800.dat", dhdt)

  fout = addfile("/localdisk/v050.b/export/yhuangfh/NCL/aqua_case_10yrs/MSE/dhdt_800.nc","c")

  setfileoption(fout,"DefineMode",True)
  dimNames = (/"time","lev","lat","lon"/)
  dimSizes = (/2920,32,84,288/)
  dimUnlim = (/ False , False,  False , False /)
  filedimdef(fout,dimNames,dimSizes,dimUnlim)
  filevardef(fout,"dhdt",typeof(dhdt),getvardims(dhdt))
  filevarattdef(fout,"dhdt",dhdt)
  setfileoption(fout,"DefineMode",False)

  fout->dhdt = dhdt

;;column-integrated MSE (tendency)
  psfc     = f1[:]->PS(:,{latS:latN},:)       ; Pa
  hyam     = f1[:]->hyam(0:31)      ; get a coefficiants
  hybm     = f1[:]->hybm(0:31)      ; get b coefficiants
  p0       = 100000.

  linlog = 1

  pm   = dhdt(:,:,:,:)       ; simple way to create a variable with appropriate metadata
  pm   = pres_hybrid_ccm (psfc, p0, hyam, hybm)    ; pm(ntim,klvl,nlat,mlon) top-to-bottom
  pm@long_name = "pressure"
  pm@units     = "Pa"

  delete(hyam)
  delete(hybm)

  pbot   = 100000.
  ptop   = 1000.

  vint = dhdt(time|:,lat|:,lon|:,lev|0)  ; create a variable with metadata
                                      ; reverse pressure order via ::-1 syntax
                                      ; pm and x will span bottom-to-top
  vint = vibeta (pm(time|:,lat|:,lon|:,lev|::-1), \
                  dhdt(time|:,lat|:,lon|:,lev|::-1),linlog,psfc,pbot,ptop)/9.8
  printVarSummary(vint) ;mse(time,lat,lon)

  delete(psfc)
  delete(pm) 

  fbindirwrite("/localdisk/v050.b/export/yhuangfh/NCL/aqua_case_10yrs/MSE/dhdt_int_800.dat", vint)