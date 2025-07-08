;==================================================
; Calculate column-integrated MSE
; 2023-04-13
; by HUANG Yuanyuan
; @HKUST
;==================================================

  latN     = 40
  latS     = -latN   ; symmetric about the equator

;;input nc data
  diri1    = "/localdisk/v050.b/export/yhuangfh/aqua_case_800/" 
  fils1    = systemfunc("ls " + diri1 + "*.nc ")
  f1       = addfiles(fils1, "r")

;;calculate height above the surface z (m)
  hgt       = f1[:]->Z3(:,:,{latS:latN},:)
  PHIS      = f1[:]->PHIS(:,{latS:latN},:) ; m2/s2 Surface geopotential (time,lat,lon) 
  PHIS      = PHIS/9.8 ; m
  phis      = conform(hgt, PHIS, (/0,2,3/)) ;(time,lev,lat,lon)
  z         = hgt-phis
  copy_VarCoords(hgt, z)
  delete(hgt)
  delete(PHIS)
  delete(phis)

  T         = f1[:]->T(:,:,{latS:latN},:)
  q         = f1[:]->Q(:,:,{latS:latN},:) ;specific humidity
  qi        = f1[:]->CLDICE(:,:,{latS:latN},:) ;grid box averaged cloud ice amount (kg/kg)
  T@_Fillvalue = 9.96921e+36
  q@_Fillvalue = 9.96921e+36
  qi@_Fillvalue = 9.96921e+36

  g    = 9.8             ; m/s^2
  Cp   = 1005.7          ; J/Kg*K ; specific heat at constant pressure for dry air
  Lv   = 2.501e6         ; [J/kg]=[m2/s2]  Latent Heat of Vaporization at 0
  Lf   = 3.337e5         ; [J/kg]=[m2/s2]  Latent Heat of Sublimation at 0
  t0   = 273.15

  MSE  = Cp*T + g*z + Lv*q + Lf*qi
  MSE@long_name = "Frozen Moist Static Energy"
  MSE@units     = "J/kg"         ; "m^2/s^2"
  MSE@info      = "FMSE = (Cp*t + g*z) + Lv*q + Lf*qi"
  MSE@short_name= "FMSE"
  copy_VarCoords(T, MSE)
  printVarSummary(MSE) ;mse(time,lev,lat,lon)
  delete(T)
  delete(z)
  delete(q)
  delete(qi)

;;column-integrated MSE
  psfc     = f1[:]->PS(:,{latS:latN},:)       ; Pa
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

  pbot   = 10000.
  ptop   = 1000.

  vint = MSE(time|:,lat|:,lon|:,lev|0)  ; create a variable with metadata
                                      ; reverse pressure order via ::-1 syntax
                                      ; pm and x will span bottom-to-top
  vint = vibeta (pm(time|:,lat|:,lon|:,lev|::-1), \
                  MSE(time|:,lat|:,lon|:,lev|::-1),linlog,psfc,pbot,ptop)/9.8
  printVarSummary(vint) ;mse(time,lat,lon)

  delete(psfc)
  delete(pm)

  printVarSummary(MSE)
  fbindirwrite("/localdisk/v050.b/export/yhuangfh/NCL/aqua_case_10yrs/MSE/mse_200.dat", MSE)
  fbindirwrite("/localdisk/v050.b/export/yhuangfh/NCL/aqua_case_10yrs/MSE/mse_int_200.dat", vint)

  fout = addfile("/localdisk/v050.b/export/yhuangfh/NCL/aqua_case_10yrs/MSE/mse_800.nc","c")

  setfileoption(fout,"DefineMode",True)
  dimNames = (/"time","lev","lat","lon"/)
  dimSizes = (/2920,32,84,288/)
  dimUnlim = (/ False , False,  False , False /)
  filedimdef(fout,dimNames,dimSizes,dimUnlim)
  filevardef(fout,"MSE",typeof(MSE),getvardims(MSE))
  filevarattdef(fout,"MSE",MSE)
  setfileoption(fout,"DefineMode",False)

  fout->MSE = MSE

