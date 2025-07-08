;;==================================================
;
; 2023-4-16
; calculate vertical advection then integrate it 
; (time, lev, lat, lon) -> (time, lat, lon)
; by HUANG Yuanyuan
; @HKUST
;
;;==================================================
;;This file still has to be loaded manually    
  load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/diagnostics_cam.ncl"    
;;==================================================

  latN     = 40
  latS     = -latN   ; symmetric about the equator

;;input nc data
  diri1    = "/localdisk/v050.b/export/yhuangfh/aqua_case_200/" 
  fils1    = systemfunc("ls " + diri1 + "*.nc ")
  f1       = addfiles(fils1, "r") 
  uwnd     = f1[:]->U(:,:,{latS:latN},:)
  vwnd     = f1[:]->V(:,:,{latS:latN},:)
  omega    = f1[:]->OMEGA(:,:,{latS:latN},:) ; vertical velocity (pa/s)

  time     = uwnd&time
  lat      = uwnd&lat
  lon      = uwnd&lon
  lev      = uwnd&lev

;;input MSE data (J/kg) 3.4*10^5
  input      = addfiles("/localdisk/v050.b/export/yhuangfh/NCL/aqua_case_10yrs/MSE/mse_200.nc", "r")
  mse        = input[:]->MSE(:,:,:,:)
  mse&time   = time 
  mse&lev    = lev
  mse&lat    = lat 
  mse&lon    = lon
  mse@_FillValue = 9.96921e+36  
  printVarSummary(mse)

;;vertical advection
  dhdp = center_finite_diff_n(mse,lev*100.,False,0,1)  ;lev:Pa
  hadv = omega*dhdp
  copy_VarMeta(mse,hadv)
  printVarSummary(hadv)
  delete(dhdp)

;;column-integrated MSE
  psfc     = f1[:]->PS(:,{latS:latN},:)       ; Pa
  hyam     = f1[:]->hyam(0:31)      ; get a coefficiants
  hybm     = f1[:]->hybm(0:31)      ; get b coefficiants
  p0       = 100000.

  linlog = 1

  pm   = hadv(:,:,:,:)       ; simple way to create a variable with appropriate metadata
  pm   = pres_hybrid_ccm (psfc, p0, hyam, hybm)    ; pm(ntim,klvl,nlat,mlon) top-to-bottom
  pm@long_name = "pressure"
  pm@units     = "Pa"

  delete(hyam)
  delete(hybm)

  pbot   = 100000.
  ptop   = 1000.

  vint = hadv(time|:,lat|:,lon|:,lev|0)  ; create a variable with metadata
                                      ; reverse pressure order via ::-1 syntax
                                      ; pm and x will span bottom-to-top
  vint = vibeta (pm(time|:,lat|:,lon|:,lev|::-1), \
                  hadv(time|:,lat|:,lon|:,lev|::-1),linlog,psfc,pbot,ptop)/9.8
  printVarSummary(vint) ;mse(time,lat,lon)

  delete(psfc)
  delete(pm) 

  fbindirwrite("/localdisk/v050.b/export/yhuangfh/NCL/aqua_case_10yrs/MSE/projecting/adv_h_200.dat", vint)





