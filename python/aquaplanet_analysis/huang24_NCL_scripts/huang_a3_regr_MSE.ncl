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

;;input nc data
  diri1    = "/localdisk/v050.b/export/yhuangfh/aqua_case_200/" 
  fils1    = systemfunc("ls " + diri1 + "*.nc ")
  f1       = addfiles(fils1, "r") 

  preci_o  = f1[:]->PRECT(:,{-40:40},:)   ; (time,lat,lon)
  preci    = 1000.*3600.*preci_o
  copy_VarMeta(preci_o,preci)
  preci@units = "mm/hr"

  time     = preci&time
  lat      = preci&lat
  lon      = preci&lon

  mse = fbindirread("/localdisk/v050.b/export/yhuangfh/NCL/aqua_case_10yrs/MSE/dhdt_int_200.dat", 0, (/2920,84,288/),"double") ; pay attention to the type
  mse!0 = "time"
  mse&time = time 
  mse!1 = "lat"
  mse&lat = lat
  mse!2 = "lon"
  mse&lon = lon
  print(mse(80,50:60,80:90))

;;======================================================
;
; define an index for regression:
; the 20-100 day band-pass filtered precipitation (Butterworth filter)
; region: a 10° × 10° square region centered at the equator
;
;;======================================================

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

  index    = conform(mse, preci_std, 0)

  mse_reg  = index*mse
  mse_reg := dim_avg_n_Wrap(mse_reg,0)
  copy_VarMeta(mse(0,:,:),mse_reg)
  printVarSummary(mse_reg)
  fbindirwrite("/localdisk/v050.b/export/yhuangfh/NCL/aqua_case_10yrs/MSE/reg_dhdt_int_200.dat", mse_reg)


