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

;;input nc data
  diri1    = "/mnt/data1/dataset/aqua_case_200/" ; "/mnt/data1/dataset/aqua_case_500/" "/mnt/data2/aqua_case_800/"
  fils1    = systemfunc("ls " + diri1 + "*.nc ")
  f1       = addfiles(fils1, "r") 

; precipitation rate (time,lat,lon) for calculate the MJO index
  preci_o  = f1[:]->PRECT(:,{-40:40},:) 
  preci    = 1000.*3600.*preci_o
  copy_VarMeta(preci_o,preci)
  preci@units = "mm/hr"

  time     = preci&time
  lat      = preci&lat
  lon      = preci&lon

; ; latent heat and sensible heat flux
;   lh       = f1[:]->LHFLX(:,{-40:40},:) ; surface latent heat flux (W/m2)
;   sh       = f1[:]->SHFLX(:,{-40:40},:) ; surface sensible heat flux (W/m2)

; ; radiation flux
;   lw_sfc   = f1[:]->FLNS(:,{-40:40},:)   ; Net longwave flux at surface
;   lw_tom   = f1[:]->FLNT(:,{-40:40},:)   ; Net longwave flux at top of model
;   sw_sfc   = f1[:]->FSNS(:,{-40:40},:)   ; Net solar flux at surface
;   sw_tom   = f1[:]->FSNT(:,{-40:40},:)   ; Net solar flux at top of model
;   lw_net   = lw_sfc - lw_tom ; sfc-tom: positive: heating
;   sw_net   = sw_tom - sw_sfc ; tom-sfc: positive: heating
;   copy_VarMeta(lw_sfc,lw_net)
;   copy_VarMeta(lw_sfc,sw_net)
;   print(avg(lw_tom(:,{-5:5},:)))
;   print(avg(sw_tom(:,{-5:5},:)))
;   print(avg(lw_sfc(:,{-5:5},:)))
;   print(avg(sw_sfc(:,{-5:5},:)))
;   print(avg(lw_net(:,{-5:5},:)))
;   print(avg(sw_net(:,{-5:5},:)))

  msf800 = fbindirread("/mnt/data1/mind-envr_code/NCL/aqua_case_10yrs/MSE/projecting/mjo_regr_projecting/hadv_meridional_200.dat", 0, (/2920,84,288/),"float") ; pay attention to the type
  msf800 = msf800
  msf800!0 = "time"
  msf800&time = time
  msf800!1 = "lat"
  msf800&lat = lat
  msf800&lat@units = "degree_north"
  msf800!2 = "lon"
  msf800&lon = lon
  msf800&lon@units = "degree_east"

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

  index    = conform(msf800, preci_std, 0)

  ; lh_reg  = index*lw_net
  ; lh_reg := dim_avg_n_Wrap(lh_reg,0)
  ; copy_VarMeta(lw_tom(0,:,:),lh_reg)
  ; printVarSummary(lh_reg)
  ; fbindirwrite("/localdisk/v050.b/export/yhuangfh/NCL/aqua_case_10yrs/MSE/projecting/reg_LW_800.dat", lh_reg)

  sh_reg  = index*msf800
  sh_reg := dim_avg_n_Wrap(sh_reg,0)
  copy_VarMeta(msf800(0,:,:),sh_reg)
  printVarSummary(sh_reg)
  fbindirwrite("/mnt/data1/mind-envr_code/NCL/aqua_case_10yrs/MSE/projecting/mjo_regr_projecting/reg_hadv_meridional_200.dat", sh_reg)
