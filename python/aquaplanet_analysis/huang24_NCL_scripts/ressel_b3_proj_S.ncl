;==================================================
; projecting anomalies in the budget quantities onto 
; the MSE anomaly and its time derivative
; 2023-04-13
; by HUANG Yuanyuan
; @HKUST
;==================================================

;; integral area:
  ; latS    = -5.
  ; latN    =  5.
  ; lonL    = 175.
  ; lonR    = 185.

  latN     = 40
  latS     = -latN   ; symmetric about the equator

  path_v    = "/localdisk/v050.b/export/yhuangfh/NCL/aqua_case_10yrs/MSE/projecting/mjo_regr_projecting/" 
  path_mse  = "/localdisk/v050.b/export/yhuangfh/NCL/aqua_case_10yrs/MSE/" 

  diri1    = "/localdisk/v050.b/export/yhuangfh/aqua_case_200/" 
  fils1    = systemfunc("ls " + diri1 + "*.nc ")
  f1       = addfiles(fils1, "r")
  hgt      = f1[:]->Z3(:,:,{latS:latN},:)
  lat      = hgt&lat
  lon      = hgt&lon
  delete(f1)
  delete(fils1)
  delete(diri1)

; <h>
  hint800 = fbindirread(path_mse + "reg_mse_int_800.dat", 0, (/84,288/),"float")
  copy_VarMeta(hgt(0,0,:,:),hint800)
  hint800 := hint800({-10:10},:)

  hint500 = fbindirread(path_mse + "reg_mse_int_500.dat", 0, (/84,288/),"float")
  copy_VarMeta(hgt(0,0,:,:),hint500)
  hint500 := hint500({-10:10},:)

  hint200 = fbindirread(path_mse + "reg_mse_int_200.dat", 0, (/84,288/),"float")
  copy_VarMeta(hgt(0,0,:,:),hint200)
  hint200 := hint200({-10:10},:)

; <dh/dt>
  dhdtint800 = fbindirread(path_mse + "reg_dhdt_int_800.dat", 0, (/84,288/),"double")
  dhdtint800 = dhdtint800/3600./24.
  copy_VarMeta(hgt(0,0,:,:),dhdtint800)
  dhdtint800 := dble2flt(dhdtint800({-10:10},:))

  dhdtint500 = fbindirread(path_mse + "reg_dhdt_int_500.dat", 0, (/84,288/),"double") 
  dhdtint500 = dhdtint500/3600./24.
  copy_VarMeta(hgt(0,0,:,:),dhdtint500)
  dhdtint500 := dble2flt(dhdtint500({-10:10},:))

  dhdtint200 = fbindirread(path_mse + "reg_dhdt_int_200.dat", 0, (/84,288/),"double")  
  dhdtint200 = dhdtint200/3600./24.
  copy_VarMeta(hgt(0,0,:,:),dhdtint200)
  dhdtint200 := dble2flt(dhdtint200({-10:10},:))
  print(dhdtint800)

  ; lw800 = fbindirread(path_v + "reg_LW_800.dat", 0, (/84,288/),"float")
  ; copy_VarMeta(hgt(0,0,:,:),lw800)
  ; lw800 := lw800({-10:10},:)  
  ; lw500 = fbindirread(path_v + "reg_LW_500.dat", 0, (/84,288/),"float")
  ; copy_VarMeta(hgt(0,0,:,:),lw500)
  ; lw500 := lw500({-10:10},:)
  ; lw200 = fbindirread(path_v + "reg_LW_200.dat", 0, (/84,288/),"float")
  ; copy_VarMeta(hgt(0,0,:,:),lw200)
  ; lw200 := lw200({-10:10},:)
  ; print(lw800)

  sw800 = fbindirread(path_v + "reg_SW_800.dat", 0, (/84,288/),"float")
  copy_VarMeta(hgt(0,0,:,:),sw800)
  sw800 := sw800({-10:10},:)  
  sw500 = fbindirread(path_v + "reg_SW_500.dat", 0, (/84,288/),"float")
  copy_VarMeta(hgt(0,0,:,:),sw500)
  sw500 := sw500({-10:10},:)
  sw200 = fbindirread(path_v + "reg_SW_200.dat", 0, (/84,288/),"float")
  copy_VarMeta(hgt(0,0,:,:),sw200)
  sw200 := sw200({-10:10},:)

  ; lh800 = fbindirread(path_v + "reg_LH_800.dat", 0, (/84,288/),"float")
  ; copy_VarMeta(hgt(0,0,:,:),lh800)
  ; lh800 := lh800({-20:20},:)  
  ; lh500 = fbindirread(path_v + "reg_LH_500.dat", 0, (/84,288/),"float")
  ; copy_VarMeta(hgt(0,0,:,:),lh500)
  ; lh500 := lh500({-20:20},:)
  ; lh200 = fbindirread(path_v + "reg_LH_200.dat", 0, (/84,288/),"float")
  ; copy_VarMeta(hgt(0,0,:,:),lh200)
  ; lh200 := lh200({-20:20},:)

  ; sh800 = fbindirread(path_v + "reg_SH_800.dat", 0, (/84,288/),"float")
  ; copy_VarMeta(hgt(0,0,:,:),sh800)
  ; sh800 := sh800({-10:10},:)  
  ; sh500 = fbindirread(path_v + "reg_SH_500.dat", 0, (/84,288/),"float")
  ; copy_VarMeta(hgt(0,0,:,:),sh500)
  ; sh500 := sh500({-10:10},:)
  ; sh200 = fbindirread(path_v + "reg_SH_200.dat", 0, (/84,288/),"float") 
  ; copy_VarMeta(hgt(0,0,:,:),sh200)
  ; sh200 := sh200({-10:10},:)

  ; advh800 = fbindirread(path_v + "reg_hadv_h_800.dat", 0, (/84,288/),"float")
  ; copy_VarMeta(hgt(0,0,:,:),advh800)
  ; advh800 := -advh800({-10:10},:) 
  ; advh500 = fbindirread(path_v + "reg_hadv_h_500.dat", 0, (/84,288/),"float")
  ; copy_VarMeta(hgt(0,0,:,:),advh500)
  ; advh500 := -advh500({-10:10},:)
  ; advh200 = fbindirread(path_v + "reg_hadv_h_200.dat", 0, (/84,288/),"float")
  ; copy_VarMeta(hgt(0,0,:,:),advh200)
  ; advh200 := -advh200({-10:10},:) 

  ; advv800 = fbindirread(path_v + "reg_hadv_v_800.dat", 0, (/84,288/),"double")
  ; copy_VarMeta(hgt(0,0,:,:),advv800)
  ; advv800 := -advv800({-10:10},:)   
  ; advv500 = fbindirread(path_v + "reg_hadv_v_500.dat", 0, (/84,288/),"double")
  ; copy_VarMeta(hgt(0,0,:,:),advv500)
  ; advv500 := -advv500({-10:10},:)   
  ; advv200 = fbindirread(path_v + "reg_hadv_v_200.dat", 0, (/84,288/),"double")
  ; copy_VarMeta(hgt(0,0,:,:),advv200)
  ; advv200 := -advv200({-10:10},:) 

;;Fractional contribution to MSE_ano maintenance
  xhint_800 = sw800*hint800
  hint2_800 = hint800*hint800
  upper_800 = sum(xhint_800) ;spatial integral without weight
  lower_800 = sum(hint2_800)
  m_800 = upper_800/lower_800*24.*3600.
  print(m_800)

  xhint_500 = sw500*hint500
  hint2_500 = hint500*hint500
  upper_500 = sum(xhint_500) ;spatial integral without weight
  lower_500 = sum(hint2_500)
  m_500 = upper_500/lower_500*24.*3600.
  print(m_500)

  xhint_200 = sw200*hint200
  hint2_200 = hint200*hint200
  upper_200 = sum(xhint_200) ;spatial integral without weight
  lower_200 = sum(hint2_200)
  m_200 = upper_200/lower_200*24.*3600.
  print(m_200)


;;Fractional contribution to MSE_ano propagation

  xdhdtint_800 = sw800*dhdtint800
  dhdtint2_800 = dhdtint800*dhdtint800
  upper1_800  = dim_sum_n_Wrap(dim_sum_n_Wrap(xdhdtint_800,0),0) ;spatial integral
  lower1_800 = dim_sum_n_Wrap(dim_sum_n_Wrap(dhdtint2_800,0),0)
  p_800 = upper1_800/lower1_800
  print(p_800)

  xdhdtint_500 = sw500*dhdtint500
  dhdtint2_500 = dhdtint500*dhdtint500
  upper1_500  = dim_sum_n_Wrap(dim_sum_n_Wrap(xdhdtint_500,0),0) ;spatial integral
  lower1_500 = dim_sum_n_Wrap(dim_sum_n_Wrap(dhdtint2_500,0),0)
  p_500 = upper1_500/lower1_500
  print(p_500)

  xdhdtint_200 = sw200*dhdtint200
  dhdtint2_200 = dhdtint200*dhdtint200
  upper1_200  = dim_sum_n_Wrap(dim_sum_n_Wrap(xdhdtint_200,0),0) ;spatial integral
  lower1_200 = dim_sum_n_Wrap(dim_sum_n_Wrap(dhdtint2_200,0),0)
  p_200 = upper1_200/lower1_200
  print(p_200)