from config import *
def dry_Kelvin_wave_dynamical_core(zonal_wavenumber, meridional_gridpoints, field_variables):

    import numpy as np
    from compute_derivatives import compute_zonal_derivative_fourier, compute_meridional_derivative_reflected, compute_meridional_derivative_finite_difference

    (zonal_velocity, column_temperature) = field_variables
    meridional_grid_spacing = np.diff(meridional_gridpoints)[0]
    # ny = len(meridional_gridpoints)
    # nx = len(zonal_wavenumber)
    
    # # Pad the temperature field to estimate the meridional gradients at the boundaries
    # extrapolated_temperature_north = column_temperature[-2, :] - (GROSS_DRY_STABILITY/gravity_wave_phase_speed**2)*(2*meridional_grid_spacing)*(CORIOLIS_PARAMETER * meridional_gridpoints[-1] * zonal_velocity[-1, :]) 
    # extrapolated_temperature_south = column_temperature[1, :] + (GROSS_DRY_STABILITY/gravity_wave_phase_speed**2)*(2*meridional_grid_spacing)*(CORIOLIS_PARAMETER * meridional_gridpoints[0] * zonal_velocity[0, :]) 

    # padded_column_temperature = np.empty((ny+2, nx))
    # padded_column_temperature[0] = extrapolated_temperature_south
    # padded_column_temperature[1:-1] = column_temperature[0]
    # padded_column_temperature[-1] = extrapolated_temperature_north

    # padded_meridional_gridpoints = np.empty((ny+2))
    # padded_meridional_gridpoints[0] = meridional_gridpoints[0] - meridional_grid_spacing
    # padded_meridional_gridpoints[-1] = meridional_gridpoints[-1] + meridional_grid_spacing
    # padded_meridional_gridpoints[1:-1] = meridional_gridpoints
    
    # dudx = compute_zonal_derivative_fourier(zonal_wavenumber, zonal_velocity)
    dTdx = compute_zonal_derivative_fourier(zonal_wavenumber, column_temperature)
    # dTdy = compute_meridional_derivative_reflected(meridional_gridpoints, column_temperature, sign=1)

    dTdy = compute_meridional_derivative_finite_difference(meridional_gridpoints, np.copy(column_temperature))
    # dTdy = compute_meridional_derivative_finite_difference(padded_meridional_gridpoints, np.copy(padded_column_temperature))[1:-1, :]
    ddTdxdy = compute_zonal_derivative_fourier(zonal_wavenumber, dTdy)

    
    column_temperature_forcing = (
        - gravity_wave_phase_speed**2 * (-1/(CORIOLIS_PARAMETER*meridional_gridpoints))[:, None] * ddTdxdy
    )
    
    return column_temperature_forcing

def dry_Matsuno_dynamical_core(zonal_wavenumber, meridional_wavenumber, meridional_gridpoints, field_variables):

    import numpy as np
    from compute_derivatives import compute_zonal_derivative_fourier, compute_meridional_derivative_reflected, compute_meridional_derivative_fourier, compute_meridional_derivative_finite_difference

    [zonal_velocity, meridional_velocity, column_temperature, column_moisture] = field_variables

    dudx = compute_zonal_derivative_fourier(zonal_wavenumber, zonal_velocity)
    # dvdy = compute_meridional_derivative_fourier(meridional_wavenumber, meridional_velocity)
    # dvdy = compute_meridional_derivative_reflected(meridional_gridpoints, meridional_velocity, sign=1)
    dvdy = compute_meridional_derivative_finite_difference(meridional_gridpoints, meridional_velocity)
    dTdx = compute_zonal_derivative_fourier(zonal_wavenumber, column_temperature)
    # dTdy = compute_meridional_derivative_fourier(meridional_wavenumber, column_temperature)
    # dTdy = compute_meridional_derivative_reflected(meridional_gridpoints, column_temperature, sign=1)
    dTdy = compute_meridional_derivative_finite_difference(meridional_gridpoints, column_temperature)
    
    ### Zonal Velocity
    zonal_velocity_forcing = (
        + CORIOLIS_PARAMETER * meridional_gridpoints[:, None] * meridional_velocity 
        - (gravity_wave_phase_speed**2 / GROSS_DRY_STABILITY) * dTdx
    )
    
    ### Meridional Velocity
    meridional_velocity_forcing = (
        - CORIOLIS_PARAMETER * meridional_gridpoints[:,None] * zonal_velocity 
        - (gravity_wave_phase_speed**2 / GROSS_DRY_STABILITY) * dTdy
    )
    
    ### Column Temperature
    column_temperature_forcing = - GROSS_DRY_STABILITY * (dudx + dvdy)
    
    ### Column Moisture
    column_moisture_forcing = np.zeros_like(column_temperature_forcing)

    return zonal_velocity_forcing, meridional_velocity_forcing, column_temperature_forcing, column_moisture_forcing

def Ahmed21_dynamical_core(zonal_wavenumber, meridional_gridpoints, field_variables, Ahmed_parameters):

    import numpy as np
    from compute_derivatives import compute_zonal_derivative_fourier, compute_meridional_derivative_reflected, compute_meridional_derivative_finite_difference

    [zonal_velocity, meridional_velocity, column_temperature, column_moisture] = field_variables
    [
        temperature_sensitivity_array, 
        moisture_sensitivity_array, 
        zonal_moistening_array, 
        meridional_moistening_array, 
        moisture_stratification_array
    ] = Ahmed_parameters

    dudx = compute_zonal_derivative_fourier(zonal_wavenumber, zonal_velocity)
    dvdy = compute_meridional_derivative_reflected(meridional_gridpoints, meridional_velocity, sign=1)
    dTdx = compute_zonal_derivative_fourier(zonal_wavenumber, column_temperature)
    dTdy = compute_meridional_derivative_reflected(meridional_gridpoints, column_temperature, sign=-1)
    
    ### Zonal Velocity
    zonal_velocity_forcing = (
        + CORIOLIS_PARAMETER * meridional_gridpoints[:, None] * meridional_velocity 
        - (gravity_wave_phase_speed**2 / GROSS_DRY_STABILITY) * dTdx
    )
    
    ### Meridional Velocity
    meridional_velocity_forcing = (
        - CORIOLIS_PARAMETER * meridional_gridpoints[:,None] * zonal_velocity 
        - (gravity_wave_phase_speed**2 / GROSS_DRY_STABILITY) * dTdy
    )
    
    ### Column Temperature
    column_temperature_forcing = (
        - GROSS_DRY_STABILITY * (dudx + dvdy)
        - temperature_sensitivity_array * (1+CLOUD_RADIATIVE_PARAMETER) * column_temperature 
        + moisture_sensitivity_array * (1+CLOUD_RADIATIVE_PARAMETER) * column_moisture 
    )
    
    ### Column Moisture
    column_moisture_forcing = (
        + zonal_moistening_array * zonal_velocity
        - meridional_moistening_array * meridional_gridpoints[:, None] * meridional_velocity 
        + moisture_stratification_array * (dudx+dvdy)
        - moisture_sensitivity_array    * column_moisture
        + temperature_sensitivity_array * column_temperature
    )

    return zonal_velocity_forcing, meridional_velocity_forcing, column_temperature_forcing, column_moisture_forcing

def Gaussian_mean_state_dynamical_core(zonal_wavenumber, meridional_gridpoints, field_variables, Gaussian_parameters):

    import numpy as np
    from compute_derivatives import compute_zonal_derivative_fourier, compute_meridional_derivative_reflected, compute_meridional_derivative_finite_difference

    [zonal_velocity, meridional_velocity, column_temperature, column_moisture] = field_variables
    [
        temperature_sensitivity_array, 
        moisture_sensitivity_array, 
        zonal_moistening_array, 
        meridional_moistening_array, 
        moisture_stratification_array,
        gaussian_length_scale
    ] = Gaussian_parameters

    dudx = compute_zonal_derivative_fourier(zonal_wavenumber, zonal_velocity)
    dvdy = compute_meridional_derivative_reflected(meridional_gridpoints, meridional_velocity, sign=1)
    dTdx = compute_zonal_derivative_fourier(zonal_wavenumber, column_temperature)
    dTdy = compute_meridional_derivative_reflected(meridional_gridpoints, column_temperature, sign=-1)
    
    ### Zonal Velocity
    zonal_velocity_forcing = (
        + CORIOLIS_PARAMETER * meridional_gridpoints[:, None] * meridional_velocity 
        - (gravity_wave_phase_speed**2 / GROSS_DRY_STABILITY) * dTdx
    )
    
    ### Meridional Velocity
    meridional_velocity_forcing = (
        - CORIOLIS_PARAMETER * meridional_gridpoints[:,None] * zonal_velocity 
        - (gravity_wave_phase_speed**2 / GROSS_DRY_STABILITY) * dTdy
    )
    
    ### Column Temperature
    column_temperature_forcing = (
        - GROSS_DRY_STABILITY * (dudx + dvdy)
        - temperature_sensitivity_array * (1+CLOUD_RADIATIVE_PARAMETER) * column_temperature 
        + moisture_sensitivity_array * (1+CLOUD_RADIATIVE_PARAMETER) * column_moisture 
    )
    
    ### Column Moisture
    column_moisture_forcing = (
        + zonal_moistening_array * zonal_velocity
        - meridional_moistening_array * (meridional_gridpoints * np.exp(-(meridional_gridpoints/gaussian_length_scale)**2))[:,None] * meridional_velocity 
        + moisture_stratification_array * (dudx+dvdy)
        - moisture_sensitivity_array    * column_moisture
        + temperature_sensitivity_array * column_temperature
    )

    return zonal_velocity_forcing, meridional_velocity_forcing, column_temperature_forcing, column_moisture_forcing
    