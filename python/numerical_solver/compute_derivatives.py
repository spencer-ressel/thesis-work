import numpy as np 

def compute_zonal_derivative_fourier(zonal_wavenumber, field_variable):

    if (field_variable.ndim != 2):
        raise ValueError("Input fields have wrong dimensions, must be (y, x)")

    # Convert fields into fourier space
    field_variable_fourier = np.fft.fft(field_variable, axis=1)

    # Compute first derivatives
    field_variable_zonal_derivative_fourier = 1j*zonal_wavenumber[None,:]*field_variable_fourier
     
    # Transform back to physical space
    field_variable_zonal_derivative = np.real(np.fft.ifft(field_variable_zonal_derivative_fourier, axis=1))

    # Return fields
    return field_variable_zonal_derivative
    

def compute_meridional_derivative_fourier(meridional_wavenumber, field_variable):

    if (field_variable.ndim != 2):
        raise ValueError("Input fields have wrong dimensions, must be (y, x)")

    # Convert fields into fourier space
    field_variable_fourier = np.fft.fft(field_variable, axis=0)

    # Compute first derivatives
    field_variable_meridional_derivative_fourier = 1j*meridional_wavenumber[:, None]*field_variable_fourier
     
    # Transform back to physical space
    field_variable_meridional_derivative = np.real(np.fft.ifft(field_variable_meridional_derivative_fourier, axis=0))

    # Return fields
    return field_variable_meridional_derivative

def compute_meridional_derivative_reflected(meridional_gridpoints, field_variable, sign=1):

    if (field_variable.ndim != 2):
        raise ValueError("Input fields have wrong dimensions, must be (y, x)")

    # sign should be positive for meridional velocity, and negative for zonal velocity and temperature
    # Reflect fields in y to ensure periodicity
    field_variable_reflected = np.concatenate(
        (
            field_variable,
            sign*field_variable[::-1],
        ),
        axis=0
    )
    
    # Compute the wavenumbers of the reflected domain
    ny = len(meridional_gridpoints)
    meridional_step_size = np.diff(meridional_gridpoints)[0]
    meridional_wavenumber_reflected = 2*np.pi*np.fft.fftfreq(2*ny, meridional_step_size)

    # Convert the fields into Fourier space
    field_variable_fourier = np.fft.fft(field_variable_reflected, axis=0)
    
    # Compute meridional derivatives in Fourier space
    field_variable_meridional_derivative_fourier = (
        1j*meridional_wavenumber_reflected[:,None] * field_variable_fourier
    )
    
    # Convert back into physical space
    field_variable_meridional_derivative = np.real(
        np.fft.ifft(field_variable_meridional_derivative_fourier, axis=0)
    )[:ny]
    
    # Return fields
    return field_variable_meridional_derivative


def compute_meridional_derivative_finite_difference(meridional_gridpoints, field_variable):
    
    
    field_variable_meridional_derivative = np.zeros_like(field_variable)
    ny = len(meridional_gridpoints)
    meridional_grid_spacing = np.diff(meridional_gridpoints)[0]
    for y_index in range(1, ny-1):
        field_variable_meridional_derivative[y_index] = (field_variable[y_index+1] - field_variable[y_index-1])/(2*meridional_grid_spacing)

    field_variable_meridional_derivative[0] = (field_variable[1] - field_variable[0])/(meridional_grid_spacing)
    field_variable_meridional_derivative[-1] = (field_variable[-1] - field_variable[-2])/(meridional_grid_spacing)
    # field_variable_meridional_derivative[0, :] = (-3*field_variable[0] + 4*field_variable[1] - field_variable[2])/(2*meridional_grid_spacing)
    # field_variable_meridional_derivative[-1, :] = (3*field_variable[-1] - 4*field_variable[-2] + field_variable[-3])/(2*meridional_grid_spacing)

    # for y_index in range(2, ny-2):
    #     field_variable_meridional_derivative[y_index] = 
    #     (
    #         (4/3)*(
    #             field_variable[y_index+1]
    #             - field_variable[y_index-1]
    #         )/(2*meridional_grid_spacing)
    #     - (1/3)*(
    #             field_variable[y_index+2]
    #             - field_variable[y_index-2]
    #         )/(4*meridional_grid_spacing)
    #     )
        
    # field_variable_meridional_derivative[0, :] = 0#(
    # #     -3*field_variable[0] 
    # #     + 4*field_variable[1] 
    # #     - field_variable[2]
    # # ) / (2*meridional_grid_spacing)
    # field_variable_meridional_derivative[1, :] = (
    #     -3*field_variable[1] 
    #     + 4*field_variable[2] 
    #     - field_variable[3]
    # ) / (2*meridional_grid_spacing)
    # field_variable_meridional_derivative[-2, :] = (
    #     3*field_variable[-2] 
    #     - 4*field_variable[-3] 
    #     + field_variable[-4]
    # ) / (2*meridional_grid_spacing)
    # field_variable_meridional_derivative[-1, :] = 0#(
    #     3*field_variable[-1] 
    #     - 4*field_variable[-2] 
    #     + field_variable[-3]
    # ) / (2*meridional_grid_spacing)

    # for y_index in range(0,ny-2):
        # field_variable_meridional_derivative[y_index] = (-3*field_variable[y_index] + 4*field_variable[y_index+1] - field_variable[y_index+2])/(2*meridional_grid_spacing)

    # field_variable_meridional_derivative[-1, :] = (3*field_variable[-1] - 4*field_variable[-2] + field_variable[-3])/(2*meridional_grid_spacing)
    # field_variable_meridional_derivative[-2, :] = (3*field_variable[-2] - 4*field_variable[-3] + field_variable[-4])/(2*meridional_grid_spacing)
    
    return field_variable_meridional_derivative
        
    