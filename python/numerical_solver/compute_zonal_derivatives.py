# def compute_zonal_derivatives_fourier(zonal_wavenumber, zonal_velocity, column_temperature):

#     import numpy as np
#     if (zonal_velocity.ndim != 2) or (column_temperature.ndim != 2):
#         raise ValueError("Input fields have wrong dimensions, must be (y, x)")

#     # Convert fields into fourier space
#     zonal_velocity_fourier = np.fft.fft(zonal_velocity, axis=1)
#     column_temperature_fourier = np.fft.fft(column_temperature, axis=1)

#     # Compute first derivatives
#     zonal_velocity_zonal_derivative_fourier = 1j*zonal_wavenumber[None,:]*zonal_velocity_fourier
#     column_temperature_zonal_derivative_fourier = 1j*zonal_wavenumber[None,:]*column_temperature_fourier
     
#     # Transform back to physical space
#     zonal_velocity_zonal_derivative = np.real(np.fft.ifft(zonal_velocity_zonal_derivative_fourier, axis=1))
#     column_temperature_zonal_derivative = np.real(np.fft.ifft(column_temperature_zonal_derivative_fourier, axis=1))

#     # Return fields
#     return zonal_velocity_zonal_derivative, column_temperature_zonal_derivative

def compute_zonal_derivatives_fourier(zonal_wavenumber, field_variable):

    import numpy as np
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
    