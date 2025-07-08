from config import *

def subset_data(variable_data, time_bounds, latitude_bounds, pressure_bounds=None):
    """
    Subset data based on given bounds.

    Parameters:
    - variable_data: data to subset
    - time_bounds: slice of (start, end) time
    - latitude_bounds: slice of (min_lat, max_lat)
    - pressure_bounds: optional, tuple of (min_pressure, max_pressure)

    Returns:
    Subsetted data.
    """

    import xarray as xr

    if 'plev' in variable_data.coords:
        variable_subset = variable_data.sel(
            time=time_bounds,
            lat=latitude_bounds,
            plev=pressure_bounds
        )

    else:
        variable_subset = variable_data.sel(
            time=time_bounds,
            lat=latitude_bounds,
        )

    variable_subset.attrs = variable_data.attrs
    variable_subset.attrs['subset'] = "True"
    return variable_subset

def detrend_data(variable_data, detrend_axis=0, detrend_type='linear'):
    """
    Detrend data based on given axis and trend type.

    Parameters:
    - variable_data: data to detrend
    - detrend_axis: optional, int of axis along which to detrend
    - detrend_type: option, str of detrend type from scipy.signal library

    Returns:
    Detrended data.
    """

    import xarray as xr
    from scipy import signal

    variable_detrended = xr.zeros_like(variable_data)

    variable_detrended[:] = signal.detrend(
        (variable_data - variable_data.mean(dim="time")).values,
        axis=detrend_axis,
        type=detrend_type,
    )

    variable_detrended.attrs = variable_data.attrs
    variable_detrended.attrs["detrended"] = "True"

    return variable_detrended

def time_filter_data(variable_data, period_bounds, filter_order=4, fs=1):
    """
    Time filter data based on given period bounds.

    Parameters:
    - variable_data: data to subset
    - period_bounds: slice of (long, short) period cutoffs
    - filter_order: optional, filter order
    - fs: optional, float of sampling frequency

    Returns:
    Subsetted data.
    """

    import xarray as xr
    from scipy import signal

    # Construct a filter
    low_frequency = (1 / period_bounds.start)
    high_frequency = (1 / period_bounds.stop)
    b, a = signal.butter(filter_order, [low_frequency, high_frequency], btype="band", fs=fs)

    variable_filtered = xr.zeros_like(variable_data)
    variable_filtered[:] = signal.filtfilt(
        b, a, variable_data, axis=variable_data.get_axis_num(dim='time')
    )

    variable_filtered.name = variable_data.name
    variable_filtered.attrs = variable_data.attrs
    variable_filtered.attrs["filtered"] = "True"
    variable_filtered.attrs["frequency_bounds"] = (low_frequency, high_frequency)

    return variable_filtered

def low_pass_filter_data(variable_data, period_bound, filter_order=4, fs=1, filter_axis=0):
    """
    Time filter data based on given period bounds.

    Parameters:
    - variable_data: data to subset
    - period_bound: period upper bound
    - filter_order: optional, filter order
    - fs: optional, float of sampling frequency
    - filter_axis: optional, int of axis along which to filter

    Returns:
    Subsetted data.
    """

    import xarray as xr
    from scipy import signal

    # Construct a filter
    high_frequency = (1 / period_bound)
    b, a = signal.butter(filter_order, high_frequency, btype="lowpass", fs=fs)

    variable_filtered = xr.zeros_like(variable_data)
    variable_filtered[:] = signal.filtfilt(
        b, a, variable_data, axis=filter_axis
    )

    variable_filtered.attrs = variable_data.attrs
    variable_filtered.attrs["filtered"] = "True"
    variable_filtered.attrs["frequency_bound"] = high_frequency

    return variable_filtered

def high_pass_filter_data(variable_data, period_bound, filter_order=4, fs=1, filter_axis=0):
    """
    Time filter data based on given period bounds.

    Parameters:
    - variable_data: data to subset
    - period_bound: period lower bound
    - filter_order: optional, filter order
    - fs: optional, float of sampling frequency
    - filter_axis: optional, int of axis along which to filter

    Returns:
    Subsetted data.
    """

    import xarray as xr
    from scipy import signal

    # Construct a filter
    low_frequency = (1 / period_bound)
    b, a = signal.butter(filter_order, low_frequency, btype="highpass", fs=fs)

    variable_filtered = xr.zeros_like(variable_data)
    variable_filtered[:] = signal.filtfilt(
        b, a, variable_data, axis=filter_axis
    )

    variable_filtered.attrs = variable_data.attrs
    variable_filtered.attrs["filtered"] = "True"
    variable_filtered.attrs["frequency_bound"] = low_frequency

    return variable_filtered

def mjo_filter_data(
    variable_time_filtered,
    wavenumber_bounds
):

    import xrft
    import xarray as xr
    import numpy as np

    # Transform the data to fourier space
    variable_fourier_transform = xrft.fft(
        variable_time_filtered,
        dim=['time', 'lon'],
        true_phase=True,
        true_amplitude=True
    )

    # Calculate the phase speed
    phase_speed = (variable_fourier_transform.freq_time/variable_fourier_transform.freq_lon)

    # Filter for only eastward propagating modes
    eastward_filtered_variable_fourier = variable_fourier_transform.where(phase_speed < 0, other=0)

    # Filter for MJO zonal scales
    mjo_filtered_variable_fourier = eastward_filtered_variable_fourier.where(
        (
            (wavenumber_bounds.start/360 <= np.abs(eastward_filtered_variable_fourier.freq_lon)) &
            (np.abs(eastward_filtered_variable_fourier.freq_lon) <= wavenumber_bounds.stop/360)
        ),
        other=0
    )

    variable_mjo_filtered = xr.zeros_like(variable_time_filtered)
    variable_mjo_filtered[:] = xrft.ifft(
        mjo_filtered_variable_fourier,
        dim=['freq_time', 'freq_lon'],
        true_phase=True,
        true_amplitude=True
    ).values

    variable_mjo_filtered.name = variable_time_filtered.name
    variable_mjo_filtered.attrs = variable_time_filtered.attrs
    variable_mjo_filtered.attrs["mjo-filtered"] = "True"
    variable_mjo_filtered.attrs["wavenumber-bounds"] = (wavenumber_bounds.start, wavenumber_bounds.stop)
    return variable_mjo_filtered

def rossby_filter_data(
    variable_time_filtered,
    wavenumber_bounds
):

    import xrft
    import xarray as xr
    import numpy as np

    # Transform the data to fourier space
    variable_fourier_transform = xrft.fft(
        variable_time_filtered,
        dim=['time', 'lon'],
        true_phase=True,
        true_amplitude=True
    )

    # Calculate the phase speed
    phase_speed = (variable_fourier_transform.freq_time/variable_fourier_transform.freq_lon)

    # Filter for only eastward propagating modes
    westward_filtered_variable_fourier = variable_fourier_transform.where(phase_speed > 0, other=0)

    # Filter for MJO zonal scales
    rossby_filtered_variable_fourier = westward_filtered_variable_fourier.where(
        (
            (wavenumber_bounds.start/360 <= np.abs(westward_filtered_variable_fourier.freq_lon)) &
            (np.abs(westward_filtered_variable_fourier.freq_lon) <= wavenumber_bounds.stop/360)
        ),
        other=0
    )

    variable_rossby_filtered = xr.zeros_like(variable_time_filtered)
    variable_rossby_filtered[:] = xrft.ifft(
        rossby_filtered_variable_fourier,
        dim=['freq_time', 'freq_lon'],
        true_phase=True,
        true_amplitude=True
    ).values

    variable_rossby_filtered.name = variable_time_filtered.name
    variable_rossby_filtered.attrs = variable_time_filtered.attrs
    variable_rossby_filtered.attrs["mjo-filtered"] = "True"
    variable_rossby_filtered.attrs["wavenumber-bounds"] = (wavenumber_bounds.start, wavenumber_bounds.stop)
    return variable_rossby_filtered

def kelvin_filter_data(
    variable_data,
    equivalent_depth_bounds,
    period_bounds
):
    import xrft
    import xarray as xr
    import numpy as np

    variable_fft = xrft.fft(
        variable_data,
        dim=['time', 'lon'],
        true_amplitude=True,
        true_phase=True
    )
    zonal_wavenumber = variable_fft.freq_lon*(180/np.pi)*(1/EARTH_RADIUS)     # 1/m
    frequency = variable_fft.freq_time                                        # 1/s
    phase_speed = frequency/zonal_wavenumber                                  # m/s

    # The left-edge of the Kelvin-wave band when frequency > 0
    inner_edge_equivalent_depth = equivalent_depth_bounds.start               # m
    inner_edge_phase_speed = -np.sqrt(GRAVITY*inner_edge_equivalent_depth)    # m/s
    inner_edge_frequencies = zonal_wavenumber*inner_edge_phase_speed          # 1/s
    inner_edge_zonal_wavenumbers = frequency/inner_edge_phase_speed           # 1/m

    # The right-edge of the Kelvin-wave band when frequency > 0
    outer_edge_equivalent_depth = equivalent_depth_bounds.stop                # m
    outer_edge_phase_speed = -np.sqrt(GRAVITY*outer_edge_equivalent_depth)    # m/s
    outer_edge_frequencies = zonal_wavenumber*outer_edge_phase_speed          # 1/s
    outer_edge_zonal_wavenumbers = frequency/outer_edge_phase_speed           # 1/m

    # Mask out zonal wavenumber components outside of the Kelvin wave band
    variable_fft_masked = variable_fft.copy()

    # Remove westward propagating components
    variable_fft_masked[:] = variable_fft_masked.where(phase_speed < 0, other=0)

    # Loop over each frequency, and zero-out wavenumber components outside of the KW band
    for f1_index, f1 in enumerate(variable_fft.freq_time):
        # If the frequency is positive, the left edge is the inner edge
        if f1 > 0:
            variable_fft_masked[f1_index] = variable_fft[f1_index].where(
                (outer_edge_zonal_wavenumbers[f1_index]*np.pi*EARTH_RADIUS/180 <= variable_fft.freq_lon)
                *(variable_fft.freq_lon <= inner_edge_zonal_wavenumbers[f1_index]*np.pi*EARTH_RADIUS/180),
                other=0
            )
        # Otherwise the left edge is the outer edge
        else:
            variable_fft_masked[f1_index] = variable_fft[f1_index].where(
                (inner_edge_zonal_wavenumbers[f1_index]*np.pi*EARTH_RADIUS/180 <= variable_fft.freq_lon)
                *(variable_fft.freq_lon <= outer_edge_zonal_wavenumbers[f1_index]*np.pi*EARTH_RADIUS/180),
                other=0
            )

    kelvin_space_filtered_variable_data = variable_data.copy()
    kelvin_space_filtered_variable_data[:] = xrft.ifft(
        variable_fft_masked,
        dim=['freq_time', 'freq_lon'],
        true_phase=True,
        true_amplitude=True
    ).values

    if period_bounds.stop == 2*np.diff(variable_data.time.values)[0].days:
        print("Upper frequency bound is Nyquist frequency, using high-pass filter")
        kelvin_filtered_variable_data = high_pass_filter_data(
            kelvin_space_filtered_variable_data,
            period_bounds.start
        )
    else:
        kelvin_filtered_variable_data = time_filter_data(
            kelvin_space_filtered_variable_data,
            slice(period_bounds.start, period_bounds.stop)
        )
    return kelvin_filtered_variable_data

# def mjo_filter_data(variable_time_filtered, wavenumber_bounds):

#     import numpy as np
#     import xarray as xr

#     n_time = len(variable_time_filtered.time)
#     n_lon = len(variable_time_filtered.lon)

#     # Transform the data to fourier space
#     variable_fourier_transform = np.fft.fft2(
#         variable_time_filtered,
#         axes=[0,2]
#     )

#     frequencies = np.fft.fftfreq(n_time, 1)
#     wavenumbers = np.fft.fftfreq(n_lon, 2.5)

#     # Calculate the phase speed
#     phase_speed = np.einsum(
#         'i,j->ij',
#         frequencies,
#         1/wavenumbers
#     )

#     # Filter for only eastward propagating modes
#     eastward_filtering_mask = np.where(phase_speed < 0, 1, 0)
#     eastward_filtered_variable_fourier = variable_fourier_transform*eastward_filtering_mask[:, np.newaxis, :]

#     # Filter for MJO zonal scales
#     mjo_scale_mask = np.where(
#         (
#             (wavenumber_bounds.start/360 <= np.abs(wavenumbers)) &
#             (np.abs(wavenumbers) <= wavenumber_bounds.stop/360)
#         ),
#         1, 0
#     )
#     mjo_filtered_variable_fourier = eastward_filtered_variable_fourier*mjo_scale_mask[np.newaxis, np.newaxis, :]

#     variable_mjo_filtered = xr.zeros_like(variable_time_filtered)
#     variable_mjo_filtered[:] = np.fft.ifft2(
#         mjo_filtered_variable_fourier,
#         axes=[0,2]
#     )

#     return variable_mjo_filtered