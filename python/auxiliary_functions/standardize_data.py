def standardize_data(data, dim="time", axis=-1, unit_variance=True):
    """
    Standardizes the input data by removing the mean along a specified dimension or axis,
    and optionally scaling to unit variance.

    Parameters
    ----------
    data : xarray.DataArray, xarray.Dataset, or numpy.ndarray
        The data to standardize. Can be an xarray object or a NumPy array.
    dim : str, optional
        The dimension along which to compute the mean and standard deviation for
        xarray objects. Default is "time".
    axis : int, optional
        The axis along which to compute the mean and standard deviation for NumPy arrays.
        Default is -1 (the last axis).
    unit_variance : bool, optional
        If True, scales the data to have unit variance after removing the mean.
        If False, only removes the mean. Default is True.

    Returns
    -------
    standardized_data : xarray.DataArray, xarray.Dataset, or numpy.ndarray
        The standardized data, with the same type as the input.

    Raises
    ------
    TypeError
        If the input data is not an instance of xarray.DataArray, xarray.Dataset,
        or numpy.ndarray.
    """

    import numpy as np
    import xarray as xr

    if isinstance(data, xr.DataArray) or isinstance(data, xr.Dataset):
        if unit_variance:
            standardized_data = (data - data.mean(dim=dim)) / data.std(dim=dim)
        else:
            standardized_data = data - data.mean(dim=dim)


    elif isinstance(data, np.ndarray):
        if unit_variance:
            standardized_data = (data - np.mean(data, axis=axis))/np.std(data, axis=axis)
        else:
            standardized_data = data - np.mean(data, axis=axis)

    else:
        raise TypeError("Input data must instance of xr.DataArray, xr.Dataset, or np.ndarray")

    return standardized_data