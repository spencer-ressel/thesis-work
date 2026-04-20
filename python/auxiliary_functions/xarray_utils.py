from cartopy.util import add_cyclic_point
import xarray as xr
import numpy as np

def add_cyclic_xarray(dataarray, dim='longitude'):
    """
    Add a cyclic (wrap-around) point to an xarray.DataArray along a given dimension.

    Parameters
    ----------
    dataarray : xarray.DataArray
        The input DataArray. Must include the coordinate specified by `dim`.
        Typically has longitude as one of the dimensions.
    dim : str, optional
        The name of the dimension along which to add the cyclic point.
        Defaults to 'longitude'.

    Returns
    -------
    xarray.DataArray
        A new DataArray with an extra grid point appended to close the cyclic boundary.
        The output preserves the original coordinates, dimensions, attributes,
        and name of the input DataArray.
    """

    # Add cyclic point and get updated coordinate
    data, new_coord = add_cyclic_point(dataarray.values, coord=dataarray[dim].values)

    # Copy coordinates and update the chosen one
    new_coords = dict(dataarray.coords)
    new_coords[dim] = new_coord

    # Rebuild the DataArray with updated longitude
    return xr.DataArray(
        data,
        dims=dataarray.dims,
        coords=new_coords,
        attrs=dataarray.attrs,
        name=dataarray.name
    )

def xarray_histogram(da, dims, bins=50, range=None):
    """
    Compute a histogram over multiple xarray dimensions using np.histogram.
    
    Parameters
    ----------
    da : xr.DataArray
        Input data.
    dims : list or tuple of str
        Dimensions to reduce (histogram over).
    bins : int or array-like
        Number of bins or explicit bin edges.
    range : tuple, optional
        Range passed to np.histogram.
    
    Returns
    -------
    xr.Dataset with:
        - counts: histogram counts with remaining dims + 'bin'
        - bin_edges: 1D array of bin edges
    """
    # Stack the reduction dims into a single axis
    stacked = da.stack(stacked_dim=dims)

    # Apply histogram along the stacked dimension
    def _hist(x):
        counts, edges = np.histogram(x, bins=bins, range=range)
        return counts

    # Apply along the last axis
    counts = np.apply_along_axis(_hist, -1, stacked.values)

    # Remaining dims (e.g., "experiment")
    remaining_dims = [d for d in stacked.dims if d != "stacked_dim"]
    remaining_coords = {d: stacked.coords[d] for d in remaining_dims}

    # Build DataArray for counts
    counts_da = xr.DataArray(
        counts,
        dims=remaining_dims + ["bin"],
        coords={**remaining_coords, "bin": np.arange(bins)},
        name="counts"
    )

    # Compute bin edges once
    _, edges = np.histogram(da.values.ravel(), bins=bins, range=range)
    edges_da = xr.DataArray(edges, dims=["bin_edge"], name="bin_edges")

    return xr.Dataset({"counts": counts_da, "bin_edges": edges_da})

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