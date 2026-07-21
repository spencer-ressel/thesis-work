from cartopy import util
import builtins
import xarray as xr
import numpy as np

def add_cyclic_point(dataarray, dim='longitude'):
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
    data, new_coord = util.add_cyclic_point(dataarray.values, coord=dataarray[dim].values)

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

def histogram(da, dims, bins=50, range=None):
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


class _StatsMixin:
    """
    Shared statistics logic for both DataArray and Dataset `.stats` accessors.
    Works unchanged on either type since .mean()/.std()/arithmetic broadcasting
    behave the same way on both.
    """

    def standardize(self, dim=None, zero_mean=True, unit_variance=False):
        """
        Standardize data along a given dimension.

        Parameters
        ----------
        dim : str, sequence of str, or None
            Dimension(s) over which to compute mean/std.
            If None (default), reduces over all dimensions.
        zero_mean : bool
            If True, subtract the mean.
        unit_variance : bool
            If True, divide by standard deviation; otherwise only remove mean.
        """
        data = self._obj

        if zero_mean:
            # Compute mean along the requested dimension
            mean = data.mean(dim=dim)
        else:
            mean = 0

        if unit_variance:
            # Compute std only if needed
            std = data.std(dim=dim)
        else:
            std = 1

        return (data - mean) / std

    def absmax(self, dim=None):
        """
        Compute the maximum absolute value along one or more dimensions.

        Parameters
        ----------
        dim : str, sequence of str, or None
            Dimension(s) over which to compute the maximum. If None,
            reduces over all dimensions.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            The maximum absolute value, with the reduced dimensions removed.
        """
        data = self._obj

        return np.abs(data).max(dim=dim)

    def magnitude(self, dim=None):
        """
        Compute the Euclidean magnitude over one or more dimensions.

        Parameters
        ----------
        dim : str, sequence of str, or None
            Dimension(s) over which to compute the magnitude. If None,
            reduces over all dimensions.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            The square root of the sum of squared values, with the reduced
            dimensions removed.
        """
        data = self._obj

        return np.sqrt((data ** 2).sum(dim=dim))

@xr.register_dataset_accessor("stats")
class StatsAccessorDataset(_StatsMixin):
    """
    Statistical operations for xarray.Dataset objects.
    Provides standardization for all data variables.
    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj


# Register a DataArray accessor named `.stats`
# After this module is imported, any DataArray will have:  da.stats.standardize()
@xr.register_dataarray_accessor("stats")
class StatsAccessor(_StatsMixin):
    def __init__(self, xarray_obj):
        # Store the underlying DataArray
        self._obj = xarray_obj

    def histogram(self, dims, bins=50, range=None):
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

        da = self._obj

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

# Register a DataArray accessor named `.stats`
# After this module is imported, any DataArray will have:  da.stats.standardize()
@xr.register_dataarray_accessor("units")
class UnitsAccessor:
    def __init__(self, xarray_obj):
        # Store the underlying DataArray
        self._obj = xarray_obj

    def to_energy_units(self):
        """
        Specifically converts Precipitation from units of mm day^-1 to units of W m^-2
        """

        data = self._obj

        if data.name != 'Precipitation':
            raise KeyError(f"Variable must be 'Precipitation', current variable is '{data.name}'")
        
        if data.attrs['units'] != 'mm day$^{-1}$':
            raise TypeError(rf"Units must be 'mm day$^{{-1}}$', current units are '{data.attrs['units']}'")
        data = (2.26*10**6/86400)*data
        data.attrs['units'] = 'W m$^{-2}$'
        return data


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


class _PrintMixin:
    """Shared value-printing behavior for DataArray and Dataset accessors."""

    def __call__(self, format=None, label=True):
        """
        Print the underlying values with optional numeric formatting and label.

        Parameters
        ----------
        format : str or None, optional
            Python format specification applied to each value, as used by
            f-strings. For example, ``"10.2f"`` or ``"0.3e"``.
        label : bool or str, optional
            If True, print the DataArray's xarray name before its values. A
            string can be supplied to label an unnamed DataArray explicitly.
            Defaults to False. Dataset variables are labeled by name in either
            case.
        """
        formatter = None
        if format is not None:
            formatter = {
                "all": lambda value: builtins.format(value, format)
            }

        def format_values(values):
            return np.array2string(
                values,
                formatter=formatter
            )

        data = self._obj

        if isinstance(data, xr.DataArray):
            values = format_values(data.values)
            if label:
                label_text = data.name if label is True else label
                print(f"{label_text}: {values}")
            else:
                print(values)
        else:
            for name, variable in data.data_vars.items():
                print(f"{name}:\n{format_values(variable.values)}")


@xr.register_dataarray_accessor("print")
class PrintAccessor(_PrintMixin):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj


@xr.register_dataset_accessor("print")
class PrintAccessorDataset(_PrintMixin):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj


class _CoordFuncsMixin:
    """
    Shared coordinate utilities for both DataArray and Dataset `.coord_funcs`
    accessors. Works unchanged on either type.
    """

    def lon_to_360(self, dim='lon'):

        data = self._obj

        # Shift negative longitudes by +360
        new_lon = xr.where(data[dim] < 0, data[dim] + 360, data[dim])

        # Assign new longitude coordinate
        new_data = data.assign_coords({dim: new_lon})

        # Sort so longitudes go 0 → 360
        new_data = new_data.sortby(dim)

        return new_data

    def add_cyclic_point(self, dim='lon'):
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
        data = self._obj

        new_data, new_coord = util.add_cyclic_point(data.values, coord=data[dim].values, axis=data.get_axis_num(dim))

        # Copy coordinates and update the chosen one
        new_coords = dict(data.coords)
        new_coords[dim] = new_coord

        # Rebuild the DataArray with updated longitude
        return xr.DataArray(
            new_data,
            dims=data.dims,
            coords=new_coords,
            attrs=data.attrs,
            name=data.name
        )

    def sel(self, *args, **kwargs):
        """
        Safe selector that only applies selections for coordinates/dimensions
        that exist on the object. For example:

            da.coord_funcs.sel(level=850)

        will return `da.sel(level=850)` if `level` is a coordinate or dimension,
        otherwise it returns the original object unchanged.

        This accepts the same basic calling pattern as `xarray`'s `.sel`:
        either a single positional `indexers` dict, and/or keyword indexers,
        plus the usual keyword options like `method`, `tolerance`, and `drop`.
        """
        obj = self._obj

        # xarray allows .sel(indexers_dict, method=..., tolerance=..., drop=...)
        # Collect an indexers dict from the first positional arg (if any)
        indexer = {}
        special = {"method", "tolerance", "drop", "fill_value"}
        forward_kwargs = {}

        if args:
            try:
                indexer = dict(args[0])
            except Exception:
                indexer = {}

        # Separate indexer keys from special kwargs
        for k, v in kwargs.items():
            if k in special:
                forward_kwargs[k] = v
            else:
                indexer[k] = v

        # Keep only indexers that exist on the object
        valid_indexer = {k: v for k, v in indexer.items() if k in obj.dims or k in obj.coords}

        if not valid_indexer:
            return obj

        return obj.sel(valid_indexer, **forward_kwargs)


@xr.register_dataarray_accessor("coord_funcs")
class CoordsAccessor(_CoordFuncsMixin):
    def __init__(self, xarray_obj):
        # Store the underlying DataArray
        self._obj = xarray_obj


@xr.register_dataset_accessor("coord_funcs")
class CoordsAccessorDataset(_CoordFuncsMixin):
    """
    Coordinate operations for xarray.Dataset objects.
    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj


def lon_to_360(data, dim="lon"):
    """
    Convert longitude coordinates from [-180, 180] to [0, 360].

    Parameters
    ----------
    da : xr.DataArray or xr.Dataset
        Input object with longitude in [-180, 180].
    lon_name : str
        Name of the longitude coordinate.

    Returns
    -------
    xr.DataArray or xr.Dataset
        Same object with longitude in [0, 360], sorted ascending.
    """

    # Shift negative longitudes by +360
    new_lon = xr.where(data[dim] < 0, data[dim] + 360, data[dim])

    # Assign new longitude coordinate
    new_data = data.assign_coords({dim: new_lon})

    # Sort so longitudes go 0 → 360
    new_data = new_data.sortby(dim)

    return new_data