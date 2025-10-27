from cartopy.util import add_cyclic_point
import xarray as xr

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
