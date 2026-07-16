import xarray as xr
from windspharm.xarray import VectorWind

def _velocity_potential_from_winds(zonal_wind, meridional_wind):
    """
    Core velocity potential computation from unbatched zonal/meridional
    wind DataArrays.

    This is the shared implementation used by both
    `calculate_velocity_potential_era5` and
    `calculate_velocity_potential_graphcast`. It exists so the tricky part
    -- the windspharm call and the latitude-ordering fix -- lives in
    exactly one place, rather than being duplicated (and potentially
    drifting out of sync) across multiple public-facing functions.

    Parameters
    ----------
    zonal_wind, meridional_wind : xarray.DataArray
        Zonal and meridional wind components, with 'lat'/'lon' dims and
        NO 'batch' dimension (that must be handled by the caller before
        this function is called).

    Returns
    -------
    velocity_potential : xarray.DataArray
        Velocity potential, same shape/coords as `zonal_wind`, with latitude
        restored to match zonal_wind's original ordering (see Notes).

    Notes
    -----
    `windspharm.xarray.VectorWind` always returns fields with latitude
    ordered north-to-south internally, *regardless* of the input's
    original orientation -- it does not flip the output back to match
    whatever ordering was passed in. Since we assume zonal_wind/meridional_wind come in
    south-to-north here, we manually re-reverse `velocity_potential_values` before writing
    it into the `zonal_wind`-shaped template.
    """

    _, velocity_potential_values = VectorWind(zonal_wind, meridional_wind).sfvp()

    velocity_potential = xr.zeros_like(zonal_wind)
    velocity_potential[:] = velocity_potential_values.isel(lat=slice(None, None, -1)).values
    velocity_potential.name = "Velocity Potential"
    velocity_potential.attrs['units'] = r"m$^{2}$ s$^{-1}$"

    return velocity_potential

def calculate_velocity_potential_era5(data, **selectors):
    """
    Compute velocity potential for unbatched data (e.g. ERA5 reanalysis),
    which has no leading 'batch' dimension.

    Parameters
    ----------
    data : xarray.Dataset
        Dataset containing 'u_component_of_wind' and 'v_component_of_wind'
        DataArrays with 'lat'/'lon' spatial dims and no 'batch' dimension.
    **selectors : dict, optional
        Coordinate-based subsetting applied to `data` *before* the wind
        components are extracted, passed straight through to
        `xarray.Dataset.sel`. Keys are dimension names; values are either
        a scalar (selects a single point, drops that dimension) or a
        slice (selects a range, keeps the dimension). e.g.:

            calculate_velocity_potential_era5(data, level=200)
            calculate_velocity_potential_era5(data, level=slice(200, 1000))

    Returns
    -------
    velocity_potential : xarray.DataArray
        Velocity potential, matching the shape/coords of the (possibly
        subset) 'u_component_of_wind' DataArray.
    """
    if selectors:
        data = data.sel(**selectors)

    zonal_wind = data['u_component_of_wind']
    meridional_wind = data['v_component_of_wind']

    return _velocity_potential_from_winds(zonal_wind, meridional_wind)

def calculate_velocity_potential_graphcast(data, batch=0, **selectors):
    """
    Compute velocity potential for batched forecast data (e.g. GraphCast,
    NeuralGCM), which has a leading 'batch' dimension that windspharm
    can't accept directly.

    Parameters
    ----------
    data : xarray.Dataset
        Dataset containing 'u_component_of_wind' and 'v_component_of_wind'
        DataArrays with a 'batch' dimension plus 'lat'/'lon' spatial dims.
    batch : int, default 0
        Integer position along the 'batch' dimension to compute velocity
        potential for. Only a single batch member is supported per call;
        loop over this function (or extend it) if you need multiple.
    **selectors : dict, optional
        Coordinate-based subsetting applied to `data` *before* the wind
        components are extracted, passed straight through to
        `xarray.Dataset.sel`. Same semantics as in
        `calculate_velocity_potential_era5`. Note: `batch` is handled
        separately (via `.isel`, since it's a position, not a coordinate
        label) and should not be passed inside `**selectors`.

    Returns
    -------
    velocity_potential : xarray.DataArray
        Velocity potential, matching the shape/coords of the (possibly
        subset) 'u_component_of_wind' DataArray, including its 'batch'
        dimension (populated only at the requested `batch` index; other
        batch members, if any, are left as zeros).
    """
    if selectors:
        data = data.sel(**selectors)

    zonal_wind = data['u_component_of_wind']
    meridional_wind = data['v_component_of_wind']

    zonal_wind_batch = zonal_wind.isel(batch=batch, drop=True)
    meridional_wind_batch = meridional_wind.isel(batch=batch, drop=True)

    velocity_potential_batch = _velocity_potential_from_winds(zonal_wind_batch, meridional_wind_batch)

    # Rebuild a full-shaped output (including the 'batch' dim) matching
    # the original zonal_wind DataArray, and slot the computed single-batch result
    # back in at the requested index.
    velocity_potential = xr.zeros_like(zonal_wind)
    velocity_potential[batch] = velocity_potential_batch.values
    velocity_potential.name = "Velocity Potential"
    velocity_potential.attrs['units'] = r"m$^{2}$ s$^{-1}$"

    return velocity_potential