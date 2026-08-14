import xarray as xr
from windspharm.xarray import VectorWind
import numpy as np

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

"""NumPy port of the JAX VelocityPotential implementation.

Same math, same grid/spectral conventions, same edge-case handling
(finite differences at the poles, safe cos(lat) guard, etc.) as the
JAX version -- just without jax/xarray_jax dependencies, so it can be
run directly on plain numpy arrays.
"""

def _clenshaw_curtis_weights(nlat: int) -> np.ndarray:
    """Quadrature weights for nlat equally-spaced colatitudes, poles included."""
    n = nlat - 1
    j = np.arange(nlat)
    theta = j * np.pi / n
    w = np.zeros(nlat)
    for jj in range(nlat):
        s = 0.0
        for k in range(1, n // 2 + 1):
            ck = 1.0 if k == n / 2 else 2.0
            s += ck / (4 * k**2 - 1) * np.cos(2 * k * theta[jj])
        w[jj] = (2.0 / n) * (1 - s)
    return w

def _normalized_legendre_all(lat: np.ndarray, nmax: int) -> np.ndarray:
    """4pi-fully-normalized associated Legendre functions Pbar[n, m, :] for
    0 <= m <= n <= nmax, via a numerically stable upward recurrence (no raw
    factorials, so this does not overflow at high degree the way
    scipy.special.lpmv does)."""
    theta = np.deg2rad(90 - lat)
    Pbar = np.zeros((nmax + 1, nmax + 1, np.cos(theta).shape[0]))
    Pbar[0, 0, :] = 1.0 / np.sqrt(4 * np.pi)
    for m in range(1, nmax + 1):
        Pbar[m, m, :] = np.sqrt((2 * m + 1) / (2 * m)) * np.sin(theta) * Pbar[m - 1, m - 1, :]
    for m in range(0, nmax + 1):
        if m + 1 <= nmax:
            Pbar[m + 1, m, :] = np.sqrt(2 * m + 3) * np.cos(theta) * Pbar[m, m, :]
        for n in range(m + 2, nmax + 1):
            a_c = np.sqrt(((2 * n - 1) * (2 * n + 1)) / ((n - m) * (n + m)))
            b_c = np.sqrt(((2 * n + 1) * (n + m - 1) * (n - m - 1)) / ((n - m) * (n + m) * (2 * n - 3)))
            Pbar[n, m, :] = a_c * np.cos(theta) * Pbar[n - 1, m, :] - b_c * Pbar[n - 2, m, :]
    return Pbar

class VelocityPotentialNumpy:
    """Precomputes grid-dependent constants once; call instances like a
    function inside your pipeline. Numpy analogue of the JAX
    VelocityPotential class -- same interface, same numerics.

    Inputs/outputs are plain numpy arrays (or anything np.asarray can
    coerce, e.g. xarray.DataArray.values-backed objects). There is no
    xarray_jax.unwrap step here since there's nothing to unwrap; if you
    pass in an xarray.DataArray directly, wrap with `.values` beforehand
    or rely on np.asarray's default coercion.
    """

    def __init__(self, nlat: int, nlon: int, radius: float = 6.371e6,
                 truncation: int | None = None, dtype=np.float64):
        self.nlat, self.nlon, self.EARTH_RADIUS = nlat, nlon, radius
        nmax = mmax = truncation if truncation is not None else min(nlat - 1, 42)
        self.mmax = mmax

        lat = np.linspace(-90, 90, nlat)
        coslat_safe = np.where(np.abs(np.cos(np.deg2rad(lat))) < 1e-6, 1e-6, np.cos(np.deg2rad(lat)))  # guard the poles

        w_theta = _clenshaw_curtis_weights(nlat)
        Pbar = _normalized_legendre_all(lat, nmax)

        forward_scale = np.full(mmax + 1, 4 * np.pi / nlon)
        forward_scale[0] = 2 * np.pi / nlon
        inverse_scale = np.full(mmax + 1, nlon / 2.0)
        inverse_scale[0] = nlon
        inverse_laplacian = np.zeros(nmax + 1)
        inverse_laplacian[1:] = -radius**2 / (np.arange(1, nmax + 1) * (np.arange(1, nmax + 1) + 1))

        self.Pbar = Pbar.astype(dtype)
        self.w_theta = w_theta.astype(dtype)
        self.forward_scale = forward_scale.astype(dtype)
        self.inverse_scale = inverse_scale.astype(dtype)
        self.inverse_laplacian = inverse_laplacian.astype(dtype)
        self.coslat_safe = coslat_safe.astype(dtype)
        self.lat_rad = np.deg2rad(lat).astype(dtype)
        self.dlambda = 2 * np.pi / nlon

    def divergence(self, zonal_wind: np.ndarray, meridional_wind: np.ndarray) -> np.ndarray:
        """Spherical divergence of (u, v). Leading batch dims are fine;
        lat/lon must be the last two axes."""

        zonal_wind = np.asarray(zonal_wind)
        meridional_wind = np.asarray(meridional_wind)

        zonal_wind_zonal_gradient = (
            np.roll(zonal_wind, -1, axis=-1) - np.roll(zonal_wind, 1, axis=-1)
        ) / (2 * self.dlambda)

        latitude_weighted_meridional_wind = meridional_wind * self.coslat_safe[:, None]
        dlat = self.lat_rad[1] - self.lat_rad[0]
        meridional_wind_meridional_gradient = (
            np.roll(latitude_weighted_meridional_wind, -1, axis=-2)
            - np.roll(latitude_weighted_meridional_wind, 1, axis=-2)
        ) / (2 * dlat)

        meridional_wind_meridional_gradient[..., 0, :] = (
            (latitude_weighted_meridional_wind[..., 1, :] - latitude_weighted_meridional_wind[..., 0, :]) / dlat
        )
        meridional_wind_meridional_gradient[..., -1, :] = (
            (latitude_weighted_meridional_wind[..., -1, :] - latitude_weighted_meridional_wind[..., -2, :]) / dlat
        )

        divergence = (
            (1.0 / (self.EARTH_RADIUS * self.coslat_safe[:, None]))
            * (zonal_wind_zonal_gradient + meridional_wind_meridional_gradient)
        )

        # North pole
        divergence[..., 0, :] = -(meridional_wind[..., 1, :] - meridional_wind[..., 0, :]) / (
            self.EARTH_RADIUS * (self.lat_rad[1] - self.lat_rad[0])
        )

        # South pole
        divergence[..., -1, :] = -(meridional_wind[..., -1, :] - meridional_wind[..., -2, :]) / (
            self.EARTH_RADIUS * (self.lat_rad[-1] - self.lat_rad[-2])
        )

        return divergence

    def __call__(self, zonal_wind: np.ndarray, meridional_wind: np.ndarray) -> np.ndarray:
        """Velocity potential chi such that the divergent wind satisfies
        u_chi = (1/(a cos(lat))) d(chi)/d(lon), v_chi = (1/a) d(chi)/d(lat).
        u, v: (..., nlat, nlon). Returns chi with the same shape."""

        zonal_wind = np.asarray(zonal_wind)
        meridional_wind = np.asarray(meridional_wind)

        divergence = self.divergence(zonal_wind, meridional_wind)
        divergence_zonally_spectral = np.fft.rfft(divergence, axis=-1)[..., :self.mmax + 1]
        divergence_spectral = np.einsum(
            'nmt,t,...tm->...nm', self.Pbar, self.w_theta, divergence_zonally_spectral
        ) * self.forward_scale
        velocity_potential_spectral = self.inverse_laplacian[:, None] * divergence_spectral
        velocity_potential_zonally_spectral = np.einsum('nmt,...nm->...tm', self.Pbar, velocity_potential_spectral)
        padding_shape = velocity_potential_zonally_spectral.shape[:-1] + (self.nlon // 2 + 1 - (self.mmax + 1),)
        padded_spectrum = np.concatenate(
            [
                velocity_potential_zonally_spectral * self.inverse_scale,
                np.zeros(padding_shape, dtype=velocity_potential_zonally_spectral.dtype),
            ],
            axis=-1,
        )
        return np.fft.irfft(padded_spectrum, n=self.nlon, axis=-1)