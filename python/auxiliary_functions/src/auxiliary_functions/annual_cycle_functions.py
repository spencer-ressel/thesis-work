import xarray as xr
import numpy as np
import pandas as pd


def _decimal_year(times):
    """
    Convert datetime64-like values to decimal years (e.g. 2004-07-01 ->
    2004.497...), normalizing by each timestamp's own calendar year length
    (365 or 366 days) so leap years don't accumulate phase drift.
    """
    times = pd.DatetimeIndex(times)
    year_start = pd.DatetimeIndex(times.year.astype(str) + '-01-01')
    year_end = pd.DatetimeIndex((times.year + 1).astype(str) + '-01-01')

    seconds_into_year = (times - year_start).total_seconds()
    seconds_in_year = (year_end - year_start).total_seconds()

    return times.year + seconds_into_year / seconds_in_year

def _compute_t(time_values, t0, account_for_leap_years):
    """
    Build the harmonic-regression time coordinate `t` and the angular
    frequency multiplier `w_base`, such that w = w_base * k gives the
    k-th harmonic's frequency.

    account_for_leap_years=True  -> decimal-year basis, period exactly 1.0
    account_for_leap_years=False -> raw day-counter basis, period 365
                                     (correct for noleap/360_day-style
                                     calendars where every year truly is
                                     365 days; will drift on real Gregorian
                                     calendar data with leap years)
    """
    if account_for_leap_years:
        dy = _decimal_year(time_values) - _decimal_year([t0])[0]
        t = xr.DataArray(dy, dims='time', coords={'time': time_values})
        w_base = 2 * np.pi  # period = 1.0 in decimal-year units
    else:
        days_since_t0 = (time_values - np.datetime64(t0)) / np.timedelta64(1, 'D')
        t = xr.DataArray(days_since_t0 + 1, dims='time', coords={'time': time_values})
        w_base = 2 * np.pi / 365  # fixed 365-day period

    return t, w_base

def _build_harmonics(t, w_base, nharmonics):
    harmonics = [xr.ones_like(t)]
    for k in range(1, nharmonics + 1):
        w = w_base * k
        harmonics.append(np.cos(w * t))
        harmonics.append(np.sin(w * t))
    harmonics_array = xr.concat(harmonics, dim="harmonic")
    harmonics_array = harmonics_array.assign_coords(harmonic=np.arange(harmonics_array.sizes["harmonic"]))
    return harmonics_array

def fit_annual_cycle(data, nharmonics=3, account_for_leap_years=True):
    """
    Fit the mean + first `nharmonics` annual harmonics to an xarray DataArray
    via least-squares (harmonic regression), for later reuse on new data.

    Parameters
    ----------
    data : xr.DataArray
        Input training data with a 'time' dimension.
    nharmonics : int
        Number of harmonics to fit (default: 3).
    account_for_leap_years : bool
        If True, use a decimal-year time basis so leap years don't
        accumulate phase drift (correct for real Gregorian-calendar data,
        e.g. ERA5, GraphCast/NeuralGCM output). If False, use a raw
        365-day-period basis (correct for calendars where every year truly
        has 365 days, e.g. CAM6 'noleap' output -- and slightly cheaper).
        Must match the value passed to apply_annual_cycle / reconstruct_annual_cycle.

    Returns
    -------
    regression_coefficients : xr.DataArray
        Fitted coefficients (dim: 'harmonic', plus any other dims of `data`).
    t0 : np.datetime64
        Reference date (first timestamp of `data`) that the harmonic phase
        is anchored to. Must be passed to apply_annual_cycle/reconstruct_annual_cycle
        for both the training data and any new data, so phase stays consistent.
    """
    t0 = data.time.values[0]
    t, w_base = _compute_t(data.time.values, t0, account_for_leap_years)
    harmonics_array = _build_harmonics(t, w_base, nharmonics)

    # factor of 2 for all cos/sin terms (harmonic index >= 1); mean term (index 0) stays x1
    scale = xr.where(harmonics_array.harmonic == 0, 1.0, 2.0)
    regression_coefficients = scale * (harmonics_array * data).mean(dim='time')
    regression_coefficients.name = data.name

    return regression_coefficients, t0

def reconstruct_annual_cycle(times, regression_coefficients, t0, nharmonics=3, account_for_leap_years=True):
    """
    Evaluate a previously-fit annual cycle at an arbitrary set of timestamps
    (does not require any actual data at those timestamps -- just dates).

    Parameters
    ----------
    times : array-like of datetime64
        Timestamps to evaluate the cycle at, e.g. from pd.date_range() or
        data.time.values.
    regression_coefficients, t0, nharmonics, account_for_leap_years
        Same as returned by / passed to fit_annual_cycle -- must match what
        was used there.

    Returns
    -------
    annual_cycle : xr.DataArray
        The reconstructed annual cycle at `times`, dim 'time' plus any
        other dims of regression_coefficients.
    """
    times = np.asarray(times, dtype='datetime64[ns]')
    t, w_base = _compute_t(times, t0, account_for_leap_years)
    harmonics_array = _build_harmonics(t, w_base, nharmonics)
    return (harmonics_array * regression_coefficients).sum("harmonic")

def apply_annual_cycle(data, regression_coefficients, t0, nharmonics=3, account_for_leap_years=True):
    """
    Reconstruct a previously-fit annual cycle at `data`'s timestamps and
    subtract it off. Works for both the original training data and new data.

    Parameters
    ----------
    data : xr.DataArray
        Data to deannualize (training or new), with a 'time' dimension.
    regression_coefficients, t0, nharmonics, account_for_leap_years
        Same as returned by / passed to fit_annual_cycle -- must match what
        was used there (mixing True/False between fit and apply will
        silently misalign phase).

    Returns
    -------
    data_deannualized : xr.DataArray
        data minus the reconstructed annual cycle.
    annual_cycle : xr.DataArray
        The reconstructed annual cycle evaluated at data's timestamps.
    """
    annual_cycle = reconstruct_annual_cycle(
        data.time.values, regression_coefficients, t0, nharmonics, account_for_leap_years
    )
    data_deannualized = data - annual_cycle
    data_deannualized.name = data.name

    return data_deannualized, annual_cycle

def remove_annual_cycle(data, nharmonics=3, account_for_leap_years=True):
    """
    Convenience one-shot wrapper: fit the annual cycle to `data` and
    immediately remove it from that same data. Equivalent to calling
    fit_annual_cycle() followed by apply_annual_cycle() on the same
    dataset -- use those directly instead if you need to reuse the fit
    on other data (e.g. train on climatology, apply to a forecast).

    Returns
    -------
    data_deannualized : xr.DataArray
    annual_cycle : xr.DataArray
    regression_coefficients : xr.DataArray
    t0 : np.datetime64
    """
    regression_coefficients, t0 = fit_annual_cycle(data, nharmonics, account_for_leap_years)
    data_deannualized, annual_cycle = apply_annual_cycle(
        data, regression_coefficients, t0, nharmonics, account_for_leap_years
    )
    return data_deannualized, annual_cycle