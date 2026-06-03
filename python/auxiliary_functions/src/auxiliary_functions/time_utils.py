import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd

def datetime64_to_yyyymmdd(dt):
    """Convert a datetime64-like value to YYYYMMDD for filenames."""
    date = np.datetime64(dt).astype("datetime64[D]")
    return str(date).replace("-", "")

def string_to_yyyymm(dt):
    """
    Convert a datetime or ISO-format datetime string into 'YYYYMM'.
    """
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt) # type: ignore
    return f"{dt.year}{dt.month:02d}"
    # return datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S.%f").strftime("%Y%m")

def convert_time_to_ns(ds):
    # Original datetime64 coordinate
    datetime = ds["time"]

    # Compute nanoseconds since first timestep
    t0 = datetime.values[0]
    time_ns = (datetime.values - t0).astype("timedelta64[ns]").astype("timedelta64[ns]")

    new_datetime = datetime.expand_dims('batch').assign_coords(
        datetime=("time", datetime.values),   # secondary coordinate
        time=("time", time_ns)                # replace primary coordinate
    )

    # Assign new coordinates
    ds = ds.expand_dims('batch').assign_coords(
        datetime=new_datetime,
        time=("time", time_ns)
    )

    return ds

def convert_ns_to_datetime(ds, start_time_str):
    """
    Convert GraphCast-style time (ns since start) into human-readable datetime64,
    and attach it as a secondary coordinate called 'datetime'.

    Parameters
    ----------
    da : xr.DataArray
        Must have a 'time' coordinate of dtype int64 (nanoseconds since start).
    start_time_str : str
        Start time of the forecast, e.g. "1992-08-14 00:00:00".

    Returns
    -------
    xr.DataArray
        Same data, with a new coordinate 'datetime' containing datetime64[ns].
    """

    # Parse the start time into datetime64[ns]
    t0 = np.datetime64(pd.to_datetime(start_time_str))

    # Extract integer nanoseconds
    time_ns = ds["time"].values.astype("int64")

    # Convert to datetime64[ns]
    datetime_vals = t0 + time_ns.astype("timedelta64[ns]")

    # Attach as a secondary coordinate
    ds_new = ds.assign_coords(
        datetime=("time", datetime_vals)
    )

    return ds_new

def extract_years_months(start, end):
    """
    Given start and end datetimes (strings or datetime objects),
    return sorted lists of unique years and months needed for a CDS request.
    """
    if isinstance(start, str):
        start = datetime.fromisoformat(start) # type: ignore
    if isinstance(end, str):
        end = datetime.fromisoformat(end) # type: ignore

    # Normalize to first day of month
    cursor = start.replace(day=1)
    end_month = end.replace(day=1)

    years = set()
    months = set()

    while cursor <= end_month:
        years.add(f"{cursor.year}")
        months.add(f"{cursor.month:02d}")
        cursor += relativedelta(months=1)

    return sorted(years), sorted(months)