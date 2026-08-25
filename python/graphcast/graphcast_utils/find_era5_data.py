import re
from pathlib import Path
from datetime import datetime

def find_era5_data(target_date: str, parent_dir: str, filename: str = "era5_data.nc") -> Path:
    """
    Locate the ERA5 data file whose containing folder (named YYYY-MM_YYYY-MM)
    spans the given target date.

    Parameters
    ----------
    target_date : str
        Date in 'YYYY-MM-DD' format (time component, if present, is ignored).
    parent_dir : str
        Directory containing the YYYY-MM_YYYY-MM subfolders.
    filename : str
        Name of the file to locate within the matching folder (default: 'era5_data.nc').

    Returns
    -------
    Path
        Full path to the matching data file.

    Raises
    ------
    ValueError
        If target_date can't be parsed, no folder matches, multiple folders
        match, or the expected file doesn't exist in the matched folder.
    """
    # Parse just the date portion (drop any 'T...' time component if present)
    date_str = target_date.split("T")[0]
    try:
        target = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid date '{target_date}': {e}")

    target_ym = target.year * 100 + target.month  # e.g. 202010

    pattern = re.compile(r"^(\d{4})-(\d{2})_(\d{4})-(\d{2})$")
    parent = Path(parent_dir)

    matches = []
    for entry in parent.iterdir():
        if not entry.is_dir():
            continue
        m = pattern.match(entry.name)
        if not m:
            continue

        start_ym = int(m.group(1)) * 100 + int(m.group(2))
        end_ym = int(m.group(3)) * 100 + int(m.group(4))

        if start_ym <= target_ym <= end_ym:
            matches.append(entry)

    if not matches:
        raise ValueError(f"No folder in {parent_dir} spans {date_str}")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple folders span {date_str}: {[str(m) for m in matches]}. "
            "Resolve the overlap before proceeding."
        )

    data_file = matches[0] / filename
    if not data_file.is_file():
        raise ValueError(f"Matched folder {matches[0]}, but {data_file} does not exist")

    return data_file