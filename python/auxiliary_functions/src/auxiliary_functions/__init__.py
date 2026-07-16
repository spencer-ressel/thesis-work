import warnings

try:
    from .plotting_utils import (
        bmh_colors,
        modified_colormap,
        tick_labeller,
        set_plot_mode,
        get_figsize,
        format_title
    )
except ImportError as e:
    warnings.warn(f"Could not import from plotting_utils: {e}")

try:
    from .xarray_utils import (
        add_cyclic_point,
        # xarray_histogram,
        # standardize_data
    )
except ImportError as e:
    warnings.warn(f"Could not import from xarray_utils: {e}")

try:
    from .time_utils import (
        datetime64_to_yyyymmdd,
        string_to_yyyymm,
        convert_time_to_ns,
        convert_ns_to_datetime,
        extract_years_months,
    )
except ImportError as e:
    warnings.warn(f"Could not import from time_utils: {e}")

try:
    from .logging_utils import (
        ElapsedFormatter,
        CombinedTimeFormatter
    )
except ImportError as e:
    warnings.warn(f"Could not import from logging_utils: {e}")

try:
    from .calculate_velocity_potential import (
        calculate_velocity_potential_era5,
        calculate_velocity_potential_graphcast
    )
except ImportError as e:
    warnings.warn(f"Could not import from calculate_velocity_potential (likely missing 'windspharm'): {e}")