from config import *

def multi_experiment_lat_lon(
    variable_data,
    fig_title,
    savefig=False,
    filename=None,
    cmap='variable',
    norm='variable',
    fontsize=24,
    reference_point=None
):
    """
    Plot multi-experiment latitude-longitude data for a given variable.

    This function generates latitude-longitude contour plots for a variable across multiple experiments
    (e.g., '-4K', '0K', '4K') and displays or saves the resulting figure. The colormap and normalization
    can be customized or set based on the variable's plotting attributes.

    Parameters:
        variable_data (xarray.DataArray):
            The data to be plotted. Must include 'lat', 'lon', and 'experiment' coordinates,
            as well as units in `attrs`.
        fig_title (str):
            The title of the figure to display above the plot.
        savefig (bool, optional):
            Whether to save the figure to a file. Defaults to False.
        filename (str, optional):
            Filename for saving the figure. Required if `savefig=True`. Defaults to None.
        cmap (str, optional):
            Colormap for the plots. Options are:
              - 'variable' (default): Uses the colormap defined in `plotting_attributes` for the variable.
              - 'variable_d': Uses a divergent colormap defined in `plotting_attributes` for the variable.
              - Any other valid colormap string (e.g., 'viridis').
        norm (str, optional):
            Normalization for the colormap. Options are:
              - 'variable' (default): Uses the normalization defined in `plotting_attributes` for the variable.
              - 'zero-centered': Centered around zero using `CenteredNorm`.
              - Any other normalization object (e.g., `Normalize`).
        fontsize (int, optional):
            Default fontsize for text

    Raises:
        ValueError: If `savefig` is True but `filename` is not provided.

    Notes:
        - The function uses a default style with a font size of 24.
        - The data is cyclically extended in longitude for continuous plotting.
        - The colorbar is shared across all experiments.
        - The function assumes that the variable's name is available as `variable_data.name`.

    Dependencies:
        - numpy
        - xarray
        - matplotlib
        - cartopy
        - Auxiliary functions: `bmh_colors` and `tick_labeller` (external imports)

    Output:
        - Displays the plot if `savefig=False`.
        - Saves the plot as a PNG file in the specified directory if `savefig=True`.
    """

    if filename is None and savefig:
        raise ValueError("Filename must be given if savefig is True")

    import numpy as np
    import xarray as xr
    xr.set_options(keep_attrs=True)
    import copy
    import matplotlib.pyplot as plt
    from cartopy import crs as ccrs
    from cartopy import feature as cf
    from cartopy import util as cutil
    from matplotlib import colors as mcolors
    from matplotlib import ticker as mticker
    from matplotlib.gridspec import GridSpec

    sys.path.insert(0, "/glade/u/home/sressel/auxiliary_functions/")
    from bmh_colors import bmh_colors
    from tick_labeller import tick_labeller

    if not variable_data.name in plotting_attributes.keys():
        plotting_cmap = 'viridis'
        plotting_norm = mcolors.Normalize()

    else:
        if cmap == 'variable':
                plotting_cmap = plotting_attributes[variable_data.name].get("cmap")
        elif cmap == 'variable_d':
                plotting_cmap = plotting_attributes[variable_data.name].get("d_cmap")
        else:
            plotting_cmap = 'viridis'

        if norm == 'variable':
                plotting_norm = plotting_attributes[variable_data.name].get("norm")
        elif norm == 'zero-centered':
            print("Using zero-centered norm")
            plotting_norm = mcolors.CenteredNorm(vcenter=0)
        else:
            plotting_norm = mcolors.Normalize()

    plotting_kwargs = {
        k: copy.deepcopy(v) for k, v in {
            "norm": plotting_norm,
            "cmap": plotting_cmap
        }.items() if v is not None}

    # print(plotting_kwargs["norm"].vcenter)
    print(variable_data.name)
    grand_max = variable_data.max()
    grand_min = variable_data.min()

    plt.style.use("default")
    plt.rcParams.update({"font.size": fontsize})

    # fig = plt.figure(figsize=(12, 12))
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, width_ratios=[30,1], height_ratios=[1,1,1], figure=fig)
    gs.update(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.3, wspace=0.1)

    axes = []
    axes.append(fig.add_subplot(gs[0,0]))
    axes.append(fig.add_subplot(gs[1,0]))
    axes.append(fig.add_subplot(gs[2,0]))
    cb_ax = fig.add_subplot(gs[:, 1])

    for ax, experiment in zip(axes, experiments_list):
        ax.set_title(f"{experiments_array.sel(experiment=experiment)['name'].item()}")

        # Add cyclic point
        cdata, clon = cutil.add_cyclic_point(
            variable_data.sel(experiment=experiment),
            coord=variable_data.lon,
        )

        # Plot data
        im = ax.contourf(
            clon,
            variable_data.lat,
            cdata,
            levels=np.linspace(grand_min, grand_max, 21),
            # norm=mcolors.CenteredNorm(vcenter=0),
            # cmap='coolwarm',
            **plotting_kwargs
        )

        # Add colorbar
        cbar = fig.colorbar(
            im,
            cax=cb_ax,
            label=variable_data.attrs['units'],
            orientation="vertical",
        )
        # if (grand_min < 0) * (grand_max > 0):
        #     tick_levels = np.delete(im.levels[::2], np.argmin(np.abs(im.levels[::2])))
        #     cbar.set_ticks(np.append(tick_levels, 0))
        # else:
        cbar.set_ticks(im.levels[::2])
        cbar.ax.tick_params(labelsize=20)

        if reference_point is not None:
            ax.plot(reference_point[0], reference_point[1].sel(experiment=experiment), marker='x', ms=14, color='k')

        # Axis parameters
        ax.set_aspect("equal")

        ax.set_xlim(0, 360)
        x_ticks = np.arange(0, 360 + 60, 60)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(tick_labeller(x_ticks, "lon"))
        ax.xaxis.set_major_locator(mticker.MaxNLocator(7, prune="lower"))
        ax.set_xlabel("Longitude")

        ax.set_ylim(-30, 30)
        y_ticks = np.arange(-30, 45, 15)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(tick_labeller(y_ticks, "lat"))
        ax.set_ylabel("Latitude")

        grid_kwargs = {"linewidth": 1, "linestyle": (0, (5, 10)), "color": "gray"}
        ax.grid(True, **grid_kwargs)

    fig.suptitle(
        fig_title,
        x=(axes[0].get_position().x0 + axes[0].get_position().x1)/2,
        ha='center',
        y=1.025
    )

    if not savefig:
        plt.show()
    else:
        plt.savefig(
            f"{filename}",
            dpi=500,
            bbox_inches="tight",
        )

def multi_experiment_lon_height(
    variable_data,
    fig_title,
    savefig=False,
    filename=None,
    cmap='variable',
    norm='variable',
    fontsize=24,
    meridional_mean_region = slice(-10,10)
):

    if filename is None and savefig:
        raise ValueError("Filename must be given if savefig is True")

    # Imports
    import numpy as np
    import xarray as xr
    xr.set_options(keep_attrs=True)
    import copy
    import matplotlib.pyplot as plt
    from cartopy import crs as ccrs
    from cartopy import feature as cf
    from cartopy import util as cutil
    from matplotlib import colors as mcolors
    from matplotlib import ticker as mticker
    from matplotlib.gridspec import GridSpec

    sys.path.insert(0, "/glade/u/home/sressel/auxiliary_functions/")
    from bmh_colors import bmh_colors
    from tick_labeller import tick_labeller
    from rounding_functions import round_out


    if cmap == 'variable':
        plotting_cmap = plotting_attributes[variable_data.name].get("cmap")
    elif cmap == 'variable_d':
        plotting_cmap = plotting_attributes[variable_data.name].get("d_cmap")
    else:
        plotting_cmap = 'viridis'

    if norm == 'variable':
        plotting_norm = plotting_attributes[variable_data.name].get("norm")
    elif norm == 'zero-centered':
        plotting_norm = mcolors.CenteredNorm(vcenter=0)
    else:
        plotting_norm = mcolors.Normalize()

    plotting_kwargs = {
        k: copy.deepcopy(v) for k, v in {
            "norm": plotting_norm,
            "cmap": plotting_cmap
        }.items() if v is not None}

    # meridional_mean_region = slice(-10,10)
    meridional_mean_variable_data = variable_data.sel(
        lat=meridional_mean_region
    ).mean(dim='lat')

    grand_min = meridional_mean_variable_data.min()
    grand_max = meridional_mean_variable_data.max()
    levels = np.linspace(grand_min, grand_max, 21)

    plt.style.use("default")
    plt.rcParams.update({"font.size": fontsize})

    fig = plt.figure(figsize=(16, 20))
    gs = GridSpec(3, 2, width_ratios=[30, 1], height_ratios=[1, 1, 1], figure=fig)
    gs.update(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.3, wspace=0.1)

    axes = []
    axes.append(fig.add_subplot(gs[0,0]))
    axes.append(fig.add_subplot(gs[1,0]))
    axes.append(fig.add_subplot(gs[2,0]))
    cb_ax = fig.add_subplot(gs[:, 1])

    for ax, experiment in zip(axes, experiments_list):

        ax.set_title(f"{experiments_array.sel(experiment=experiment)['name'].item()}")

        # Add cyclic point
        cdata, clon = cutil.add_cyclic_point(
            meridional_mean_variable_data.sel(experiment=experiment).transpose("plev", "lon"),
            coord=meridional_mean_variable_data.lon,
        )

        # Plot data
        im = ax.contourf(
            clon,
            meridional_mean_variable_data.plev,
            cdata,
            levels=levels,
            **plotting_kwargs
        )

        # Add colorbar
        cbar = fig.colorbar(
            im,
            cax=cb_ax,
            label=meridional_mean_variable_data.attrs['units'],
            orientation="vertical",
        )
        cbar.ax.tick_params(labelsize=20)

        ax.contour(
            clon,
            meridional_mean_variable_data.plev,
            cdata,
            levels=levels[np.abs(levels) >= 0.9*np.diff(levels)[0]],
            colors='gray',
            linewidths=1
        )

        # ax.axvline(x=180, ls=':', color='darkgray')

        # Axis parameters
        ax.set_xlim(0, 360)
        x_ticks = np.arange(0, 360 + 60, 60)
        ax.set_xticks(x_ticks)
        # ax.set_xticklabels(tick_labeller(x_ticks, "lon"))
        # ax.xaxis.set_major_locator(mticker.MaxNLocator(7, prune="lower"))
        ax.set_xlabel("Longitude")

        ax.set_ylim(100, 950)
        ax.set_yticks(np.arange(100, 1000, 100))
        ax.set_ylabel("Pressure (hPa)")
        ax.invert_yaxis()

        ax.set_aspect('auto')

    fig.suptitle(
        fig_title,
        x=(axes[0].get_position().x0 + axes[0].get_position().x1)/2,
        ha='center',
        y=1.025
    )

    if not savefig:
        plt.show()
    else:
        plt.savefig(
            f"{filename}.png",
            dpi=500,
            bbox_inches="tight",
        )