def bmh_colors(color_name):
    """
    Returns hex color codes corresponding to specific colors in the 'bmh' style sheet.

    The colors can be accessed by their name (e.g., 'blue', 'red') or by their numeric
    position in the 'bmh' style sheet (e.g., 1, 2). This provides a convenient way to
    retrieve consistent color schemes for visualizations.

    More information about the 'bmh' style can be found here:
    https://viscid-hub.github.io/Viscid-docs/docs/dev/styles/bmh.html

    Parameters
    ----------
    color_name : str or int
        The name or number of the color to retrieve. If a string is provided, it should
        match one of the predefined color names (e.g., 'blue', 'red'). If an integer is
        provided, it should correspond to the order of the colors as specified in the
        'bmh' style sheet.

    Returns
    -------
    str
        The hexadecimal code of the specified color.
    """

    colors = {}

    # Specify colors by name
    colors['blue'] = '#348ABD'
    colors['red'] = '#A60628'
    colors['purple'] = '#7A68A6'
    colors['green'] = '#467821'
    colors['orange'] = '#D55E00'
    colors['pink'] = '#CC79A7'
    colors['lightblue'] = '#56B4E9'
    colors['lightgreen'] = '#009E73'
    colors['yellow'] = '#F0E442'
    colors['darkblue'] = '#0072B2'
    colors['edgecolor'] = '#bcbcbc'
    colors['facecolor'] = '#eeeeee'

    # Specify colors by number
    colors[1] = '#348ABD'
    colors[2] = '#A60628'
    colors[3] = '#7A68A6'
    colors[4] = '#467821'
    colors[5] = '#D55E00'
    colors[6] = '#CC79A7'
    colors[7] = '#56B4E9'
    colors[8] = '#009E73'
    colors[9] = '#F0E442'
    colors[10] = '#0072B2'
    colors[11] = '#bcbcbc'
    colors[12] = '#eeeeee'

    if color_name not in colors:
        raise KeyError(
            f"Specific color unsupported."
            + f" Please choose from one of the following colors/indices: {[key for key in colors.keys()]}"
        )

    return colors[color_name]

def modified_colormap(colormap, central_color, central_width, blend_strength):
    '''
    This function modifies a colormap to set the central region to be white.
    Within the region specified by the 'width' parameter, the colormap is blended towards white using a linspace.

    Parameters:
        colormap (str): The name of an existing matplotlib colormap
        central_color (str): The name of an existing matplotlib color
        central_width (float): The width of the region to be set to white
        blend_strength (float): The width of the regions to be blended to white

    Returns:
        modified_colormap (matplotlib.colors.LinearSegmentedColormap): The modified colormap

    '''
    # Import libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors

    try:
        c = mcolors.cnames[central_color]
    except:
        raise KeyError('Not a matplotlib named color')

    central_color = list(mcolors.to_rgba(central_color))

    # Raise an error if the width is not between 0 and 1
    if ((central_width < 0)+(central_width > 1)):
        raise ValueError('Central width must be in range [0, 1]')
    elif ((blend_strength < 0) + (blend_strength > 1)):
        raise ValueError('Blend strength must be in range [0, 1]')

    # Convert the widths to the range [0, 127]
    else:
        central_width = int(127*central_width)
        blend_strength = int(blend_strength*(127-central_width))

    # Get the colormap values
    original_colormap = plt.cm.get_cmap(colormap)
    newcolors = original_colormap(np.linspace(0, 1, 256))

    # Get the value of the colormap 'width' values left of the center, and blend from that value to white at the center
    newcolors[128-central_width-blend_strength:128-central_width, :] = np.linspace(
        newcolors[128-central_width-blend_strength, :],
        central_color,
        blend_strength
    )

    newcolors[128-central_width:128+central_width, :] = central_color

    # Get the value of the colormap 'width' values right of the center, and blend from white at the center to that value
    newcolors[128+central_width:128+central_width+blend_strength, :] = np.linspace(
        central_color,
        newcolors[128+central_width+blend_strength, :],
        blend_strength
    )

    # Create a new colormap object from the modified map
    modified_colormap = mcolors.LinearSegmentedColormap.from_list(colormap+'_modified', newcolors)

    return modified_colormap

def tick_labeller(ticks, direction, degree_symbol=True, precision=0):
    """
    This function takes in a numpy array of tick locations and formats the list as latitude or longitude points.

    # Parameters
    ticks (numpy.ndarray) : An array containing the locations of the ticks
    direction (str)       : Either 'lat' or 'lon', specifying which coordinate the ticks represent
    degree_symbol (bool)  : Default = True, determines whether the tick strings contain the symbol '°'. Should be set to 'False' for
                            directory names, file names, etc.

    # Returns
    labels (list)         : A list of the text string labels of each tick location specified in 'ticks'
    """

    import numpy as np
    labels = []
    for i in range(len(ticks)):
        if direction == 'lon':
            normalized_tick = ticks[i] % 360
            if normalized_tick == 0 or normalized_tick == 180:
                labels.append(f"{normalized_tick:.{precision}f}{('°' if degree_symbol else '')}")
            elif 0 < normalized_tick < 180:
                labels.append(f"{normalized_tick:.{precision}f}{('°' if degree_symbol else '')}E")
            elif -180 < ticks[i] < 0:
                labels.append(f"{-ticks[i]:.{precision}f}{('°' if degree_symbol else '')}W")
            elif 180 < normalized_tick < 360:
                labels.append(f"{360 - normalized_tick:.{precision}f}{('°' if degree_symbol else '')}W")
            elif -360 < ticks[i] < -180:
                labels.append(f"{360 + ticks[i]:.{precision}f}{('°' if degree_symbol else '')}E")

        elif direction=='lat':
            if ticks[i] == 0:
                labels.append(f"{np.abs(ticks[i]):.{precision}f}{('°' if degree_symbol else '')}")
            elif ticks[i] < 0:
                labels.append(f"{np.abs(ticks[i]):.{precision}f}{('°' if degree_symbol else '')}S")
            elif ticks[i] > 0:
                labels.append(f"{np.abs(ticks[i]):.{precision}f}{('°' if degree_symbol else '')}N")

    return labels

import matplotlib.pyplot as plt

def set_plot_mode(mode):
    """
    Set global matplotlib rcParams for a given plotting mode.

    Parameters
    ----------
    mode : {"publication", "slides"}
        Plotting style preset.

        "publication"
            Small fonts and thin lines for journal figures.

        "slides"
            Larger fonts and thicker lines for presentations.

    Notes
    -----
    This function updates matplotlib.rcParams globally.
    Call this once near the start of a script or notebook.
    """

    if mode == "publication":
        plt.rcParams.update({
            "figure.figsize": (3.5, 3),
            "font.size": 8,
            # "axes.titlesize": 9,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "lines.linewidth": 1.0,
            "axes.titlepad": 4.0
        })

    elif mode == "slides":
        plt.rcParams.update({
            "figure.figsize": (8, 6),
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "lines.linewidth": 2.5,
            "axes.titlepad": 6
        })

    else:
        raise ValueError(
            f"Unknown mode '{mode}'. "
            "Valid options: 'publication', 'slides'."
        )


def get_figsize(mode, layout):
    """
    Return a figure size for a given plotting mode and layout.

    Parameters
    ----------
    mode : {"publication", "slides"}
        Plotting style preset.

    layout : {"onecol", "twocol", "full", "square"}
        Figure layout preset.

        onecol
            Single-column journal figure.

        twocol
            Two-column journal figure.

        full
            Full-width figure.

        square
            Square figure.

    Returns
    -------
    tuple of float
        (width, height) in inches.

    Notes
    -----
    Intended to be used with:

        fig, ax = plt.subplots(
            figsize=get_figsize(mode, layout)
        )
    """

    FIG_SIZES = {
        "publication": {
            "onecol": (3.2, 2.6),
            "twocol": (5.5, 3.5),
            "full": (5.5, 5.5),
            "square": (3.2, 3.2),
        },
        "slides": {
            "onecol": (6, 4),
            "twocol": (10, 5),
            "full": (10, 7),
            "square": (6, 6),
        },
    }

    try:
        return FIG_SIZES[mode][layout]
    except KeyError:
        raise ValueError(
            f"Invalid mode/layout: mode='{mode}', layout='{layout}'. "
            "Valid modes: 'publication', 'slides'. "
            "Valid layouts: 'onecol', 'twocol', 'full', 'square'."
        )

def format_title(text, mode, layout):
    """
    Format a title string with automatic line wrapping based on plot mode
    and figure layout.

    Parameters
    ----------
    text : str
        Title text to format.

    mode : {"publication", "slides"}
        Plotting mode controlling font size and overall scaling.

        publication
            Small fonts, narrow figures.

        slides
            Larger fonts, wider figures.

    layout : {"onecol", "twocol", "full", "square"}
        Figure layout preset.

        onecol
            Single-column figure.

        twocol
            Two-column figure.

        full
            Full-width figure.

        square
            Square figure.

    Returns
    -------
    str
        Wrapped title string with newline characters inserted.

    Notes
    -----
    Intended for use with matplotlib:

        ax.set_title(format_title(title, mode, layout))

    Wrapping width is chosen based on both mode and layout so that
    titles scale correctly for publication vs slides.
    """

    import textwrap

    WRAP_WIDTHS = {
        "publication": {
            "onecol": 20,
            "twocol": 30,
            "full": 30,
            "square": 20,
        },
        "slides": {
            # "onecol": 40,
            # "twocol": 60,
            # "full": 60,
            # "square": 40,
            "onecol": 20,
            "twocol": 30,
            "full": 30,
            "square": 20,
        },
    }

    try:
        width = WRAP_WIDTHS[mode][layout]
    except KeyError:
        raise ValueError(
            f"Invalid mode/layout: mode='{mode}', layout='{layout}'. "
            "Valid modes: 'publication', 'slides'. "
            "Valid layouts: 'onecol', 'twocol', 'full', 'square'."
        )

    return textwrap.fill(text, width)