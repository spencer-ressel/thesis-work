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