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