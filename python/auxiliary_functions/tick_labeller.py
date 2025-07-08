def tick_labeller(ticks, direction, degree_symbol=True, precision=1):
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