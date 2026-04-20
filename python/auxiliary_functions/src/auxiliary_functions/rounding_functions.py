def round_out(value, order, reference=0, step=1.0):
    """
    Round a number away from a reference value to the nearest fractional step.

    Parameters:
    value (float): The number to be rounded.
    order (int): The order of magnitude for rounding (e.g., 0 = 1s, 1 = 0.1s).
    reference (float): The reference point to round away from.
    step (float): The fractional step within each rounding unit (default = 1.0). 
                  Use 0.5 for half steps, 0.25 for quarter steps, etc.

    Returns:
    float: The rounded number.
    """
    import math

    factor = 10 ** order
    full_step = step / factor  # e.g., 0.5 at order=0, 0.05 at order=1

    # Normalize value to step scale
    scaled = value / full_step
    ref_scaled = reference / full_step

    if value > reference:
        rounded = math.ceil(scaled)
    elif value < reference:
        rounded = math.floor(scaled)
    else:
        return value

    return rounded * full_step
