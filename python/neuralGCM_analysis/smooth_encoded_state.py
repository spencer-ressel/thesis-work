import dataclasses
def smooth_encoded_state(state, n: int = 42):
    """
    Zeroes out (sets to 0) all vertical levels >= m for each 3D variable
    in the ModelState's inner `state` object, except for `tracers` and `sim_time`.

    For tracer fields, it explicitly applies the same vertical cutoff
    to selected tracers ('specific_cloud_liquid_water_content',
    'specific_cloud_ice_water_content', 'specific_humidity').

    The function returns a *new* ModelState (immutably updated),
    leaving the input `initial_state` unchanged.
    """


    # Extract the inner State object (contains variables like vorticity, divergence, etc.)
    new_state = state.state

    # Loop over each field in the dataclass (e.g., vorticity, divergence, tracers, sim_time, ...)
    for field in dataclasses.fields(state.state):
        name = field.name

        # For all normal state variables except tracers and sim_time:
        if name != 'tracers' and name != 'sim_time':
            # Get the variable (a JAX array)
            smoothed_value = getattr(state.state, name)

            # Set all vertical levels >= m to zero (immutably)
            smoothed_value = smoothed_value.at[..., n:].set(0)

            # Update this field in the new_state dataclass
            new_state = new_state.replace(**{name: smoothed_value})

        # Special handling for tracer variables (a dict of arrays)
        if name == 'tracers':
            smoothed_value = {}

            # Only apply to selected tracers
            for tracer in [
                'specific_cloud_liquid_water_content',
                'specific_cloud_ice_water_content',
                'specific_humidity',
            ]:
                # Get the tracer array
                smoothed_value[tracer] = state.state.tracers[tracer]

                # Zero out levels >= m
                smoothed_value[tracer] = smoothed_value[tracer].at[..., n:].set(0)

            # Replace the entire tracers dict in the new_state
            new_state = new_state.replace(tracers=smoothed_value)

    smoothed_state = state.replace(state=new_state)

    # Restore sim_time from the original state
    smoothed_state = smoothed_state.replace(
        state=smoothed_state.state.replace(sim_time=state.state.sim_time)
    )

    return smoothed_state
