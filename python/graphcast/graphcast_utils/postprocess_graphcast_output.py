def postprocess_graphcast_output(ds):
    return ds.assign_coords({'time':ds.datetime.isel(batch=0, drop=True)}).isel(batch=0, drop=True).drop_vars(["number", "expver", "datetime"])