import glob
import xarray as xr
# Logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

first_directory = "/glade/u/home/sressel/spencer-scratch/graphcast_input_data/1992-06_1992-07"
second_directory = "/glade/u/home/sressel/spencer-scratch/graphcast_input_data/1992-08_1993-04"
output_directory = "/glade/u/home/sressel/spencer-scratch/graphcast_input_data/1992-06_1993-04"

variables_to_skip = ['land_sea_mask', 'geopotential_at_surface']
files_list = sorted(glob.glob(f"{first_directory}/*.nc"))

for file in files_list:
    variable = file.split("/")[-1].split(".nc")[0]
    logger.info(f"{variable}")

    first_data = xr.open_dataset(f"{first_directory}/{variable}.nc")
    second_data = xr.open_dataset(f"{second_directory}/{variable}.nc")

    if variable == 'era5_data':
        concatenated_data = xr.concat(
            [first_data, second_data],
            dim='time'
        )
        concatenated_data['geopotential_at_surface'] = concatenated_data['geopotential_at_surface'].isel(time=0, drop=True)
        concatenated_data['land_sea_mask'] = concatenated_data['land_sea_mask'].isel(time=0, drop=True)
        
    if variable not in variables_to_skip:
        concatenated_data = xr.concat(
            [first_data, second_data],
            dim='time'
        )
    else:
        concatenated_data = first_data

    concatenated_data.to_netcdf(f"{output_directory}/{variable}.nc")   

logger.info("Finished") 