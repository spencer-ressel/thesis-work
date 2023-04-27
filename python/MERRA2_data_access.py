import os

# Set the URL for the MERRA-2 data
url = 'ftp://goldsmr4.sci.gsfc.nasa.gov/data/s4pa/MERRA2/M2T1NXSLV.5.12.4'

# Set the years and months to download
years = range(1990, 2018)
months = range(1, 13)

# Set the latitudes and longitudes to download
lat_start = -25
lat_end = 25
lon_start = 0
lon_end = 360

# Set the output directory to save the downloaded files
output_dir = '/path/to/output/directory'

# Loop over the years and months to download the data
for year in years:
    for month in months:
        # Set the filename for the data
        filename = f'MERRA2_{year}{month:02d}_inst1_2d_asm_Nx.nc4'
        # Set the URL for the data file
        file_url = f'{url}/{year}/{filename}'
        # Set the output path for the data file
        output_path = os.path.join(output_dir, filename)
        # Download the data file using wget
        os.system(f'wget -c -N -O {output_path} "{file_url}"')