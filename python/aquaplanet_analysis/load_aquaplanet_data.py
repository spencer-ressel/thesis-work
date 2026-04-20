import xarray as xr
import numpy as np
import glob
import config
import coords

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

def load_aquaplanet_data(variables_to_load, experiment, pressure_subset_bounds=slice(100, 950)):

    variables_loaded = []
    variables_dict = {}

    input_data_directory = rf"/glade/campaign/univ/uwas0152/post_processed_data/{experiment}/daily_pressure-level_data"

    print('=' * config.SEP_WIDTH)
    print(f"{f'Loading {experiment} experiment data':^{config.SEP_WIDTH}}")
    print('=' * config.SEP_WIDTH)

    # PRCP
    if 'Precipitation' in variables_to_load:
        print(f"{'Precipitation':<{config.SEP_WIDTH-1}}")

        print(f"{'→ Data...':<{config.SEP_WIDTH-1}}", end="")
        convective_precipitation = xr.open_dataset(
            f"{input_data_directory}/SST_AQP3_Qobs_27_{experiment}_1D_20y_PRECC.nc"
        )["PRECC"]
        large_scale_precipitation = xr.open_dataset(
            f"{input_data_directory}/SST_AQP3_Qobs_27_{experiment}_1D_20y_PRECL.nc"
        )["PRECL"]
        precipitation = (convective_precipitation + large_scale_precipitation) * (
            1000 * config.SECONDS_PER_DAY
        )
        print(rf"{'✔':>1}")

        print(f"{'→ Attributes...':<{config.SEP_WIDTH-1}}", end="")
        precipitation.name = "Precipitation"
        precipitation.attrs["description"] = "Sum of large scale and convective precipitation"
        precipitation.attrs["units"] = r"mm day$^{-1}$"
        precipitation.attrs["file_id"] = "PRCP"
        precipitation.attrs["short_name"] = "PRCP"
        # variables_dict['Precipitation'] = precipitation
        variables_loaded.append(precipitation)
        print(rf"{'✔':>1}")
        print(f"{'-' * config.SEP_WIDTH}")

    # OLR
    if 'Outgoing Longwave Radiation' in variables_to_load:
        print(f"{'Outgoing Longwave Radiation':<{config.SEP_WIDTH-1}}")

        print(f"{'→ Data...':<{config.SEP_WIDTH-1}}", end="")
        outgoing_longwave_radiation = xr.open_dataset(
            f"{input_data_directory}/SST_AQP3_Qobs_27_{experiment}_1D_20y_FLUT.nc"
        )["FLUT"]
        print(rf"{'✔':>1}")

        print(f"{'→ Attributes...':<{config.SEP_WIDTH-1}}", end="")
        outgoing_longwave_radiation.name = "Outgoing Longwave Radiation"
        outgoing_longwave_radiation.attrs["description"] = "Outgoing longwave radiation at TOA"
        outgoing_longwave_radiation.attrs["units"] = r"W m$^{-2}$"
        outgoing_longwave_radiation.attrs["file_id"] = "OLR"
        outgoing_longwave_radiation.attrs["short_name"] = "OLR"
        # variables_dict['Outgoing Longwave Radiation'] = outgoing_longwave_radiation
        variables_loaded.append(outgoing_longwave_radiation)
        print(rf"{'✔':>1}")
        print(f"{'-'*config.SEP_WIDTH}")

    # U
    if 'Zonal Wind' in variables_to_load:
        print(f"{'Zonal Wind':<{config.SEP_WIDTH-1}}")
        print(f"{'→ Data...':<{config.SEP_WIDTH-1}}", end="")
        zonal_wind = xr.open_dataset(
            f"{input_data_directory}/SST_AQP3_Qobs_27_{experiment}_1D_20y_U.nc"
        )["U"]
        print(rf"{'✔':>1}")

        print(f"{'→ Attributes...':<{config.SEP_WIDTH-1}}", end="")
        zonal_wind.name = "Zonal Wind"
        zonal_wind.attrs["description"] = "Zonal Wind"
        zonal_wind.attrs["units"] = r"m s$^{-1}$"
        zonal_wind.attrs["file_id"] = "U"
        zonal_wind.attrs["short_name"] = "U"
        # variables_dict['Zonal Wind'] = zonal_wind
        variables_loaded.append(zonal_wind)
        print(rf"{'✔':>1}")
        print(f"{'-'*config.SEP_WIDTH}")

    # V
    if 'Meridional Wind' in variables_to_load:
        print(f"{'Meridional Wind':<{config.SEP_WIDTH-1}}")

        print(f"{'→ Data...':<{config.SEP_WIDTH-1}}", end="")
        meridional_wind = xr.open_dataset(
            f"{input_data_directory}/SST_AQP3_Qobs_27_{experiment}_1D_20y_V.nc"
        )["V"]
        print(rf"{'✔':>1}")

        print(f"{'→ Attributes...':<{config.SEP_WIDTH-1}}", end="")
        meridional_wind.name = "Meridional Wind"
        meridional_wind.attrs["description"] = "Meridional Wind"
        meridional_wind.attrs["units"] = r"m s$^{-1}$"
        meridional_wind.attrs["file_id"] = "V"
        meridional_wind.attrs["short_name"] = "V"
        # variables_dict['Meridional Wind'] = meridional_wind
        variables_loaded.append(meridional_wind)
        print(rf"{'✔':>1}")
        print(f"{'-'*config.SEP_WIDTH}")

    # OMEGA
    if 'Vertical Wind' in variables_to_load:
        print(f"{'Vertical Wind':<{config.SEP_WIDTH-1}}")

        print(f"{'→ Data...':<{config.SEP_WIDTH-1}}", end="")
        vertical_wind = xr.open_dataset(
            f"{input_data_directory}/SST_AQP3_Qobs_27_{experiment}_1D_20y_OMEGA.nc"
        )["OMEGA"]
        print(rf"{'✔':>1}")

        print(f"{'→ Attributes...':<{config.SEP_WIDTH-1}}", end="")
        vertical_wind.name = "Vertical Wind"
        vertical_wind.attrs["description"] = "Vertical Wind"
        vertical_wind.attrs["units"] = r"Pa s$^{-1}$"
        vertical_wind.attrs["file_id"] = "ω"
        vertical_wind.attrs["short_name"] = "ω"
        # variables_dict['Vertical Wind'] = vertical_wind
        variables_loaded.append(vertical_wind)
        print(rf"{'✔':>1}")
        print(f"{'-'*config.SEP_WIDTH}")

    # T
    if (
        'Temperature' in variables_to_load
        or 'Column Temperature' in variables_to_load
        or 'Moist Static Energy' in variables_to_load
        or 'Column Moist Static Energy' in variables_to_load
    ):

        print(f"{'Temperature':<{config.SEP_WIDTH-1}}")

        print(f"{'→ Data...':<{config.SEP_WIDTH-1}}", end="")
        temperature = xr.open_dataset(
            f"{input_data_directory}/SST_AQP3_Qobs_27_{experiment}_1D_20y_T.nc")['T']
        print(rf"{'✔':>1}")

        print(f"{'→ Attributes...':<{config.SEP_WIDTH-1}}", end="")
        temperature.name = "Temperature"
        temperature.attrs["description"] = "Temperature"
        temperature.attrs["units"] = r"K"
        temperature.attrs["file_id"] = "T"
        temperature.attrs["short_name"] = "T"
        # variables_dict['Temperature'] = temperature
        variables_loaded.append(temperature)

        print(rf"{'✔':>1}")
        print(f"{'-'*config.SEP_WIDTH}")

    # CWV
    if (
        'Moisture' in variables_to_load
        or 'Column Water Vapor' in variables_to_load
        or 'Moist Static Energy' in variables_to_load
        or 'Column Moist Static Energy' in variables_to_load
    ):
        print(f"{'Moisture':<{config.SEP_WIDTH-1}}")

        print(f"{'→ Data...':<{config.SEP_WIDTH-1}}", end="")
        moisture = xr.open_dataset(
            f"{input_data_directory}/SST_AQP3_Qobs_27_{experiment}_1D_20y_Q.nc")['Q']
        print(rf"{'✔':>1}")

        print(f"{'→ Attributes...':<{config.SEP_WIDTH-1}}", end="")
        moisture.name = "Moisture"
        moisture.attrs["description"] = "Specific humidity"
        moisture.attrs["units"] = r"kg kg$^{-1}$"
        moisture.attrs["file_id"] = "Q"
        moisture.attrs["short_name"] = "Q"
        # variables_dict['Moisture'] = moisture
        variables_loaded.append(moisture)
        print(rf"{'✔':>1}")
        print(f"{'-'*config.SEP_WIDTH}")

    # Z3
    if (
        'Geopotential Height' in variables_to_load
        or 'Moist Static Energy' in variables_to_load
        or 'Column Moist Static Energy' in variables_to_load
    ):
        print(f"{'Geopotential Height':<{config.SEP_WIDTH-1}}")

        print(f"{'→ Data...':<{config.SEP_WIDTH-1}}", end="")
        geopotential_height = xr.open_dataset(
            f"{input_data_directory}/SST_AQP3_Qobs_27_{experiment}_1D_20y_Z3.nc")['Z3']
        print(rf"{'✔':>1}")

        print(f"{'→ Attributes...':<{config.SEP_WIDTH-1}}", end="")
        geopotential_height.name = "Geopotential Height"
        geopotential_height.attrs["description"] = "Geopotential Height"
        geopotential_height.attrs["units"] = r"m"
        geopotential_height.attrs["file_id"] = "Z"
        geopotential_height.attrs["short_name"] = "Z"
        # variables_dict['Geopotential Height'] = geopotential_height
        variables_loaded.append(geopotential_height)
        print(rf"{'✔':>1}")
        print(f"{'-'*config.SEP_WIDTH}")

    # QRS
    if 'Shortwave Heating Rate' in variables_to_load or 'Column Shortwave Heating' in variables_to_load:
        print(f"{'Shortwave Heating Rate':<{config.SEP_WIDTH-1}}")

        print(f"{'→ Data...':<{config.SEP_WIDTH-1}}", end="")
        shortwave_heating_rate = xr.open_dataset(
            f"{input_data_directory}/SST_AQP3_Qobs_27_{experiment}_1D_20y_QRS.nc")['QRS']
        print(rf"{'✔':>1}")

        print(f"{'→ Attributes...':<{config.SEP_WIDTH-1}}", end="")
        shortwave_heating_rate.name = "Shortwave Heating Rate"
        shortwave_heating_rate.attrs["description"] = "Shortwave Heating Rate"
        shortwave_heating_rate.attrs["units"] = r"K s$^{-1}$"
        shortwave_heating_rate.attrs["file_id"] = "QRS"
        shortwave_heating_rate.attrs["short_name"] = "QRS"
        # variables_dict['Shortwave Heating Rate'] = shortwave_heating_rate
        variables_loaded.append(shortwave_heating_rate)
        print(rf"{'✔':>1}")
        print(f"{'-'*config.SEP_WIDTH}")

    # QRL
    if 'Longwave Heating Rate' in variables_to_load or 'Column Longwave Heating' in variables_to_load:
        print(f"{'Longwave Heating Rate':<{config.SEP_WIDTH-1}}")

        print(f"{'→ Data...':<{config.SEP_WIDTH-1}}", end="")
        longwave_heating_rate = xr.open_dataset(
            f"{input_data_directory}/SST_AQP3_Qobs_27_{experiment}_1D_20y_QRL.nc")['QRL']
        print(rf"{'✔':>1}")

        print(f"{'→ Attributes...':<{config.SEP_WIDTH-1}}", end="")
        longwave_heating_rate.name = "Longwave Heating Rate"
        longwave_heating_rate.attrs["description"] = "Longwave Heating Rate"
        longwave_heating_rate.attrs["units"] = r"K s$^{-1}$"
        longwave_heating_rate.attrs["file_id"] = "QRL"
        longwave_heating_rate.attrs["short_name"] = "QRL"
        # variables_dict['Longwave Heating Rate'] = longwave_heating_rate
        variables_loaded.append(longwave_heating_rate)
        print(rf"{'✔':>1}")
        print(f"{'-'*config.SEP_WIDTH}")

    # LHFLX
    if 'Latent Heat Flux' in variables_to_load:
        print(f"{'Latent Heat Flux':<{config.SEP_WIDTH-1}}")

        print(f"{'→ Data...':<{config.SEP_WIDTH-1}}", end="")
        latent_heat_flux = xr.open_dataset(
            f"{input_data_directory}/SST_AQP3_Qobs_27_{experiment}_1D_20y_LHFLX.nc")['LHFLX']
        print(rf"{'✔':>1}")

        print(f"{'→ Attributes...':<{config.SEP_WIDTH-1}}", end="")
        latent_heat_flux.name = "Latent Heat Flux"
        latent_heat_flux.attrs["description"] = "Surface Latent Heat Flux"
        latent_heat_flux.attrs["units"] = r"W m$^{-2}$"
        latent_heat_flux.attrs["file_id"] = "LH"
        latent_heat_flux.attrs["short_name"] = "LH"
        # variables_dict['Latent Heat Flux'] = latent_heat_flux
        variables_loaded.append(latent_heat_flux)
        print(rf"{'✔':>1}")
        print(f"{'-'*config.SEP_WIDTH}")

    # SHFLX
    if 'Sensible Heat Flux' in variables_to_load:
        print(f"{'Sensible Heat Flux':<{config.SEP_WIDTH-1}}")

        print(f"{'→ Data...':<{config.SEP_WIDTH-1}}", end="")
        sensible_heat_flux = xr.open_dataset(
            f"{input_data_directory}/SST_AQP3_Qobs_27_{experiment}_1D_20y_SHFLX.nc")['SHFLX']
        print(rf"{'✔':>1}")

        print(f"{'→ Attributes...':<{config.SEP_WIDTH-1}}", end="")
        sensible_heat_flux.name = "Sensible Heat Flux"
        sensible_heat_flux.attrs["description"] = "Surface Sensible Heat Flux"
        sensible_heat_flux.attrs["units"] = r"W m$^{-2}$"
        sensible_heat_flux.attrs["file_id"] = "SH"
        sensible_heat_flux.attrs["short_name"] = "SH"
        # variables_dict['Sensible Heat Flux'] = sensible_heat_flux
        variables_loaded.append(sensible_heat_flux)
        print(rf"{'✔':>1}")
        print(f"{'-'*config.SEP_WIDTH}")

    if 'Surface Pressure' in variables_to_load:
        print(f"{'Surface Pressure':<{config.SEP_WIDTH-1}}")

        print(f"{'→ Data...':<{config.SEP_WIDTH-1}}", end="")
        surface_pressure = xr.open_dataset(
            f"{input_data_directory}/SST_AQP3_Qobs_27_{experiment}_1D_20y_PS.nc"
        )["PS"]/100
        print(rf"{'✔':>1}")

        print(f"{'→ Attributes...':<{config.SEP_WIDTH-1}}", end="")
        surface_pressure.name = "Surface Pressure"
        surface_pressure.attrs["description"] = "Surface pressure in hPa"
        surface_pressure.attrs["units"] = r"hPa"
        surface_pressure.attrs["file_id"] = "PS"
        surface_pressure.attrs["short_name"] = "PS"
        # variables_dict['Surface Pressure'] = surface_pressure
        variables_loaded.append(surface_pressure)
        print(rf"{'✔':>1}")

    if 'Relative Humidity' in variables_to_load:
        print(f"{'Relative Humidity':<{config.SEP_WIDTH-1}}")

        print(f"{'→ Data...':<{config.SEP_WIDTH-1}}", end="")
        relative_humidity = xr.open_dataset(
            f"{input_data_directory}/SST_AQP3_Qobs_27_{experiment}_1D_20y_RELHUM.nc"
        )["RELHUM"]
        print(rf"{'✔':>1}")

        print(f"{'→ Attributes...':<{config.SEP_WIDTH-1}}", end="")
        relative_humidity.name = "Relative Humidity"
        relative_humidity.attrs["description"] = "Relative Humidity"
        relative_humidity.attrs["units"] = r"%"
        relative_humidity.attrs["file_id"] = "RELHUM"
        relative_humidity.attrs["short_name"] = "RELHUM"
        variables_loaded.append(relative_humidity)
        print(rf"{'✔':>1}")

    if 'Surface Longwave Flux' in variables_to_load:
        print(f"{'Surface Longwave Flux':<{config.SEP_WIDTH-1}}")

        print(f"{'→ Data...':<{config.SEP_WIDTH-1}}", end="")
        surface_longwave_flux = xr.open_dataset(
            f"{input_data_directory}/SST_AQP3_Qobs_27_{experiment}_1D_20y_FLNS.nc"
        )["FLNS"]
        print(rf"{'✔':>1}")

        print(f"{'→ Attributes...':<{config.SEP_WIDTH-1}}", end="")
        surface_longwave_flux.name = "Surface Longwave Flux"
        surface_longwave_flux.attrs["description"] = "Surface Longwave Radiative Flux"
        surface_longwave_flux.attrs["units"] = r"W m$^{-2}$"
        surface_longwave_flux.attrs["file_id"] = "FLNS"
        surface_longwave_flux.attrs["short_name"] = "FLNS"
        variables_loaded.append(surface_longwave_flux)
        print(rf"{'✔':>1}")

    if 'Surface Shortwave Flux' in variables_to_load:
        print(f"{'Surface Shortwave Flux':<{config.SEP_WIDTH-1}}")

        print(f"{'→ Data...':<{config.SEP_WIDTH-1}}", end="")
        surface_shortwave_flux = xr.open_dataset(
            f"{input_data_directory}/SST_AQP3_Qobs_27_{experiment}_1D_20y_FSNS.nc"
        )["FSNS"]
        print(rf"{'✔':>1}")

        print(f"{'→ Attributes...':<{config.SEP_WIDTH-1}}", end="")
        surface_shortwave_flux.name = "Surface Shortwave Flux"
        surface_shortwave_flux.attrs["description"] = "Surface Shortwave Radiative Flux"
        surface_shortwave_flux.attrs["units"] = r"W m$^{-2}$"
        surface_shortwave_flux.attrs["file_id"] = "FSNS"
        surface_shortwave_flux.attrs["short_name"] = "FSNS"
        variables_loaded.append(surface_shortwave_flux)
        print(rf"{'✔':>1}")

    if 'TOA Longwave Flux' in variables_to_load:
        print(f"{'TOA Longwave Flux':<{config.SEP_WIDTH-1}}")

        print(f"{'→ Data...':<{config.SEP_WIDTH-1}}", end="")
        toa_longwave_flux = xr.open_dataset(
            f"{input_data_directory}/SST_AQP3_Qobs_27_{experiment}_1D_20y_FLNT.nc"
        )["FLNT"]
        print(rf"{'✔':>1}")

        print(f"{'→ Attributes...':<{config.SEP_WIDTH-1}}", end="")
        toa_longwave_flux.name = "TOA Longwave Flux"
        toa_longwave_flux.attrs["description"] = "Top of Atmosphere Longwave Radiative Flux"
        toa_longwave_flux.attrs["units"] = r"W m$^{-2}$"
        toa_longwave_flux.attrs["file_id"] = "FLNT"
        toa_longwave_flux.attrs["short_name"] = "FLNT"
        variables_loaded.append(toa_longwave_flux)
        print(rf"{'✔':>1}")

    if 'TOA Shortwave Flux' in variables_to_load:
        print(f"{'TOA Shortwave Flux':<{config.SEP_WIDTH-1}}")

        print(f"{'→ Data...':<{config.SEP_WIDTH-1}}", end="")
        toa_shortwave_flux = xr.open_dataset(
            f"{input_data_directory}/SST_AQP3_Qobs_27_{experiment}_1D_20y_FSNT.nc"
        )["FSNT"]
        print(rf"{'✔':>1}")

        print(f"{'→ Attributes...':<{config.SEP_WIDTH-1}}", end="")
        toa_shortwave_flux.name = "TOA Shortwave Flux"
        toa_shortwave_flux.attrs["description"] = "Top of Atmosphere Shortwave Radiative Flux"
        toa_shortwave_flux.attrs["units"] = r"W m$^{-2}$"
        toa_shortwave_flux.attrs["file_id"] = "FSNT"
        toa_shortwave_flux.attrs["short_name"] = "FSNT"
        variables_loaded.append(toa_shortwave_flux)
        print(rf"{'✔':>1}")

    if 'Net Shortwave Flux' in variables_to_load:
        print(f"{'Net Shortwave Flux':<{config.SEP_WIDTH-1}}")

        print(f"{'→ Data...':<{config.SEP_WIDTH-1}}", end="")
        toa_shortwave_flux = xr.open_dataset(
            f"{input_data_directory}/SST_AQP3_Qobs_27_{experiment}_1D_20y_FSNT.nc"
        )["FSNT"]
        surface_shortwave_flux = xr.open_dataset(
            f"{input_data_directory}/SST_AQP3_Qobs_27_{experiment}_1D_20y_FSNS.nc"
        )["FSNS"]
        net_shortwave_flux = toa_shortwave_flux - surface_shortwave_flux
        print(rf"{'✔':>1}")

        print(f"{'→ Attributes...':<{config.SEP_WIDTH-1}}", end="")
        net_shortwave_flux.name = "Net Shortwave Flux"
        net_shortwave_flux.attrs["description"] = "Net Shortwave Radiative Flux"
        net_shortwave_flux.attrs["units"] = r"W m$^{-2}$"
        net_shortwave_flux.attrs["file_id"] = "NSW"
        net_shortwave_flux.attrs["short_name"] = "NSW"
        variables_loaded.append(net_shortwave_flux)
        print(rf"{'✔':>1}")

    if 'Net Longwave Flux' in variables_to_load:
        print(f"{'Net Longwave Flux':<{config.SEP_WIDTH-1}}")

        print(f"{'→ Data...':<{config.SEP_WIDTH-1}}", end="")
        toa_longwave_flux = xr.open_dataset(
            f"{input_data_directory}/SST_AQP3_Qobs_27_{experiment}_1D_20y_FLNT.nc"
        )["FLNT"]
        surface_longwave_flux = xr.open_dataset(
            f"{input_data_directory}/SST_AQP3_Qobs_27_{experiment}_1D_20y_FLNS.nc"
        )["FLNS"]
        net_longwave_flux = surface_longwave_flux - toa_longwave_flux
        print(rf"{'✔':>1}")

        print(f"{'→ Attributes...':<{config.SEP_WIDTH-1}}", end="")
        net_longwave_flux.name = "Net Longwave Flux"
        net_longwave_flux.attrs["description"] = "Net Longwave Radiative Flux"
        net_longwave_flux.attrs["units"] = r"W m$^{-2}$"
        net_longwave_flux.attrs["file_id"] = "NLW"
        net_longwave_flux.attrs["short_name"] = "NLW"
        variables_loaded.append(net_longwave_flux)
        print(rf"{'✔':>1}")

    if 'Cloud Ice Water Content' in variables_to_load:
        print(f"{'Cloud Ice Water Content':<{config.SEP_WIDTH-1}}")

        print(f"{'→ Data...':<{config.SEP_WIDTH-1}}", end="")
        cloud_ice_water_content = xr.open_dataset(
            f"{input_data_directory}/SST_AQP3_Qobs_27_{experiment}_1D_20y_CLDICE.nc"
        )["CLDICE"].transpose("time", "lat", "lon", "plev")
        print(rf"{'✔':>1}")

        print(f"{'→ Attributes...':<{config.SEP_WIDTH-1}}", end="")
        cloud_ice_water_content.name = "Cloud Ice Water Content"
        cloud_ice_water_content.attrs["description"] = "Cloud Ice Water Content"
        cloud_ice_water_content.attrs["units"] = r"kg kg$^{-1}$"
        cloud_ice_water_content.attrs["file_id"] = "CLDICE"
        cloud_ice_water_content.attrs["short_name"] = "CLDICE"
        variables_loaded.append(cloud_ice_water_content)
        print(rf"{'✔':>1}")

    if 'Cloud Liquid Water Content' in variables_to_load:
        print(f"{'Cloud Liquid Water Content':<{config.SEP_WIDTH-1}}")

        print(f"{'→ Data...':<{config.SEP_WIDTH-1}}", end="")
        cloud_liquid_water_content = xr.open_dataset(
            f"{input_data_directory}/SST_AQP3_Qobs_27_{experiment}_1D_20y_CLDLIQ.nc"
        )["CLDLIQ"].transpose("time", "lat", "lon", "plev")
        print(rf"{'✔':>1}")

        print(f"{'→ Attributes...':<{config.SEP_WIDTH-1}}", end="")
        cloud_liquid_water_content.name = "Cloud Liquid Water Content"
        cloud_liquid_water_content.attrs["description"] = "Cloud Liquid Water Content"
        cloud_liquid_water_content.attrs["units"] = r"kg kg$^{-1}$"
        cloud_liquid_water_content.attrs["file_id"] = "CLDLIQ"
        cloud_liquid_water_content.attrs["short_name"] = "CLDLIQ"
        variables_loaded.append(cloud_liquid_water_content)
        print(rf"{'✔':>1}")

    if 'Cloud Fraction' in variables_to_load:
        print(f"{'Cloud Fraction':<{config.SEP_WIDTH-1}}")

        print(f"{'→ Data...':<{config.SEP_WIDTH-1}}", end="")
        cloud_fraction = xr.open_dataset(
            f"{input_data_directory}/SST_AQP3_Qobs_27_{experiment}_1D_20y_CLOUD.nc"
        )["CLOUD"].transpose("time", "lat", "lon", "plev")
        print(rf"{'✔':>1}")

        print(f"{'→ Attributes...':<{config.SEP_WIDTH-1}}", end="")
        cloud_fraction.name = "Cloud Fraction"
        cloud_fraction.attrs["description"] = "Cloud Fraction"
        cloud_fraction.attrs["units"] = r""
        cloud_fraction.attrs["file_id"] = "CLOUD"
        cloud_fraction.attrs["short_name"] = "CLOUD"
        variables_loaded.append(cloud_fraction)
        print(rf"{'✔':>1}")

    # Zonal Moisture Advection
    if 'Zonal Moisture Advection' in variables_to_load:
        print(f"{'Zonal Moisture Advection':<{config.SEP_WIDTH-1}}")
        print(f"{'→ Data...':<{config.SEP_WIDTH-1}}", end="")
        zonal_moisture_advection = xr.open_dataset(
            f"/glade/derecho/scratch/sressel/vertically_resolved_budget_terms/moisture/{experiment}_zonal_advection.nc"
        )["Zonal Advection"]
        print(rf"{'✔':>1}")

        print(f"{'→ Attributes...':<{config.SEP_WIDTH-1}}", end="")
        zonal_moisture_advection.name = "Zonal Moisture Advection"
        zonal_moisture_advection.attrs["description"] = "Zonal Moisture Advection"
        zonal_moisture_advection.attrs["units"] = r"kg s$^{-1}$"
        zonal_moisture_advection.attrs["file_id"] = "zonal_advection"
        zonal_moisture_advection.attrs["short_name"] = "zonal_advection"
        # variables_dict['Zonal Wind'] = zonal_wind
        variables_loaded.append(zonal_moisture_advection)
        print(rf"{'✔':>1}")
        print(f"{'-'*config.SEP_WIDTH}")

        # Zonal Moisture Advection
    if 'Meridional Moisture Advection' in variables_to_load:
        print(f"{'Meridional Moisture Advection':<{config.SEP_WIDTH-1}}")
        print(f"{'→ Data...':<{config.SEP_WIDTH-1}}", end="")
        meridional_moisture_advection = xr.open_dataset(
            f"/glade/derecho/scratch/sressel/vertically_resolved_budget_terms/moisture/{experiment}_meridional_advection.nc"
        )["Meridional Advection"]
        print(rf"{'✔':>1}")

        print(f"{'→ Attributes...':<{config.SEP_WIDTH-1}}", end="")
        meridional_moisture_advection.name = "Meridional Moisture Advection"
        meridional_moisture_advection.attrs["description"] = "Meridional Moisture Advection"
        meridional_moisture_advection.attrs["units"] = r"kg s$^{-1}$"
        meridional_moisture_advection.attrs["file_id"] = "meridional_advection"
        meridional_moisture_advection.attrs["short_name"] = "meridional_advection"
        # variables_dict['Zonal Wind'] = zonal_wind
        variables_loaded.append(meridional_moisture_advection)
        print(rf"{'✔':>1}")
        print(f"{'-'*config.SEP_WIDTH}")

        # Zonal Moisture Advection
    if 'Vertical Moisture Advection' in variables_to_load:
        print(f"{'Vertical Moisture Advection':<{config.SEP_WIDTH-1}}")
        print(f"{'→ Data...':<{config.SEP_WIDTH-1}}", end="")
        vertical_moisture_advection = xr.open_dataset(
            f"/glade/derecho/scratch/sressel/vertically_resolved_budget_terms/moisture/{experiment}_vertical_advection.nc"
        )["Vertical Advection"]
        print(rf"{'✔':>1}")

        print(f"{'→ Attributes...':<{config.SEP_WIDTH-1}}", end="")
        vertical_moisture_advection.name = "Vertical Moisture Advection"
        vertical_moisture_advection.attrs["description"] = "Vertical Moisture Advection"
        vertical_moisture_advection.attrs["units"] = r"kg s$^{-1}$"
        vertical_moisture_advection.attrs["file_id"] = "vertical_advection"
        vertical_moisture_advection.attrs["short_name"] = "vertical_advection"
        # variables_dict['Zonal Wind'] = zonal_wind
        variables_loaded.append(vertical_moisture_advection)
        print(rf"{'✔':>1}")
        print(f"{'-'*config.SEP_WIDTH}")

    print(f"{'='*config.SEP_WIDTH}")
    print(f"{'Derived Variables':^{config.SEP_WIDTH}}")
    print(f"{'='*config.SEP_WIDTH}")

    # MSE
    if 'Moist Static Energy' in variables_to_load or 'Column Moist Static Energy' in variables_to_load:
        print(f"{'Moist Static Energy':<{config.SEP_WIDTH-1}}")

        print(f"{'→ Data...':<{config.SEP_WIDTH-1}}", end="")
        moist_static_energy = (
            config.HEAT_OF_VAPORIZATION*moisture
            + config.SPECIFIC_HEAT*temperature
            + config.GRAVITY*geopotential_height
        )
        print(rf"{'✔':>1}")

        print(f"{'→ Attributes...':<{config.SEP_WIDTH-1}}", end="")
        moist_static_energy.name = "Moist Static Energy"
        moist_static_energy.attrs["description"] = "Moist Static Energy"
        moist_static_energy.attrs["units"] = r"J kg$^{-1}$"
        moist_static_energy.attrs["file_id"] = "MSE"
        moist_static_energy.attrs["short_name"] = "MSE"
        # variables_dict['Moist Static Energy'] = moist_static_energy
        variables_loaded.append(moist_static_energy)
        print(rf"{'✔':>1}")
        print(f"{'-'*config.SEP_WIDTH}")

    # CIT
    if 'Column Temperature' in variables_to_load:
        print(f"{f'Column Temperature ({coords.pressure_subset_bounds.start}-{coords.pressure_subset_bounds.stop} hPa)':<{config.SEP_WIDTH-1}}")
        print(f"{'→ Data...':<{config.SEP_WIDTH-1}}", end="")
        column_temperature = (100/config.GRAVITY)*temperature.sel(plev=coords.pressure_subset_bounds).integrate('plev')
        print(rf"{'✔':>1}")

        print(f"{'→ Attributes...':<{config.SEP_WIDTH-1}}", end="")
        column_temperature.name = "Column Temperature"
        column_temperature.attrs["description"] = "Column integrated temperature"
        column_temperature.attrs["units"] = r"K kg m$^{-2}$"
        column_temperature.attrs["file_id"] = "CIT"
        column_temperature.attrs["short_name"] = r"$\langle$T$\rangle$"
        column_temperature.attrs["integration_bounds"]  = (coords.pressure_subset_bounds.start, coords.pressure_subset_bounds.stop)
        # variables_dict['Column Temperature'] = column_temperature
        variables_loaded.append(column_temperature)
        print(rf"{'✔':>1}")
        print(f"{'-'*config.SEP_WIDTH}")

    # CWV
    if 'Column Water Vapor' in variables_to_load:
        print(f"{f'Column Water Vapor ({coords.pressure_subset_bounds.start}-{coords.pressure_subset_bounds.stop} hPa)':<{config.SEP_WIDTH-1}}")
        print(f"{'→ Data...':<{config.SEP_WIDTH-1}}", end="")
        column_water_vapor = (100/config.GRAVITY)*moisture.sel(plev=coords.pressure_subset_bounds).integrate('plev')
        print(rf"{'✔':>1}")

        print(f"{'→ Attributes...':<{config.SEP_WIDTH-1}}", end="")
        column_water_vapor.name = "Column Water Vapor"
        column_water_vapor.attrs["description"] =\
            "Column water vapor"
        column_water_vapor.attrs["units"] = r"mm"
        column_water_vapor.attrs["file_id"] = "CWV"
        column_water_vapor.attrs["short_name"] = r"$\langle$Q$\rangle$"
        column_water_vapor.attrs["integration_bounds"]  = (coords.pressure_subset_bounds.start, coords.pressure_subset_bounds.stop)
        # variables_dict['Column Water Vapor'] = column_water_vapor
        variables_loaded.append(column_water_vapor)
        print(rf"{'✔':>1}")
        print(f"{'-'*config.SEP_WIDTH}")

    # CSWH
    if 'Column Shortwave Heating' in variables_to_load:
        print(f"{f'Column Shortwave Heating ({coords.pressure_subset_bounds.start}-{coords.pressure_subset_bounds.stop} hPa)':<{config.SEP_WIDTH-1}}")
        print(f"{'→ Data...':<{config.SEP_WIDTH-1}}", end="")
        column_shortwave_heating = config.SPECIFIC_HEAT*(100/config.GRAVITY)*shortwave_heating_rate.sel(plev=coords.pressure_subset_bounds).integrate('plev')
        print(rf"{'✔':>1}")

        print(f"{'→ Attributes...':<{config.SEP_WIDTH-1}}", end="")
        column_shortwave_heating.name = "Column Shortwave Heating"
        column_shortwave_heating.attrs["description"] = "Column shortwave heating from 1000 hPa to 100 hPa in energy units"
        column_shortwave_heating.attrs["units"] = r"W m$^{-2}$"
        column_shortwave_heating.attrs["file_id"] = "CSW"
        column_shortwave_heating.attrs["short_name"] = r"$\langle$SW$\rangle$"
        column_shortwave_heating.attrs["integration_bounds"]  = (coords.pressure_subset_bounds.start, coords.pressure_subset_bounds.stop)
        # variables_dict['Column Shortwave Heating'] = column_shortwave_heating
        variables_loaded.append(column_shortwave_heating)
        print(rf"{'✔':>1}")
        print(f"{'-'*config.SEP_WIDTH}")

    # CLWH
    if 'Column Longwave Heating' in variables_to_load:
        print(f"{f'Column Longwave Heating ({coords.pressure_subset_bounds.start}-{coords.pressure_subset_bounds.stop} hPa)':<{config.SEP_WIDTH-1}}")
        print(f"{'→ Data...':<{config.SEP_WIDTH-1}}", end="")
        column_longwave_heating = config.SPECIFIC_HEAT*(100/config.GRAVITY)*longwave_heating_rate.sel(plev=coords.pressure_subset_bounds).integrate('plev')
        print(rf"{'✔':>1}")

        print(f"{'→ Attributes...':<{config.SEP_WIDTH-1}}", end="")
        column_longwave_heating.name = "Column Longwave Heating"
        column_longwave_heating.attrs["description"] = "Column longwave heating from 1000 hPa to 100 hPa in energy units"
        column_longwave_heating.attrs["units"] = r"W m$^{-2}$"
        column_longwave_heating.attrs["file_id"] = "CLW"
        column_longwave_heating.attrs["short_name"] = r"$\langle$LW$\rangle$"
        column_longwave_heating.attrs["integration_bounds"]  = (coords.pressure_subset_bounds.start, coords.pressure_subset_bounds.stop)
        # variables_dict['Column Longwave Heating'] = column_longwave_heating
        variables_loaded.append(column_longwave_heating)
        print(rf"{'✔':>1}")
        print(f"{'-'*config.SEP_WIDTH}")


    if 'Column Moist Static Energy' in variables_to_load:
        print(f"{f'Column Moist Static Energy ({coords.pressure_subset_bounds.start}-{coords.pressure_subset_bounds.stop} hPa)':<{config.SEP_WIDTH-1}}")

        print(f"{'→ Data...':<{config.SEP_WIDTH-1}}", end="")
        column_moist_static_energy = (100/config.GRAVITY)*moist_static_energy.sel(plev=coords.pressure_subset_bounds).integrate('plev')
        print(rf"{'✔':>1}")

        print(f"{'→ Attributes...':<{config.SEP_WIDTH-1}}", end="")
        column_moist_static_energy.name = "Column Moist Static Energy"
        column_moist_static_energy.attrs["description"] = "Column-integrated Moist Static Energy from 100 hPa-975 hPa"
        column_moist_static_energy.attrs["units"] = r"J m$^{-2}$"
        column_moist_static_energy.attrs["file_id"] = "CMSE"
        column_moist_static_energy.attrs["short_name"] = "$\langle$m$\rangle$"
        column_moist_static_energy.attrs["integration_bounds"]  = (coords.pressure_subset_bounds.start, coords.pressure_subset_bounds.stop)
        # variables_dict['Column Moist Static Energy'] = column_moist_static_energy
        variables_loaded.append(column_moist_static_energy)
        print(rf"{'✔':>1}")

    variables_dict = {variable_data.name:variable_data for variable_data in variables_loaded if variable_data.name in variables_to_load}
    return variables_dict

def load_multi_experiment_daily_model_level_data(variables_to_load, multi_experiment_daily_variables = {}, reload_data=False):
    """
    Load a data variables across multiple experiments and store them in a dictionary.

    Parameters:
    - variables_to_load (list): List of variable names to load.

    Returns:
    - dict: Updated dictionary containing the loaded variables.
    """
    print(f"{f'Loading daily model-level data':^{config.SEP_WIDTH}}")

    # Initialize dictionary to store loaded data if reloading
    variable_data_by_experiment = {}
    if reload_data:
        multi_experiment_daily_variables = {}

    for index, variable_name in enumerate(variables_to_load):
        # Print the current variable being processed
        print(f"{'='*config.SEP_WIDTH}")
        print(f"{f'({index+1}/{len(variables_to_load)}) {variable_name}...':<{config.SEP_WIDTH}}")
        print(f"{'-'*config.SEP_WIDTH}")

        if variable_name in multi_experiment_daily_variables:
            print("Variable already loaded")
            continue

        # Initialize list to hold data for this variable across experiments
        variable_data_by_experiment[variable_name] = []

        # Iterate through each experiment and try to load the variable data
        print("Loading data...")
        for experiment in coords.experiments.values:
            print(f"{f'    Experiment: {experiment}':<{config.SEP_WIDTH-1}}", end="")

            # Define file location for the current experiment
            file_location = rf"/glade/campaign/univ/uwas0152/post_processed_data/{experiment}/daily_model-level_data"

            # Find NetCDF files corresponding to the variable
            data_files = glob.glob(f"{file_location}/SST_AQP3_Qobs_27_{experiment}_1D_{variable_name}.nc")

            # Check if data files are found
            if data_files:
                # Open and store each found file
                for file in data_files:
                    variable_data_by_experiment[variable_name].append(xr.open_dataarray(file))
                print(rf"{'✔':>1}")
            else:
                # Indicate that no data was found for this variable
                print(rf"{'✘':>1}")
                print(f"    No data")

        # If data was loaded, concatenate across experiments and store in the subset dictionary
        if variable_data_by_experiment[variable_name]:
            print(f"{'Concatenating data...':<{config.SEP_WIDTH-1}}", end="")
            multi_experiment_daily_variables[variable_name] = xr.concat(
                variable_data_by_experiment[variable_name], dim=coords.experiments
            )
            print(rf"{'✔':>1}")

    print(f"{'-'*config.SEP_WIDTH}")
    print("Data Loaded")

    return multi_experiment_daily_variables

def load_multi_experiment_processed_data(variables_to_load, processing_type, multi_experiment_variables_processed={}, reload_data=False):
    """
    Load processed data variables across multiple experiments and store them in a dictionary.

    Parameters:
    - variables_to_load (list): List of variable names to load.
    - multi_experiment_variables_processed (dict): Dictionary to store loaded variables.
    - reload_subset (bool): If True, resets `multi_experiment_variables_subset` before loading.

    Returns:
    - dict: Updated dictionary containing the loaded variables.
    """
    logger.info(f"Loading {processing_type} data")

    # Initialize dictionary to store loaded data if reloading
    variable_data_by_experiment = {}
    if not multi_experiment_variables_processed or reload_data:
        multi_experiment_variables_processed = {}

    for index, variable_name in enumerate(variables_to_load):
        # Print the current variable being processed
        logger.info(f"({index+1}/{len(variables_to_load)}) {variable_name}...")

        # If the variable is already loaded, skip it
        if variable_name in multi_experiment_variables_processed:
            continue

        # Initialize list to hold data for this variable across experiments
        variable_data_by_experiment[variable_name] = []

        # Iterate through each experiment and try to load the variable data
        for experiment in coords.experiments.values:
            logger.info(f"    Experiment: {experiment}...")

            # Define file location for the current experiment
            file_location = rf"/glade/campaign/univ/uwas0152/post_processed_data/{experiment}/variables_{processing_type}"

            # Find NetCDF files corresponding to the variable
            processed_files = glob.glob(f"{file_location}/{variable_name.lower().replace(' ', '_')}.nc")

            # Check if data files are found
            if processed_files:
                # Open and store each found file
                for file in processed_files:
                    variable_data_by_experiment[variable_name].append(xr.open_dataarray(file))
            else:
                # Indicate that no data was found for this variable
                logger.warning(f"No data found for {variable_name} in experiment {experiment}")

        # If data was loaded, concatenate across experiments and store in the processed dictionary
        if variable_data_by_experiment[variable_name]:
            # print(f"{'Concatenating data...':<{config.SEP_WIDTH-1}}", end="")
            multi_experiment_variables_processed[variable_name] = xr.concat(
                variable_data_by_experiment[variable_name], dim=coords.experiments
            )

    # logger.info(f"{'-'*config.SEP_WIDTH}")
    logger.info("Finished")

    return multi_experiment_variables_processed

# def load_multi_experiment_mjo_filtered_data():
