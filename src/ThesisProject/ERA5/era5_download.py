import cdsapi
c = cdsapi.Client()

month=[['01', '02', '03'], ['04', '05', '06'], ['07', '08', '09'],['10', '11', '12']]

for year in [str(y) for y in range(2013,2025)]:
    i=0
    for m in month:
        i=i+1
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': ['reanalysis'],
                'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature',
                             '2m_temperature', 'mean_sea_level_pressure', 'sea_surface_temperature', 'surface_pressure',
                             '100m_u_component_of_wind', '100m_v_component_of_wind', '10m_u_component_of_neutral_wind',
                             '10m_v_component_of_neutral_wind', '10m_wind_gust_since_previous_post_processing',
                             'instantaneous_10m_wind_gust', 'surface_latent_heat_flux', 'surface_sensible_heat_flux',
                             'air_density_over_the_oceans', 'boundary_layer_dissipation', 'boundary_layer_height',
                             'charnock', 'forecast_surface_roughness', 'friction_velocity', 'land_sea_mask'],
                'year': ['{}'.format(year)],
                'month': m,
                'day': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                    '13', '14', '15',
                    '16', '17', '18',
                    '19', '20', '21',
                    '22', '23', '24',
                    '25', '26', '27',
                    '28', '29', '30',
                    '31',
                ],
                'time': [
                    '00:00', '01:00', '02:00',
                    '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00',
                    '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00',
                    '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00',
                    '21:00', '22:00', '23:00',
                ],
                'data_format': 'netcdf',
                'download_format': 'unarchived',
                'area': [42.03, 25.9, 35.9, 44.57]
            }).download(f'ERA5/{year}_{i}.nc')