import netCDF4
import pandas as pd
import numpy as np
import datetime
import gc

def get_era5_data(year, quarter):
    era5_year = str(year)
    data = netCDF4.Dataset('/home/wheatley/WFD/ERA5/{}_{}/data_stream-oper.nc'.format(era5_year, quarter), 'r')
    unit_time = data.variables['valid_time'].units
    times = data.variables['valid_time'][:]
    lat = data.variables['latitude'][:]
    long = data.variables['longitude'][:]

    variables_to_extract = ['number', 'valid_time', 'latitude', 'longitude', 'expver', 'u10', 'v10', 'd2m', 't2m', 'msl', 'sst', 'sp', 'u100', 'v100', 'u10n', 'v10n', 'fg10', 'i10fg', 'slhf', 'sshf', 'bld', 'blh', 'chnk', 'fsr', 'zust', 'lsm']
    extracted_data = []
    for var_name in variables_to_extract:
        extracted_data.append(data.variables[var_name][:])

    ref_date_str = unit_time.split("since ")[1]
    ref_date = datetime.datetime.strptime(ref_date_str, "%Y-%m-%d")

    data.close()
    del data
    gc.collect()
    return times, lat, long, extracted_data, ref_date

def calc_wf_data(times, lat, long, extracted_data, ref_date, value):

    number, valid_time, latitude, longitude, expver, u10, v10, d2m, t2m, msl, sst, sp, u100, v100, u10n, v10n, fg10, i10fg, slhf, sshf, bld, blh, chnk, fsr, zust, lsm = extracted_data

    # Find square difference of lat-longs and select the minimum difference
    sq_diff_lat = ((lat - value['latitude']) ** 2).argmin()
    sq_diff_long = ((long - value['longitude']) ** 2).argmin()

    # Initialize lists/arrays
    timestamp = list()
    number_data = np.zeros(shape=(len(times),1))
    valid_time_data = np.zeros(shape=(len(times),1))
    expver_data = []
    temp_data = np.zeros(shape=(len(times),1))
    pressure_data = np.zeros(shape=(len(times),1))
    dew_data = np.zeros(shape=(len(times),1))
    sensible_heat_data = np.zeros(shape=(len(times),1))
    wind_comp_u100 = np.zeros(shape=(len(times),1))
    wind_comp_v100 = np.zeros(shape=(len(times),1))
    wind_comp_u10 = np.zeros(shape=(len(times),1))
    wind_comp_v10 = np.zeros(shape=(len(times),1))
    wind_speed100 = np.zeros(shape=(len(times),1))
    wind_speed10 = np.zeros(shape=(len(times),1))
    wind_direction100 = np.zeros(shape=(len(times),1))
    wind_direction10 = np.zeros(shape=(len(times),1))
    msl_data = np.zeros(shape=(len(times),1))
    sst_data = np.zeros(shape=(len(times),1))
    u10n_data = np.zeros(shape=(len(times),1))
    v10n_data = np.zeros(shape=(len(times),1))
    fg10_data = np.zeros(shape=(len(times),1))
    i10fg_data = np.zeros(shape=(len(times),1))
    slhf_data = np.zeros(shape=(len(times),1))
    bld_data = np.zeros(shape=(len(times),1))
    blh_data = np.zeros(shape=(len(times),1))
    chnk_data = np.zeros(shape=(len(times),1))
    fsr_data = np.zeros(shape=(len(times),1))
    zust_data = np.zeros(shape=(len(times),1))
    lsm_data = np.zeros(shape=(len(times),1))

    # Air density calculation
    air_density = np.zeros(shape=(len(times), 1))
    relative_humidity = np.zeros(shape=(len(times), 1))

    # Constants for air density calculation
    Alpha = 1.00062
    Beta = 3.14*10**-8
    Gamma = 5.6*10**-7
    C1 = 1.2378847*10**-5
    C2 = -1.9121316*10**-2
    C3 = 33.93711047
    C4 = -6.3431645*10**3
    ao = 1.58123*10**-6
    a1 = -2.9331*10**-8
    a2 = 1.1043*10**-10
    bo = 5.707*10**-6
    b1 = -2.051*10**-8
    co = 1.9898*10**-4
    c1 = -2.376*10**-6
    d = 1.83*10**-11
    e = -0.765*10**-8
    R = 8.314472
    Ma = 28.96546*10**-3
    Mv = 18.01525*10**-3

    def tK(T):
        return 273.15 + T

    def xv(t, P, RH):
        # Ensure RH is used as a fraction (0 to 1)
        RH = RH / 100 if RH > 1 else RH
        return RH * (Alpha + Beta * P + Gamma * t**2) * np.exp(C1 * tK(t)**2 + C2 * tK(t) + C3 + C4 / tK(t)) / P

    def Z(t, P, RH):
        x_v = xv(t, P, RH)
        return 1 - (P / tK(t)) * (ao + a1 * t + a2 * t**2 + (bo + b1 * t) * x_v + (co + c1 * t) * x_v**2) + (P**2 / tK(t)**2) * (d + e * x_v**2)

    def calculate_rho(t, P, RH):
        x_v = xv(t, P, RH)
        return ((P * Ma) / (Z(t, P, RH) * R * tK(t))) * (1 - x_v * (1 - (Mv / Ma)))

    # Create timeseries
    for index, time in enumerate(times):
        date_time = ref_date + datetime.timedelta(seconds=int(time))
        timestamp.append(date_time)

        temp_data[index] = t2m[index, sq_diff_lat, sq_diff_long]
        pressure_data[index] = sp[index, sq_diff_lat, sq_diff_long]
        dew_data[index] = d2m[index, sq_diff_lat, sq_diff_long]
        sensible_heat_data[index] = sshf[index, sq_diff_lat, sq_diff_long]
        wind_comp_u100[index] = u100[index, sq_diff_lat, sq_diff_long]
        wind_comp_v100[index] = v100[index, sq_diff_lat, sq_diff_long]
        wind_comp_u10[index] = u10[index, sq_diff_lat, sq_diff_long]
        wind_comp_v10[index] = v10[index, sq_diff_lat, sq_diff_long]
        msl_data[index] = msl[index, sq_diff_lat, sq_diff_long]
        sst_data[index] = sst[index, sq_diff_lat, sq_diff_long]
        u10n_data[index] = u10n[index, sq_diff_lat, sq_diff_long]
        v10n_data[index] = v10n[index, sq_diff_lat, sq_diff_long]
        fg10_data[index] = fg10[index, sq_diff_lat, sq_diff_long]
        i10fg_data[index] = i10fg[index, sq_diff_lat, sq_diff_long]
        slhf_data[index] = slhf[index, sq_diff_lat, sq_diff_long]
        bld_data[index] = bld[index, sq_diff_lat, sq_diff_long]
        blh_data[index] = blh[index, sq_diff_lat, sq_diff_long]
        chnk_data[index] = chnk[index, sq_diff_lat, sq_diff_long]
        fsr_data[index] = fsr[index, sq_diff_lat, sq_diff_long]
        zust_data[index] = zust[index, sq_diff_lat, sq_diff_long]
        lsm_data[index] = lsm[index, sq_diff_lat, sq_diff_long]

        # Calculate relative humidity using Magnus formula
        # RH is calculated using the dew point temperature and the actual temperature
        relative_humidity[index] = 100 * (np.exp((17.625 * (dew_data[index]-273.15)) / (243.04 + (dew_data[index]-273.15))) / np.exp((17.625 * (temp_data[index]-273.15)) / (243.04 + (temp_data[index]-273.15))))

        # Calculate air density using the provided formula
        air_density[index] = calculate_rho(temp_data[index]-273.15, pressure_data[index], relative_humidity[index]) # Using temperature in Celsius

    # Wind Speed and Direction
    for i in range(len(wind_comp_u100)):
        wind_speed100[i] = (np.sqrt(wind_comp_u100[i]**2 + wind_comp_v100[i]**2))
        wind_direction100[i] = (np.arctan2(wind_comp_u100[i], wind_comp_v100[i]) * 180 / np.pi)
        wind_direction100[i] += 180
        wind_direction100[i] = wind_direction100[i] % 360
    for i in range(len(wind_comp_u10)):
        wind_speed10[i] = (np.sqrt(wind_comp_u10[i]**2 + wind_comp_v10[i]**2))
        wind_direction10[i] = (np.arctan2(wind_comp_u10[i], wind_comp_v10[i]) * 180 / np.pi)
        wind_direction10[i] += 180
        wind_direction10[i] = wind_direction10[i] % 360
    gc.collect()

    # Creating dataframe

    # Dimensions
    df_raw = pd.DataFrame(timestamp, columns=['timestamp']) # Changed to use the timestamp list directly
    df_raw = pd.DataFrame(timestamp_id.loc[timestamp, 'ts_id'].values , columns=['ts_id']) # This was the old line that caused the error
    df_raw["latitude"] = value["latitude"]
    df_raw["longitude"] = value["longitude"]

    # Variables
    df_raw['temperature'] = temp_data.squeeze()
    df_raw['pressure'] = pressure_data.squeeze()
    df_raw['dew_point'] = dew_data.squeeze()
    df_raw['surface_sensible_heat_flux'] = sensible_heat_data.squeeze()
    df_raw['u100'] = wind_comp_u100.squeeze()
    df_raw['v100'] = wind_comp_v100.squeeze()
    df_raw['u10'] = wind_comp_u10.squeeze()
    df_raw['v10'] = wind_comp_v10.squeeze()
    df_raw['mean_sea_level_pressure'] = msl_data.squeeze()
    df_raw['sea_surface_temperature'] = sst_data.squeeze()
    df_raw['u10n'] = u10n_data.squeeze()
    df_raw['v10n'] = v10n_data.squeeze()
    df_raw['fg10'] = fg10_data.squeeze()
    df_raw['i10fg'] = i10fg_data.squeeze()
    df_raw['surface_latent_heat_flux'] = slhf_data.squeeze()
    df_raw['boundary_layer_dissipation'] = bld_data.squeeze()
    df_raw['boundary_layer_height'] = blh_data.squeeze()
    df_raw['charnock'] = chnk_data.squeeze()
    df_raw['forecast_surface_roughness'] = fsr_data.squeeze()
    df_raw['friction_velocity'] = zust_data.squeeze()
    df_raw['land_sea_mask'] = lsm_data.squeeze()

    # Calculated Variables
    df_calc = pd.DataFrame(timestamp, columns=['timestamp']) # Changed to use the timestamp list directly
    df_calc = pd.DataFrame(timestamp_id.loc[timestamp, 'ts_id'].values, columns=['ts_id']) # This was the old line that caused the error
    df_calc["latitude"] = value["latitude"]
    df_calc["longitude"] = value["longitude"]
    df_calc["ws100"] = wind_speed100
    df_calc["wd100"] = wind_direction100
    df_calc["ws10"] = wind_speed10
    df_calc["wd10"] = wind_direction10
    df_calc["air_density"] = air_density
    df_calc["relative_humidity"] = relative_humidity

    # Free RAM
    del timestamp
    del temp_data
    del pressure_data
    del wind_speed100
    del wind_direction100
    del wind_speed10
    del wind_direction10
    del dew_data
    del sensible_heat_data
    del wind_comp_u100
    del wind_comp_v100
    del wind_comp_u10
    del wind_comp_v10
    del expver_data
    del msl_data
    del sst_data
    del u10n_data
    del v10n_data
    del fg10_data
    del i10fg_data
    del slhf_data
    del bld_data
    del blh_data
    del chnk_data
    del fsr_data
    del zust_data
    del lsm_data
    del air_density
    del relative_humidity
    gc.collect()

    # Save to the database
    df_raw.to_sql('era5_raw_data', con=engine, if_exists='append', index=False)
    del df_raw
    df_calc.to_sql('era5_calculated_data', con=engine, if_exists='append', index=False)
    del df_calc
    gc.collect()

#from src.sshDb import ssh_db
#engine = ssh_db.connect_ssh_db()
import sqlalchemy
import os
#os.environ['SQLITE_TMPDIR'] = '/home/wfd/tmp/'

#path = '/home/wfd/Backups/wfd.db'
path = '/home/wheatley/WFD/wfd.db'
engine = sqlalchemy.create_engine('sqlite:///' + path)

from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import text
global timestamp_id

# YOU NEED TO DELETE THE TABLES era5_raw_data and era5_calculated_data before running this script. Otherwise, it will append the data to the existing tables, creating duplicates.
# delete the tables
print("Dropping the tables era5_raw_data and era5_calculated_data")
with engine.connect() as con:
    con.execute(text("DROP TABLE IF EXISTS era5_raw_data"))
    con.execute(text("DROP TABLE IF EXISTS era5_calculated_data"))

if 'timestamp_id' not in locals():
    print("Getting timestamp_id from the database")
    timestamp_id = pd.read_sql_table('timestamp_id', con=engine)
    timestamp_id.set_index('timestamp', inplace=True)

# Get ERA5 data for the years 2013-2023
for i in range(2013, 2024):
    for j in range(1, 5):
        print("Getting ERA5 data for the year {} and quarter {}".format(i, j))
        times, lat, long, extracted_data, ref_date = get_era5_data(i, j)

        # Calculate the data for EACH latitude and longitude pair in the yearly era5 data
        max_workers = 1 # Adjust based on your CPU cores
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for lat_index, latitude in enumerate(lat):
                for long_index, longitude in enumerate(long):
                    value = {'latitude': latitude, 'longitude': longitude}
                    #print("Processing latitude {} and longitude {}".format(latitude, longitude))
                    executor.submit(calc_wf_data, times, lat, long, extracted_data, ref_date, value)

del i, j, times, lat, long, extracted_data, ref_date, timestamp_id
gc.collect()

# vacuum the database
print("Vacuuming the database")
with engine.connect() as con:
    con.execute(text("VACUUM"))

print("Finished processing ERA5 data")