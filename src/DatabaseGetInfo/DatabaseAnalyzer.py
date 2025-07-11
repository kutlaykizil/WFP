import sqlite3
import pandas as pd
import numpy as np
import rasterio

def get_timestamps_from_dates(start_date, end_date):
    """
    Converts start and end date strings to corresponding timestamps.

    Args:
    start_date (str): The starting date string.
    end_date (str): The ending date string.

    Returns:
        tuple: A tuple containing the start and end timestamps as integers.
    """
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    # Conversion logic from datetime_to_ts_id
    start_ts_id = 3682081 + int((start_date - pd.Timestamp('2017-01-01')).total_seconds() / 60)
    end_ts_id = 3682081 + int((end_date - pd.Timestamp('2017-01-01')).total_seconds() / 60)

    return start_ts_id, end_ts_id


def ts_id_to_datetime(ts_id):
    """
    Converts a timestamp ID to a datetime object.

    Args:
        ts_id (int): The timestamp ID.

    Returns:
        datetime.datetime: The corresponding datetime object.
    """
    start_date = pd.Timestamp('2017-01-01 00:00:00')
    return start_date + pd.Timedelta(minutes=ts_id - 3682081)


class WindFarmAnalyzer:
    """
    Class for analyzing wind farm data from a SQLite database.
    """

    def __init__(self, db_path):
        """
        Initializes the analyzer with the database path.

        Args:
            db_path (str): Path to the SQLite database file.
        """
        self.db_path = db_path
        self.connection = None
        self.cursor = None

    def connect_to_db(self):
        """
        Connects to the SQLite database and returns a new cursor.
        """
        connection = sqlite3.connect(self.db_path)  # Create a new connection each time
        cursor = connection.cursor()
        return connection, cursor

    def get_wf_id_list(self):
        wf_id_list = pd.DataFrame()
        connection, cursor = self.connect_to_db()
        query = """SELECT DISTINCT wf_id FROM wf where wf_id is not null and license_status is 'Yürürlükte' order by wf_id"""
        cursor.execute(query)
        wf_id_list = cursor.fetchall()
        cursor.close()
        connection.close()

        for wf_id in wf_id_list:
            wf_id_list = pd.DataFrame(wf_id_list, columns=['wf_id'])
        # reindex and change the column name of the dataframe
        wf_id_list = wf_id_list.reset_index(drop=True)
        wf_id_list.columns = ['wf_id']
        return wf_id_list['wf_id'].tolist()

    def get_farm_info(self, wf_id):
        """
        Fetches information about a specific wind farm site.

        Args:
            wf_id (int): The ID of the wind farm site.

        Returns:
            dict: A dictionary containing site information.
        """
        connection, cursor = self.connect_to_db()
        query = """
            SELECT wf.*, wf_turbines.*, ministry_of_energy.*
            FROM wf 
            JOIN wf_turbines ON wf.wf_id = wf_turbines.wf_id
            JOIN ministry_of_energy ON wf.wf_id = ministry_of_energy.wf_id
            WHERE wf.wf_id = ?
        """
        cursor.execute(query, (wf_id,))
        result = cursor.fetchone()
        cursor.close()
        connection.close()
        if result:
            columns = [column[0] for column in cursor.description]
            farm_info = dict(zip(columns, result))
            return farm_info
        else:
            print(f"No site found with ID {wf_id}")
            return None

    def get_moe_data(self, wf_id):
        """
        Fetches capacity and power data from the 'ministry_of_energy' table.

        Args:
            wf_id (int): The ID of the wind farm site.

        Returns:
            pd.DataFrame: A DataFrame containing the capacity and power data.
        """
        connection, cursor = self.connect_to_db()
        query = """
            SELECT unit_power_electrical, unit_number, additional_unit_power_electrical, acceptance_date
            FROM ministry_of_energy
            WHERE wf_id = ?
            ORDER BY acceptance_date ASC
        """
        cursor.execute(query, (wf_id,))
        result = cursor.fetchall()
        cursor.close()
        connection.close()
        if result:
            columns = [column[0] for column in cursor.description]
            moe_data = pd.DataFrame(result, columns=columns)
            return moe_data
        else:
            print(f"No capacity and power data found for wind farm with ID {wf_id}")
            return None

    def get_wf_turbines_data(self, wf_id):
        """
        Fetches turbine data from the 'wf_turbines' table.

        Args:
            wf_id (int): The ID of the wind farm site.

        Returns:
            pd.DataFrame: A DataFrame containing the turbine data.
        """
        connection, cursor = self.connect_to_db()
        query = """
            SELECT installed_power, turbine_brand, turbine_model, turbine_power, start_date_of_operation, turbine_number
            FROM wf_turbines
            WHERE wf_id = ?
        """
        cursor.execute(query, (wf_id,))
        result = cursor.fetchall()
        cursor.close()
        connection.close()
        if result:
            columns = [column[0] for column in cursor.description]
            wf_turbines_data = pd.DataFrame(result, columns=columns)
            return wf_turbines_data
        else:
            print(f"No turbine data found for wind farm with ID {wf_id}")
            return None

    def get_wf_data(self, wf_id):
        """
        Fetches information about a specific wind farm from the 'wf' table.

        Args:
            wf_id (int): The ID of the wind farm site.

        Returns:
            pd.DataFrame: A DataFrame containing the turbine data.
        """
        connection, cursor = self.connect_to_db()
        query = """
            SELECT installed_power_mechanical, installed_power_electrical, capacity_mechanical, capacity_electrical
            FROM wf 
            WHERE wf_id = ?
        """
        cursor.execute(query, (wf_id,))
        result = cursor.fetchone()
        cursor.close()
        connection.close()
        if result:
            columns = [column[0] for column in cursor.description]
            wf_data = pd.DataFrame([result], columns=columns)
            return wf_data
        else:
            print(f"No site found with ID {wf_id}")
            return None

    def get_wind_production_data(self, wf_id, start_date='2017-01-01', end_date='2023-12-31', CF=False, CF_ava=False, frequency='hourly', type='uevm', availability=False, cap_filter=False):
        """
        Fetches production data and (optionally) calculates capacity factors for a specific wind farm.

        Args:
            wf_id (int): The ID of the wind farm site.
            start_date (str or datetime, optional): The starting date or datetime object (default: '2017-01-01').
            end_date (str or datetime, optional): The ending date or datetime object (default: '2023-12-31').
            CF (bool, optional): Whether to calculate and include capacity factors (default: False).
            frequency (str, optional): Resampling frequency ('hourly', 'daily', 'weekly', 'monthly', 'yearly').
                                       Defaults to 'hourly'.
            type (str, optional): The type of the production data to fetch ('uevm' or 'realtime').
            availability (bool, optional): Whether to include availability data (default: False).

        Returns:
            pd.DataFrame: A DataFrame containing production data and optionally capacity factors.
            If no data is found for the wind farm ID and date range or if there is missing capacity data, returns None.
        """

        # Fetch production data
        start_ts_id, end_ts_id = get_timestamps_from_dates(start_date, end_date)
        connection, cursor = self.connect_to_db()

        if frequency != 'hourly':
            if availability:
                print("Warning: Availability data cannot be averaged over time periods. It will be dropped.")
                availability = False

        if type == 'uevm':
            if availability:
                query = """
                    SELECT ts_id, wind_uevm as production, wind_eak as availability
                    FROM productions
                    WHERE wf_id = ? and ts_id between ? and ? and wind_uevm is not null
                """
            else:
                query = """
                    SELECT ts_id, wind_uevm as production
                    FROM productions
                    WHERE wf_id = ? and ts_id between ? and ? and wind_uevm is not null
                """
        elif type == 'realtime':
            if availability:
                query = """
                    SELECT ts_id, wind_rt as production, wind_eak as availability
                    FROM productions
                    WHERE wf_id = ? and ts_id between ? and ? and wind_rt is not null
                """
            else:
                query = """
                    SELECT ts_id, wind_rt as production
                    FROM productions
                    WHERE wf_id = ? and ts_id between ? and ? and wind_rt is not null
                """
        else:
            raise ValueError("Invalid production data type. Choose 'uevm' or 'realtime'.")

        cursor.execute(query, (wf_id, start_ts_id, end_ts_id))
        result = cursor.fetchall()
        cursor.close()
        connection.close()
        if not result:  # Handle case where no production data is found
            print(f"No production data found for wind farm with ID {wf_id}")
            return None

        columns = [column[0] for column in cursor.description]
        production_data = pd.DataFrame(result, columns=columns)
        production_data['ts_id'] = production_data['ts_id'].apply(ts_id_to_datetime)
        production_data = production_data.rename(columns={'ts_id': 'timestamp'})
        production_data['production'] = production_data['production'].astype(float)

        if cap_filter:
            moe_data = self.get_moe_data(wf_id)
            moe_data['acceptance_date'] = pd.to_datetime(moe_data['acceptance_date'])
            moe_data = moe_data.groupby('acceptance_date')['additional_unit_power_electrical'].sum().reset_index()
            moe_data.loc[-1] = [moe_data['acceptance_date'].min() - pd.DateOffset(days=1), 0]
            moe_data.loc[-2] = [pd.Timestamp(end_date), moe_data['additional_unit_power_electrical'].sum()]
            moe_data = moe_data.sort_values(by="acceptance_date")
            moe_data['cumulative_capacity'] = moe_data['additional_unit_power_electrical'].cumsum()

            # Merge moe_data into production_data based on timestamp
            moe_data['timestamp'] = moe_data['acceptance_date']  # Rename column for merging
            production_data = pd.merge_asof(production_data, moe_data[['timestamp', 'cumulative_capacity']], on='timestamp', direction='backward')

            # Clip production based on the cumulative capacity at each timestamp
            production_data['production'] = production_data.apply(lambda row: min(row['production'], row['cumulative_capacity']), axis=1)
            production_data['production'] = production_data['production'].clip(lower=0)  # Ensure production is not negative


        if CF:
            # Fetch capacity data
            moe_data = self.get_moe_data(wf_id)

            if moe_data is None:  # Handle case where capacity data is missing
                print(f"Capacity data not found for wind farm with ID {wf_id}")
                return None

            # Convert 'acceptance_date' to datetime
            moe_data['acceptance_date'] = pd.to_datetime(moe_data['acceptance_date'])

            # Ensure production values are numeric
            production_data['production'] = pd.to_numeric(production_data['production'], errors='coerce')
            production_data = production_data.fillna(0)

            # Calculate capacity factor for each timestamp (hourly)
            capacity_factor = []
            for _, row in production_data.iterrows():
                # Get installed capacity for the date
                installed_capacity = moe_data.loc[
                    (moe_data['acceptance_date'] <= row['timestamp']),
                    'additional_unit_power_electrical'
                ].sum()

                # Calculate capacity factor (hourly)
                if installed_capacity > 0:
                    cf = row['production'] / installed_capacity
                else:
                    cf = np.nan  # No capacity yet installed
                capacity_factor.append(cf)

            # Combine results into a DataFrame
            production_data['capacity_factor'] = capacity_factor
            production_data.loc[production_data['capacity_factor'] > 1, 'capacity_factor'] = np.nan

        production_data.set_index('timestamp', inplace=True)
        if frequency == 'daily':
            if CF:
                production_data = production_data.resample('D').agg({
                    'production': 'sum',
                    'capacity_factor': 'mean'
                })
            else:
                production_data = production_data.resample('D').agg({
                    'production': 'sum'
                })
        elif frequency == 'weekly':
            if CF:
                production_data = production_data.resample('W').agg({
                    'production': 'sum',
                    'capacity_factor': 'mean'
                })
            else:
                production_data = production_data.resample('W').agg({
                    'production': 'sum'
                })
        elif frequency == 'monthly':
            if CF:
                production_data = production_data.resample('ME').agg({
                    'production': 'sum',
                    'capacity_factor': 'mean'
                })
            else:
                production_data = production_data.resample('ME').agg({
                    'production': 'sum'
                })
        elif frequency == 'yearly':
            if CF:
                production_data = production_data.resample('Y').agg({
                    'production': 'sum',
                    'capacity_factor': 'mean'
                })
            else:
                production_data = production_data.resample('Y').agg({
                    'production': 'sum'
                })

        if CF_ava:
            if 'availability' in production_data.columns:
                production_data['capacity_factor_ava'] = production_data.apply(
                    lambda row: row['production'] / row['availability'] if row['availability'] else np.nan, axis=1
                )
                production_data.loc[production_data['capacity_factor_ava'] > 1, 'capacity_factor_ava'] = np.nan
            else:
                raise ValueError("Availability data must be included to calculate capacity factor with respect to availability.")

        # Convert to hours in case the values become NaT
        production_data['timestamp'] = production_data.index
        production_data = production_data.reset_index(drop=True)

        # reorder columns
        if CF:
            if availability:
                if CF_ava:
                    production_data = production_data[['timestamp', 'production', 'capacity_factor', 'availability', 'capacity_factor_ava']]
                else:
                    production_data = production_data[['timestamp', 'production', 'capacity_factor', 'availability']]
            else:
                production_data = production_data[['timestamp', 'production', 'capacity_factor']]
        else:
            if availability:
                if CF_ava:
                    production_data = production_data[['timestamp', 'production', 'availability', 'capacity_factor_ava']]
                else:
                    production_data = production_data[['timestamp', 'production', 'availability']]
            else:
                production_data = production_data[['timestamp', 'production']]

        return production_data


    def get_turbine_locations(self, wf_id, centroid=False):
        """Retrieves either centroid or individual turbine locations.

        Args:
            wf_id: The ID of the wind farm.
            centroid (bool, optional): If True, returns the centroid.
                                       If False, returns all individual turbine locations.
                                       Defaults to True.

        Returns:
            If centroid=True: A tuple (latitude, longitude) of the centroid.
            If centroid=False: A list of tuples, each representing a turbine's (latitude, longitude).

        Raises:
            KeyError: If no turbine coordinates are found for the given wf_id.
        """

        conn = sqlite3.connect(self.db_path)
        conn.enable_load_extension(True)
        conn.load_extension('mod_spatialite')
        #conn.load_extension('/home/kutlay/.conda/envs/WFP/lib/mod_spatialite')
        #conn.load_extension('/home/wfd/miniforge3/envs/WFP5/lib/mod_spatialite')

        proj_db_path = '/home/wheatley/.conda/envs/WFP2/share/proj/proj.db'
        conn.execute(f"SELECT PROJ_SetDatabasePath('{proj_db_path}');")

        if centroid:
            query = """
                SELECT  ST_Y(ST_Transform(ST_Centroid(ST_Collect(wtc.geom)), 4326)) AS lat,
                        ST_X(ST_Transform(ST_Centroid(ST_Collect(wtc.geom)), 4326)) AS long
                FROM wf_turbine_coordinates AS wtc
                WHERE wtc.geom IS NOT NULL
                      AND wf_id = ?;
            """
            result = conn.execute(query, (wf_id,)).fetchone()
        else:
            query = """
                SELECT  turbine_index,
                        ST_Y(ST_Transform(wtc.geom, 4326)) AS lat,
                        ST_X(ST_Transform(wtc.geom, 4326)) AS long
                FROM wf_turbine_coordinates AS wtc
                WHERE wtc.geom IS NOT NULL
                      AND wf_id = ?
                      order by turbine_index;
            """
            cursor = conn.cursor()
            cursor.execute(query, (wf_id,))
            result = cursor.fetchall()

        conn.close()

        if not result:  # Check if any results were found
            raise KeyError(f"Wind farm with ID {wf_id} not found or has no turbine coordinates.")

        return result

    def find_elevations(self, wf_id):
        """
        Find the elevation of the turbines in the wind farm

        Parameters:
        wf_id (int): Wind farm id

        Returns:
        list: List of tuples containing the latitude, longitude and elevation of the turbines
        """

        # Get the turbine locations
        locations = self.get_turbine_locations(wf_id, centroid=False)

        # List to store the latitude, longitude and elevation of the turbines
        turbine_elevations = []

        # Iterate over the turbine locations
        for index, lat, long in locations:
            latitude = float(lat)
            longitude = float(long)

            tiff_file = f"/home/wheatley/WFD/DEM/Copernicus_DSM_10_N{int(latitude):02d}_00_E0{int(longitude):02d}_00_DEM.tif"

            # Find the elevation of the turbine
            with rasterio.open(tiff_file) as dataset:
                row, col = dataset.index(longitude, latitude)
                elevation = dataset.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]

                if 0 <= row < dataset.height and 0 <= col < dataset.width:
                    elevation = dataset.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]
                else:
                    elevation = None

            # Append the latitude, longitude and elevation to the list
            turbine_elevations.append((index, latitude, longitude, elevation))

        return turbine_elevations


    def find_closest_four_era5_location(self, wf_id, distance=False):
        """
        Finds the closest 4 ERA5 data locations for the given wind farm ID.

        Args:
        wf_id (int): The ID of the wind farm.

        Returns:
            tuple: A tuple containing (latitude, longitude) of the closest four ERA5 location.
        """
        latitude, longitude = self.get_turbine_locations(wf_id, centroid=True)
        connection, cursor = self.connect_to_db()
        query = """
            SELECT latitude, longitude, 
                   MIN( (latitude - ?) * (latitude - ?) + (longitude - ?) * (longitude - ?) ) AS distance
            FROM era5_raw_data
            GROUP BY latitude, longitude
            ORDER BY distance ASC
            LIMIT 4
        """
        if latitude is None or longitude is None:
            raise KeyError(f"Wind farm with ID {wf_id} not found or has no turbine coordinates.")
        cursor.execute(query, (latitude, latitude, longitude, longitude))
        result = cursor.fetchall()
        cursor.close()
        connection.close()

        if result:
            if distance:
                return result
            else:
                return [location[:2] for location in result]
        else:
            return None

    def get_era5_data(self, wf_id, start_date='2017-01-01', end_date='2023-12-31', grid_number=0, variables_to_plot=None):
        """
        Fetches selected ERA5 data for a specific wind farm and date range, finding the closest data location.

        Args:
            wf_id (int): The ID of the wind farm.
            start_date (datetime.datetime): The starting datetime object.
            end_date (datetime.datetime): The ending datetime object.
            grid_number (int): The number of the closest grid location to use (0-3).
            variables_to_plot (list, optional): List of ERA5 variables to fetch (e.g., ['temp', 'ws100']).
                                                 Defaults to all available variables.

        Returns:
            pd.DataFrame: A DataFrame containing the selected ERA5 data for the closest location.
        """

        if variables_to_plot is None:
            variables_to_plot = ["temperature",
                                "pressure",
                                "dew_point",
                                "surface_sensible_heat_flux",
                                "u100",
                                "v100",
                                "u10",
                                "v10",
                                "mean_sea_level_pressure",
                                "sea_surface_temperature",
                                "u10n",
                                "v10n",
                                "fg10",
                                "i10fg",
                                "surface_latent_heat_flux",
                                "boundary_layer_dissipation",
                                "boundary_layer_height",
                                "charnock",
                                "forecast_surface_roughness",
                                "friction_velocity",
                                "land_sea_mask",
                                "relative_humidity",
                                "air_density"]

        # Find closest ERA5 data location
        closest_location = self.find_closest_four_era5_location(wf_id)[grid_number]

        if not closest_location:
            print(f"No nearby ERA5 data location found for wind farm {wf_id}.")
            return None
        era5_latitude, era5_longitude = closest_location

        # Convert datetime inputs to timestamp IDs
        start_ts_id, end_ts_id = get_timestamps_from_dates(start_date, end_date)

        # Determine which variables are in which table

        raw_variables = [var for var in variables_to_plot if var in ["temperature",
                                                                    "pressure",
                                                                    "dew_point",
                                                                    "surface_sensible_heat_flux",
                                                                    "u100",
                                                                    "v100",
                                                                    "u10",
                                                                    "v10",
                                                                    "mean_sea_level_pressure",
                                                                    "sea_surface_temperature",
                                                                    "u10n",
                                                                    "v10n",
                                                                    "fg10",
                                                                    "i10fg",
                                                                    "surface_latent_heat_flux",
                                                                    "boundary_layer_dissipation",
                                                                    "boundary_layer_height",
                                                                    "charnock",
                                                                    "forecast_surface_roughness",
                                                                    "friction_velocity",
                                                                    "land_sea_mask"]]


        calculated_variables = [var for var in variables_to_plot if var in ['ws100', 'wd100', 'ws10', 'wd10', "relative_humidity", "air_density"]]

        # Construct queries based on selected variables
        raw_data_query = None
        if raw_variables:
            raw_data_query = f"""
                SELECT ts_id, {', '.join(f'`{var}`' for var in raw_variables)}
                FROM era5_raw_data
                WHERE latitude = ? AND longitude = ? AND ts_id BETWEEN ? AND ?
            """

        calculated_data_query = None
        if calculated_variables:
            calculated_data_query = f"""
                SELECT ts_id, {', '.join(f'`{var}`' for var in calculated_variables)}
                FROM era5_calculated_data
                WHERE latitude = ? AND longitude = ? AND ts_id BETWEEN ? AND ?
            """

        connection, cursor = self.connect_to_db()

        dataframes = []
        if raw_data_query:
            cursor.execute(raw_data_query, (era5_latitude, era5_longitude, start_ts_id, end_ts_id))
            raw_data = pd.DataFrame(cursor.fetchall(), columns=[column[0] for column in cursor.description])
            dataframes.append(raw_data)

        if calculated_data_query:
            cursor.execute(calculated_data_query, (era5_latitude, era5_longitude, start_ts_id, end_ts_id))
            calculated_data = pd.DataFrame(cursor.fetchall(), columns=[column[0] for column in cursor.description])
            dataframes.append(calculated_data)

        cursor.close()
        connection.close()
        if dataframes:
            merged_data = pd.concat(dataframes, axis=1)  # Combine DataFrames if multiple tables were queried
            merged_data = merged_data.loc[:,~merged_data.columns.duplicated()]  # Drop duplicate ts_id
            merged_data['ts_id'] = merged_data['ts_id'].apply(ts_id_to_datetime)  # Convert to datetime
            merged_data = merged_data.rename(columns={'ts_id': 'timestamp'})
            return merged_data
        else:
            print(f"No ERA5 data found for closest location to wind farm {wf_id} between {start_date} and {end_date}")
            return None

    def check_solar(self, wf_id, start_date, end_date):
        """
        Check if the wind farm has a solar power plant by checking the production table if there is any solar production data that is not null and above 0

        Args:
            wf_id (int): The ID of the wind farm.

        Returns:
            bool: True if the wind farm has a solar power plant, False otherwise.
        """
        connection, cursor = self.connect_to_db()
        query = """
            SELECT COUNT(*)
            FROM productions
            WHERE wf_id = ?
              AND ts_id BETWEEN ? AND ?
              AND (
                (solar_uevm IS NOT NULL AND solar_uevm > 0)
                OR
                (solar_rt IS NOT NULL AND solar_rt > 0)
              )
        """
        start_ts_id, end_ts_id = get_timestamps_from_dates(start_date, end_date)
        cursor.execute(query, (wf_id, start_ts_id, end_ts_id))
        result = cursor.fetchone()
        cursor.close()
        connection.close()
        return result[0] > 0