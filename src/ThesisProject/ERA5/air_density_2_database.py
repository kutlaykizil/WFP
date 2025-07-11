import pandas as pd
import numpy as np
import sqlalchemy
import logging
from numpy import exp
import os
import time

# --- Configuration ---
DATABASE_PATH = "/home/kutlay/WFD/wfd.db"  # Update with your database path
CHUNK_SIZE = 5000000  # Adjust based on your system (start with 100,000 and tune)
RETRIES = 3  # Number of retries if the database is locked
RETRY_DELAY = 5  # Seconds to wait between retries

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Constants for Air Density Calculation ---
Alpha = 1.00062
Beta = 3.14 * 10**-8
Gamma = 5.6 * 10**-7
C1 = 1.2378847 * 10**-5
C2 = -1.9121316 * 10**-2
C3 = 33.93711047
C4 = -6.3431645 * 10**3
ao = 1.58123 * 10**-6
a1 = -2.9331 * 10**-8
a2 = 1.1043 * 10**-10
bo = 5.707 * 10**-6
b1 = -2.051 * 10**-8
co = 1.9898 * 10**-4
c1 = -2.376 * 10**-6
d = 1.83 * 10**-11
e = -0.765 * 10**-8
R = 8.314472
Ma = 28.96546 * 10**-3
Mv = 18.01525 * 10**-3

# --- Functions for Air Density Calculation ---
def tK(T):
    """Converts temperature from Celsius to Kelvin."""
    return 273.15 + T

def xv(t, P, RH):
    """Calculates the mole fraction of water vapor."""
    RH = RH / 100 if RH > 1 else RH
    return (
        RH
        * (Alpha + Beta * P + Gamma * t**2)
        * exp(C1 * tK(t) ** 2 + C2 * tK(t) + C3 + C4 / tK(t))
        / P
    )

def Z(t, P, RH):
    """Calculates the compressibility factor."""
    x_v = xv(t, P, RH)
    return (
        1
        - (P / tK(t))
        * (
            ao
            + a1 * t
            + a2 * t**2
            + (bo + b1 * t) * x_v
            + (co + c1 * t) * x_v**2
        )
        + (P**2 / tK(t) ** 2) * (d + e * x_v**2)
    )

def calculate_rho(t, P, RH):
    """Calculates the air density."""
    x_v = xv(t, P, RH)
    return ((P * Ma) / (Z(t, P, RH) * R * tK(t))) * (1 - x_v * (1 - (Mv / Ma)))

def calculate_relative_humidity(T_kelvin, dew_point_kelvin):
    """
    Calculates relative humidity from temperature and dew point temperature.

    Parameters:
    T_kelvin (float): Temperature in Kelvin.
    dew_point_kelvin (float): Dew point temperature in Kelvin.

    Returns:
    float: Relative humidity as a percentage (0-100).
    """
    # Constants for Magnus formula
    a = 17.625
    b = 243.04

    # Convert temperatures from Kelvin to Celsius
    T_celsius = T_kelvin - 273.15
    dew_point_celsius = dew_point_kelvin - 273.15

    # Calculate saturation vapor pressure (es) and actual vapor pressure (ea) using Magnus formula
    es = 6.112 * np.exp(a * T_celsius / (b + T_celsius))
    ea = 6.112 * np.exp(a * dew_point_celsius / (b + dew_point_celsius))

    # Calculate relative humidity
    RH = (ea / es) * 100

    return RH

def update_database_with_retries(chunk, engine, retries=RETRIES, retry_delay=RETRY_DELAY):
    """
    Updates the database with the calculated air density, handling potential 'database is locked' errors with retries.
    """
    for i in range(retries):
        try:
            with engine.begin() as connection:
                # Get the corresponding data from era5_calculated_data
                calculated_data_chunk = pd.read_sql_query(
                    f"SELECT * FROM era5_calculated_data WHERE ts_id IN {tuple(chunk['ts_id'].tolist())} AND latitude IN {tuple(chunk['latitude'].tolist())} AND longitude IN {tuple(chunk['longitude'].tolist())}",
                    con=connection,
                )

                # Merge the air_density data based on ts_id, latitude, and longitude
                merged_chunk = pd.merge(
                    calculated_data_chunk,
                    chunk,
                    on=["ts_id", "latitude", "longitude"],
                    how="left",
                )

                # Append the merged chunk to a temporary table
                merged_chunk.to_sql(
                    "temp_calculated_data",
                    con=connection,
                    if_exists="append",
                    index=False,
                )
            return  # Success, exit the function
        except sqlalchemy.exc.OperationalError as e:
            if "database is locked" in str(e):
                logging.warning(
                    f"Database is locked, retrying in {retry_delay} seconds (attempt {i+1}/{retries})"
                )
                time.sleep(retry_delay)
            else:
                raise  # Re-raise if it's a different OperationalError

    raise Exception(f"Failed to update database after {retries} retries")

def main():
    """
    Calculates air density from data in the database and adds it as a new column.
    Processes data in chunks to avoid memory issues.
    Handles 'database is locked' errors with retries.
    """
    try:
        engine = sqlalchemy.create_engine("sqlite:///" + DATABASE_PATH)
        logging.info("Successfully connected to the database.")

        # Enable WAL mode (optional - uncomment if you want to try it)
        with engine.connect() as connection:
            connection.execute(sqlalchemy.text("PRAGMA journal_mode=WAL"))

        logging.info("Updating era5_calculated_data table...")
        with engine.begin() as connection:
            # Create the air_density column if it doesn't exist
            # connection.execute(
            #     sqlalchemy.text(
            #         "ALTER TABLE era5_calculated_data ADD COLUMN air_density REAL"
            #     )
            # )

            # Read and process data in chunks
            for chunk in pd.read_sql_table(
                "era5_raw_data", con=engine, chunksize=CHUNK_SIZE
            ):
                logging.info(f"Processing chunk of size {len(chunk)}")

                chunk["relative_humidity"] = calculate_relative_humidity(
                    chunk["temperature"], chunk["dew_point"]
                )

                # Convert temperature from Kelvin to Celsius
                chunk["temperature_celsius"] = chunk["temperature"] - 273.15

                # Calculate air density
                chunk["air_density"] = chunk.apply(
                    lambda row: calculate_rho(
                        row["temperature_celsius"],
                        row["pressure"],
                        row["relative_humidity"],
                    ),
                    axis=1,
                )

                # Select only necessary columns for merging
                chunk = chunk[["ts_id", "latitude", "longitude", "air_density"]]

                # Update database with retries
                update_database_with_retries(chunk, engine)

        with engine.begin() as connection:
            # Replace the original table with the temporary table
            connection.execute(sqlalchemy.text("DROP TABLE era5_calculated_data"))
            connection.execute(
                sqlalchemy.text(
                    "ALTER TABLE temp_calculated_data RENAME TO era5_calculated_data"
                )
            )

            # Add an index to the new table (optional, but can improve performance)
            connection.execute(
                sqlalchemy.text(
                    "CREATE INDEX IF NOT EXISTS idx_ts_lat_lon ON era5_calculated_data (ts_id, latitude, longitude)"
                )
            )

        logging.info("Table updated successfully.")

    except Exception as e:
        logging.error(f"Error updating the database: {e}")
    finally:
        engine.dispose()

if __name__ == "__main__":
    main()