import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import folium
from folium.plugins import MarkerCluster
import pandas as pd
import numpy as np

def plot_production(analyzer, wf_id, start_date, end_date, type='uevm'):
    """
    Plots production data with timestamps using output from Analyzer.
    """
    if type == 'uevm':
        production_data = analyzer.get_wind_production_data(wf_id, start_date, end_date)
    elif type == 'realtime':
        production_data = analyzer.get_wind_production_data(wf_id, start_date, end_date, type='realtime')

    if production_data is not None:
        production_data['production'] = production_data['production'].astype(float)
        plt.figure(figsize=(16, 9), dpi=120)
        plt.plot(production_data['timestamp'], production_data['production'])
        plt.xlabel('Timestamp')
        plt.ylabel('Production (MWh)')
        plt.title('Production Data' + f' for Wind Farm id={wf_id}')
        plt.show()

def plot_monthly_production(analyzer, wf_id, start_date, end_date):
    """
    Plots monthly production data using output from Analyzer,
    checking if the data range spans a full year.
    """
    production_data = analyzer.get_wind_production_data(wf_id, start_date, end_date)

    if production_data is not None:
        production_data['production'] = production_data['production'].astype(float)

        # Check if the data range spans a full year
        start_date = production_data['timestamp'].min()
        end_date = production_data['timestamp'].max()
        length = ((end_date - start_date).days)
        # Check if the data range spans a full year
        if length > 365:
            # Extract month and aggregate data
            production_data['month'] = production_data['timestamp'].dt.month
            monthly_production = production_data.groupby('month').sum(numeric_only=True)

            # Create the plot
            plt.figure(figsize=(16, 9), dpi=120)
            plt.bar(monthly_production.index, monthly_production['production'])
            plt.xlabel('Month')
            plt.ylabel('Production (MWh)')
            plt.title('Monthly Production Data' + f' for Wind Farm id={wf_id}')
            plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            plt.show()
        else:
            print("Data range does not span a full year. Monthly plot cannot be generated.")

def print_capacity_info(analyzer, wf_id):
    """
    Prints capacity and power information using output from Analyzer.
    """
    moe = analyzer.get_moe_data(wf_id)
    wf_turbines = analyzer.get_wf_turbines_data(wf_id)
    wf = analyzer.get_wf_data(wf_id)

    if moe is not None and wf_turbines is not None and wf is not None:
        print("The wind farm has the turbines with brand and models of: ")
        print(wf_turbines[['turbine_brand', 'turbine_model']].drop_duplicates())
        print("\n")
        print("The wind farm holds the following production license capacity (MWe):")
        print("From epdk: " + str(wf['capacity_electrical']))
        print("From moe: " + str(moe['additional_unit_power_electrical'].sum()))
        print("\n")
        print("The wind farm has the following mechanical installed power (MWm):")
        print("From epdk: " + str(wf['installed_power_mechanical']))
        print("From turbines: " + str(wf_turbines['installed_power'].sum()))
        print("\n")
        print("The wind farm has the following changes throughout the years:")
        # Print the new installed power and the aggregated installed power next to it from moe data (moe has the column named acceptance_date for the year data)
        # reorder the columns by the acceptance_date
        moe = moe.sort_values(by='acceptance_date')
        moe['agg_installed_power'] = moe['additional_unit_power_electrical'].cumsum()
        print(moe[['acceptance_date', 'additional_unit_power_electrical', 'agg_installed_power']])

def print_farm_info(analyzer, wf_id):
    """
    Prints formatted site information using output from Analyzer.
    """
    farm_info = analyzer.get_farm_info(wf_id)
    if farm_info:
        for key, value in farm_info.items():
            print(f"\033[1m{key}:\033[0m {value}")

def plot_turbine_centroids(analyzer, wf_id):
    """
    Plots turbine centroids on a map using Folium.
    """
    locations = analyzer.find_elevations(wf_id)
    centroid = analyzer.get_turbine_locations(wf_id, centroid=True)

    elevation_values = [{'latitude': loc[1], 'longitude': loc[2], 'elevation': loc[3]} for loc in locations]

    for i in range(len(locations)):
        location = locations[i]
        rounded_elevation = (round(location[3], 2)).astype(str) + 'm'
        locations[i] = (location[0], location[1], location[2], rounded_elevation)

    # Create a map centered at the mean latitude and longitude of the turbines
    map = folium.Map(location=centroid)

    # Create a MarkerCluster object
    #marker_cluster = MarkerCluster().add_to(map)

    # Add a tile layer to the map
    # Select here: https://leaflet-extras.github.io/leaflet-providers/preview/
    #folium.TileLayer('https://tiles.stadiamaps.com/tiles/alidade_satellite/{z}/{x}/{y}{r}.{ext}', attr='Stadia').add_to(map) # Satellite
    folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr='Esri').add_to(map) # Satellite

    # add topology layer
    #folium.TileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', attr='OpenTopoMap').add_to(map) # Topology

    # Add markers to the map
    #for i in range(len(locations)): # AS CLUSTER
    #    folium.Marker([locations[i][0], locations[i][1]]).add_to(marker_cluster)

    for i in range(len(locations)): # AS MARKER
        folium.Marker([locations[i][1], locations[i][2]]).add_to(map)
        folium.Marker([locations[i][1], locations[i][2]], popup="Elevation: {}".format(locations[i][3])).add_to(map)
        #folium.Marker([locations[i][1], locations[i][2]], icon=folium.DivIcon(html=f'<div style="font-size: 32; color: blue">{locations[i][3]}</div>')).add_to(map)


    map.fit_bounds([[locations[i][1], locations[i][2]] for i in range(len(locations))], padding=(12, 12))

    # remove_close_locations function
    def remove_close_locations(locations, distance):
        """
        Removes close locations from a list of locations based on a given distance.
        """
        # Create a new list to store the filtered locations
        filtered_locations = []

        # Loop through each location in the list
        for i in range(len(locations)):
            # Get the current location
            current_location = locations[i]

            # Flag to check if the location is close to another location
            close = False

            # Loop through each location again to compare with the current location
            for j in range(len(locations)):
                # Skip the current location
                if i == j:
                    continue

                # Get the other location
                other_location = locations[j]

                # Calculate the distance between the two locations
                dist = np.sqrt((current_location[1] - other_location[1]) ** 2 + (current_location[2] - other_location[2]) ** 2)

                # Check if the distance is less than the given distance
                if dist < distance:
                    close = True
                    break

            # Add the location to the filtered list if it is not close to any other location
            if not close:
                filtered_locations.append(current_location)

        return filtered_locations


    # before adding the elevation markers, remove the close locations so it is not cluttered.
    locations = remove_close_locations(locations, 0.001)


    for i in range(len(locations)):
        folium.Marker([locations[i][1], locations[i][2]], icon=folium.DivIcon(html=
                                                                              f'<div style="font-size: 18px;'
                                                                              f'color: white;'
                                                                              f'font-family: Arial;'
                                                                              f'text-shadow: 1px 1px 0 rgba(0, 0, 0, 0.8);'
                                                                              f'">{locations[i][3]}</div>')).add_to(map)

    # zoom until the markers are visible
    #map.fit_bounds([[locations[i][1], locations[i][2]] for i in range(len(locations))])
    # zoom out a bit

    return map, elevation_values

def plot_era5_data(analyzer, wf_id, start_date, end_date, grid_number=0, variables_to_plot=None):
    """Plots selected ERA5 data variables on individual plots."""

    # Default variables to plot if not specified
    if variables_to_plot is None:
        variables_to_plot = ['temp', 'pressure', 'dew', 'sensible_heat', 'ws100', 'wd100', 'ws10', 'wd10']

    # Titles and labels for each variable (same as before)
    variable_info = {
        'temp': ('Temperature (Kelvin)', 'Timestamp', 'Temperature (Kelvin)'),
        'pressure': ('Pressure (Pa)', 'Timestamp', 'Pressure (Pa)'),
        'dew': ('Dew Point Temperature (Kelvin)', 'Timestamp', 'Dew Point Temperature (Kelvin)'),
        'sensible_heat': ('Sensible Heat Flux (W/m^2)', 'Timestamp', 'Sensible Heat Flux (W/m^2)'),
        'ws100': ('Wind Speed 100m (m/s)', 'Timestamp', 'Wind Speed (m/s)'),
        'wd100': ('Wind Direction 100m (Degrees)', 'Timestamp', 'Wind Direction (Degrees)'),
        'ws10': ('Wind Speed 10m (m/s)', 'Timestamp', 'Wind Speed (m/s)'),
        'wd10': ('Wind Direction 10m (Degrees)', 'Timestamp', 'Wind Direction (Degrees)')
    }

    # Fetch only the selected variables from the database
    era5_data = analyzer.get_era5_data(wf_id, start_date, end_date, grid_number, variables_to_plot)

    if era5_data is not None:
        # Loop through each variable to be plotted
        for var in variables_to_plot:
            if var in era5_data.columns:
                title, xlabel, ylabel = variable_info[var]  # Get plot info
                plt.figure(figsize=(10, 3))  # Create a new figure for each plot
                plt.plot(era5_data['timestamp'], era5_data[var], label=title)
                plt.title(title)
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.legend()
                plt.show()
    else:
        print('No ERA5 data found for the given dates.')

def plot_production_capacity_CF(analyzer, wf_id, start_date='2017-01-01', end_date='2023-12-31', type='uevm', cap_filter=False):
    """
    Fetches production and capacity factor data, and creates a plot.

    Args:
        wf_id (int): The ID of the wind farm site.
        start_date (str or datetime, optional): The starting date or datetime object (default: '2017-01-01').
        end_date (str or datetime, optional): The ending date or datetime object (default: '2023-12-31').

    Returns:
        None: This function does not return a value, it only generates the plot.
    """

    if type == 'uevm':
        if cap_filter:
            production_data = analyzer.get_wind_production_data(wf_id, start_date, end_date, cap_filter=True)
            capacity_factor_data = analyzer.get_wind_production_data(wf_id, start_date, end_date, CF=True, cap_filter=True, frequency='monthly', type='uevm')
        elif cap_filter == False:
            production_data = analyzer.get_wind_production_data(wf_id, start_date, end_date, cap_filter=False)
            capacity_factor_data = analyzer.get_wind_production_data(wf_id, start_date, end_date, CF=True, frequency='monthly', type='uevm')
    elif type == 'realtime':
        if cap_filter:
            production_data = analyzer.get_wind_production_data(wf_id, start_date, end_date, type='realtime', cap_filter=True)
            capacity_factor_data = analyzer.get_wind_production_data(wf_id, start_date, end_date, CF=True, cap_filter=True, frequency='monthly', type='realtime')
        else:
            production_data = analyzer.get_wind_production_data(wf_id, start_date, end_date, type='realtime')
            capacity_factor_data = analyzer.get_wind_production_data(wf_id, start_date, end_date, CF=True, frequency='monthly', type='realtime')

    moe_data = analyzer.get_moe_data(wf_id)


    # Check if all dataframes are not None
    if production_data is not None and capacity_factor_data is not None and moe_data is not None:

        # Convert acceptance_date to datetime if it's not already
        moe_data['acceptance_date'] = pd.to_datetime(moe_data['acceptance_date'])

        # Aggregate 'additional_unit_power_electrical' by 'acceptance_date'
        moe_data = moe_data.groupby('acceptance_date')['additional_unit_power_electrical'].sum().reset_index()

        # add start and end date rows
        #moe_data.loc[-1] = [moe_data['acceptance_date'].min() - pd.DateOffset(years=1), 0]
        moe_data.loc[-1] = [moe_data['acceptance_date'].min() - pd.DateOffset(days=1), 0]
        moe_data.loc[-2] = [pd.Timestamp(end_date), moe_data['additional_unit_power_electrical'].sum()]
        moe_data = moe_data.sort_values(by="acceptance_date")

        # Set up the plot with three y-axes
        fig, ax1 = plt.subplots(figsize=(30, 10), dpi=100)

        # Plot production (left y-axis)
        ax1.plot(production_data['timestamp'], production_data['production'], color='blue', label='Production (MWh)')
        ax1.set_xlabel('Timestamp')
        ax1.set_ylabel('Production (MWh)', color='blue')
        ax1.set_ylim(production_data['production'].min(), production_data['production'].max())

        # Plot capacity factor (right y-axis)
        ax2 = ax1.twinx()
        ax2.plot(capacity_factor_data['timestamp'], capacity_factor_data['capacity_factor'], color='red', label='Capacity Factor (Yearly Average)')
        ax2.set_ylabel('Capacity Factor', color='red')
        ax2.set_ylim(0, 1)
        ax2.yaxis.set_major_locator(MultipleLocator(0.1))

        # Calculate and plot linear trend line for capacity factor
        z = np.polyfit(capacity_factor_data['timestamp'].astype(np.int64) / 1e9, capacity_factor_data['capacity_factor'], 1)
        p = np.poly1d(z)
        ax2.plot(capacity_factor_data['timestamp'], p(capacity_factor_data['timestamp'].astype(np.int64) / 1e9), linestyle='--', color='orange', label='Capacity Factor Trendline', linewidth=4)
        # add the decline rate to the plot as a percentage
        decline_rate = (p(capacity_factor_data['timestamp'].astype(np.int64).max() / 1e9) - p(capacity_factor_data['timestamp'].astype(np.int64).min() / 1e9)) / p(capacity_factor_data['timestamp'].astype(np.int64).min() / 1e9) * 100
        ax2.text(capacity_factor_data['timestamp'].max(), p(capacity_factor_data['timestamp'].astype(np.int64).max() / 1e9), f'Decline Rate: {decline_rate:.2f}%', color='orange', fontsize=12, verticalalignment='center')


        # Plot installed capacity (right y-axis)
        ax3 = ax1.twinx()  # Create a third y-axis
        ax3.plot(moe_data['acceptance_date'], moe_data['additional_unit_power_electrical'].cumsum(),
                 color='green', label='Installed Capacity (MWe)', linestyle='-', drawstyle='steps-post')
        ax3.set_ylabel('Installed Capacity (MWe)', color='green')
        ax3.spines["right"].set_position(("axes", 1.1))  # Move the third y-axis to the right
        ax3.spines["right"].set_visible(True)
        # Adjust y-axis limits as the same as the production data
        ax3.set_ylim(production_data['production'].min(), production_data['production'].max())


        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax2.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')
        plt.title(f'Production, Capacity Factor, and Installed Capacity for Wind Farm {wf_id}')
        plt.xlim(pd.Timestamp(start_date), pd.Timestamp(end_date))

        plt.show()
    else:
        print('No data found for the given dates.')