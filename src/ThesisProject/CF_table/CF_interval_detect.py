import os
from fontTools.misc.cython import returns
from src.DatabaseGetInfo import DatabaseAnalyzer
from src.DatabaseGetInfo import DatabasePlotter as dbPlot
from bokeh.plotting import figure, show, output_notebook
from selenium import webdriver


#import chromedriver_binary
#options = webdriver.ChromeOptions()
#options.add_argument('--headless')
#driver = webdriver.Chrome(options=options)


# Database path
path = os.path.abspath('/home/wheatley/WFD/wfd.db')
analyzer = DatabaseAnalyzer.WindFarmAnalyzer(path)
plotter = dbPlot

# Enable notebook output
output_notebook()

def detect_and_plot(wf_id, start_date, end_date, plot=True):
    global downwards_trend
    options = webdriver.FirefoxOptions()
    options.add_argument('--headless')
    driver = webdriver.Firefox(options=options)

    wf_id = wf_id
    start_date = start_date
    end_date = end_date

    import pandas as pd
    import numpy as np
    import datetime
    from bokeh.io import export_png
    from bokeh.models import HoverTool, DatetimeTickFormatter, LinearAxis, Range1d
    from bokeh.models import DatetimeTickFormatter
    from bokeh.models import LinearAxis, Range1d, ColumnDataSource, HoverTool, Legend, Label
    from bokeh.io import push_notebook, export_svg
    from bokeh.plotting import figure, show, output_notebook, save, output_file
    from bokeh.models import HoverTool, DatetimeTickFormatter, LinearAxis, Range1d, ColumnDataSource, Legend
    from sklearn.cluster import DBSCAN

    try:
        farm_name = analyzer.get_farm_info(wf_id)['plant_name']
    except TypeError:
        print("No data found for the ID {}".format(wf_id))
        driver.close()
        return None, None

    if plot == True:
        import shutil
        if os.path.exists("plots") == False:
            os.mkdir(f"plots")

        os.chdir("plots")


        if os.path.exists(f"{farm_name}({wf_id})"):
            shutil.rmtree(f"{farm_name}({wf_id})")
        os.mkdir(f"{farm_name}({wf_id})")
        os.chdir(f"{farm_name}({wf_id})")


    productions = analyzer.get_wind_production_data(wf_id, start_date, end_date, cap_filter=False, availability=True, CF=True, CF_ava=True)
    if productions is None:
        print("No production data found for the ID {}".format(wf_id))
        if plot == True:
            os.chdir("../..")
        driver.close()
        return None, None

    moe_data = analyzer.get_moe_data(wf_id)
    moe_data['acceptance_date'] = pd.to_datetime(moe_data['acceptance_date'])
    moe_data = moe_data.groupby('acceptance_date')['additional_unit_power_electrical'].sum().reset_index()
    moe_data.loc[-1] = [moe_data['acceptance_date'].min() - pd.DateOffset(days=1), 0]
    moe_data.loc[-2] = [pd.Timestamp(end_date), moe_data['additional_unit_power_electrical'].sum()]
    moe_data = moe_data.sort_values(by="acceptance_date")
    moe_data['cumulative_capacity'] = moe_data['additional_unit_power_electrical'].cumsum()
    # print(moe_data['cumulative_capacity'])
    # remove the last row of moe_data
    moe_data = moe_data.iloc[:-1]

    # Merge moe_data into production_data based on timestamp
    moe_data['timestamp'] = moe_data['acceptance_date']  # Rename column for merging
    # print(moe_data)
    productions = pd.merge_asof(productions, moe_data[['timestamp', 'cumulative_capacity']], on='timestamp', direction='backward')
    productions.rename(columns={'cumulative_capacity': 'installed_capacity'}, inplace=True)
    if wf_id == 25:
        # set timestamp of the last moe data to 2020-01-01
        moe_data.loc[moe_data['acceptance_date'] == moe_data['acceptance_date'].max(), 'acceptance_date'] = '2020-01-01'

    era5_ws100 = analyzer.get_era5_data(wf_id, start_date, end_date, grid_number=0, variables_to_plot=['ws100'])
    def find_good_quality_intervals(productions, era5_ws100, availability_threshold=0.90):

        # Merge wind speed data with production data
        productions = pd.merge(productions, era5_ws100, on='timestamp', how='inner')

        # special corrections because we are *sure* that something is incorrect from the source
        if wf_id == 9:
            availability_threshold = 0.96
        elif wf_id == 12:
            productions['installed_capacity'] = productions['installed_capacity'] + 1.5
        elif wf_id == 18:
            # remove data after 2023-09-01
            productions = productions[productions['timestamp'] < '2023-09-01']
        elif wf_id == 29:
            productions['installed_capacity'] = productions['installed_capacity'] + 10
            productions = productions[productions['timestamp'] > '2020-02-01']
        elif wf_id == 42:
            productions['installed_capacity'] = productions['installed_capacity'] + 24.1
        elif wf_id == 44:
            # remove data before 2020-03-01
            productions = productions[productions['timestamp'] > '2020-03-01']
        elif wf_id == 51:
            productions['installed_capacity'] = productions['installed_capacity'] - 9
        elif wf_id == 82:
            productions['installed_capacity'] = productions['installed_capacity'] + 28.202
        elif wf_id == 62:
            productions['installed_capacity'] = productions['installed_capacity'] + 0.85
            # remove data before 2020-10-01
            productions = productions[productions['timestamp'] > '2020-11-01']
        elif wf_id == 68:
            # remove data before 2020-03-01
            productions = productions[productions['timestamp'] > '2020-03-01']
        elif wf_id == 95:
            # set availability before 2020-10-01 to 21.6
            productions.loc[productions['timestamp'] < '2020-09-01', 'availability'] = 21.6
        elif wf_id == 111:
            # remove data after 2023-10-01
            productions = productions[productions['timestamp'] < '2023-10-01']
        elif wf_id == 112:
            # remove data after 2023-10-01
            productions = productions[productions['timestamp'] < '2023-11-01']
        elif wf_id == 122:
            # remove data before 2021-12-01
            productions = productions[productions['timestamp'] > '2021-12-01']
        elif wf_id == 124:
            # set availability before 2022-01-01 and after 2020-10-01 to 3.9
            productions.loc[(productions['timestamp'] < '2022-01-01') & (productions['timestamp'] > '2020-11-15'), 'availability'] = 3.9
            # set availability before 2020-10-01 to 3.2
            productions.loc[productions['timestamp'] < '2020-11-16', 'availability'] = 3.2
        elif wf_id == 126:
            # remove data before 2021-02-01
            productions = productions[productions['timestamp'] > '2021-02-01']
        elif wf_id == 128:
            # remove data before 2020-02-01
            productions = productions[productions['timestamp'] > '2020-02-01']
        elif wf_id == 131:
            # set availability before 2022-01-01 to 52.5
            productions.loc[productions['timestamp'] < '2022-01-01', 'availability'] = 52.5
        elif wf_id == 135:
            # remove data before 2020-07-01
            productions = productions[productions['timestamp'] > '2020-07-01']
        elif wf_id == 136:
            # remove data before 2021-10-01
            productions = productions[productions['timestamp'] > '2021-10-01']
        elif wf_id == 144:
            # remove data before 2020-12-01
            productions = productions[productions['timestamp'] > '2020-12-01']
        elif wf_id == 146:
            # remove data after 2023-01-01
            productions = productions[productions['timestamp'] < '2023-01-01']
        elif wf_id == 151:
            # set availability after 2022-01-01 to 4.8
            productions.loc[productions['timestamp'] > '2022-01-01', 'availability'] = 4.8
        elif wf_id == 159:
            # set availability before 2022-01-01 to 35
            productions.loc[productions['timestamp'] < '2022-01-01', 'availability'] = 35
        elif wf_id == 161:
            # remove data before 2021-07-01
            productions = productions[productions['timestamp'] > '2021-07-01']
        elif wf_id == 174:
            # remove data before 2020-10-01
            productions = productions[productions['timestamp'] > '2020-10-01']
        elif wf_id == 184:
            # set availability before 2022-01-01 to 12
            productions.loc[productions['timestamp'] < '2022-01-01', 'availability'] = 12
        elif wf_id == 186:
            # remove data before 2020-12-01
            productions = productions[productions['timestamp'] > '2020-12-01']
        elif wf_id == 190:
            # remove data before 2021-08-01
            productions = productions[productions['timestamp'] > '2021-08-01']
        elif wf_id == 194:
            # set availability before 2021-01-01 to 10
            productions.loc[productions['timestamp'] < '2021-01-01', 'availability'] = 10
        elif wf_id == 213:
            # remove data after 2023-10-01
            productions = productions[productions['timestamp'] < '2023-10-01']
        elif wf_id == 226:
            # remove data before 2021-07-01
            productions = productions[productions['timestamp'] > '2021-07-01']
        elif wf_id == 228:
            # remove data before 2022-01-01
            productions = productions[productions['timestamp'] > '2022-01-01']
        elif wf_id == 236:
            # remove data before 2020-11-01
            productions = productions[productions['timestamp'] > '2020-11-01']
        elif wf_id == 242:
            # remove data before 2021-03-01
            productions = productions[productions['timestamp'] > '2021-03-01']
        elif wf_id == 258:
            # remove data after 2023-12-01
            productions = productions[productions['timestamp'] < '2023-12-01']
        elif wf_id == 260:
            # remove data before 2022-01-01
            productions = productions[productions['timestamp'] > '2022-01-01']
        elif wf_id == 264:
            # remove data before 2021-10-01
            productions = productions[productions['timestamp'] > '2021-10-01']
        elif wf_id == 268:
            # remove data before 2021-08-01
            productions = productions[productions['timestamp'] > '2021-09-01']
        elif wf_id == 287:
            # remove data before 2022-01-01
            productions = productions[productions['timestamp'] > '2022-01-01']


        productions_ = productions.copy()

        # 1. Pre-filtering
        # remove data above capacity if the most of the data is not above capacity (so that we make sure the capacity is not wrong)
        if len(productions[productions['production'] > productions['installed_capacity']]) > len(productions) / 2:
            print('Most of the data is above the installed capacity, checking if the availability data is better')
            if len(productions[productions['production'] > productions['availability']]) > len(productions) / 2:
                print('Most of the data is above the availability.')
                print('\033[91m' + 'The data is not reliable.' + '\033[0m')
                if plot == True:
                    os.chdir("../..")
                driver.close()
                return None, None
            else:
                # use the availability data for pre-filtering
                print('Using the availability data for pre-filtering')
                productions = productions[productions['production'] <= productions['availability'] + productions['installed_capacity'].max() * 0.05]
                # remove the data where the given availability is lower than the 95th percent of the availability data
                productions = productions[productions['availability'] >= productions['availability'].quantile(availability_threshold)]

        elif (len(productions[productions['production'] > productions['availability']]) > len(productions) / 3 or
                    (len(productions[productions['availability'] == 0]) > len(productions) / 2)):
            print('Most of the data is above the availability (1/3), checking if the capacity data is better')
            if len(productions[productions['production'] > productions['installed_capacity']]) > len(productions) / 2:
                print('Most of the data is above the installed capacity.')
                print('\033[91m' + 'The data is not reliable.' + '\033[0m')
                if plot == True:
                    os.chdir("../..")
                driver.close()
                return None, None
            else:
                # use the installed capacity data for pre-filtering
                print('Using the installed capacity data for pre-filtering (Availability percentage is NOT used)')
                productions = productions[productions['production'] <= productions['installed_capacity'] + productions['installed_capacity'].max() * 0.05]

        else:
            # remove data above capacity
            productions = productions[productions['production'] <= productions['installed_capacity'] + productions['installed_capacity'].max() * 0.05] # add 5% for error margin

            # calculate the availability percentage (we know the capacity is correct now (at least for the most part))
            productions['availability_percentage'] = productions['availability'] / productions['installed_capacity']

            # remove data below the availability threshold
            productions = productions[productions['availability_percentage'] >= availability_threshold]

            # remove data above availability
            productions = productions[productions['availability_percentage'] <= 1 + (1 - availability_threshold)]

        if productions is None:
            print('We are filtering everything out for {}'.format(wf_id))
            if plot == True:
                os.chdir("../..")
            driver.close()
            return None, None
        def filter_low_production_periods(productions, percentage=0.50, duration=14):

            # Calculate 85% of the installed capacity
            productions['capacity_threshold'] = productions['installed_capacity'] * percentage

            # Identify periods where production is ALWAYS below the threshold
            productions['below_threshold'] = productions['production'] < productions['capacity_threshold']
            productions['period'] = (productions['below_threshold'] != productions['below_threshold'].shift()).cumsum()

            # Calculate the duration of each period
            period_duration = productions.groupby('period')['timestamp'].agg(['min', 'max'])
            period_duration['duration'] = period_duration['max'] - period_duration['min']

            # Identify periods longer than 7 days where production is ALWAYS below the threshold
            periods_to_remove = period_duration[
                (period_duration['duration'] > pd.Timedelta(days=duration))
                & (productions.groupby('period')['below_threshold'].all())
            ].index

            # Get the indices of the data points within the periods to remove
            indices_to_remove = productions[productions['period'].isin(periods_to_remove)].index

            # Remove data points corresponding to low production periods
            productions_filtered = productions.drop(index=indices_to_remove)

            return productions_filtered

        # Some data requires special treatment, unfortunately this is done manually here instead of coming up with a better algorithm.
        # Meter data cleaning is a manual work...
        if wf_id == 5:
            productions = filter_low_production_periods(productions, percentage=0.5, duration=14)
        elif wf_id == 7:
            productions = filter_low_production_periods(productions, percentage=0.5, duration=14)
        elif wf_id == 8:
            productions = filter_low_production_periods(productions, percentage=0.4, duration=14)
        elif wf_id == 9:
            productions = filter_low_production_periods(productions, percentage=0.4, duration=14)
        elif wf_id == 15:
            productions = filter_low_production_periods(productions, percentage=0.95, duration=30)
        elif wf_id == 30:
            productions = filter_low_production_periods(productions, percentage=0.95, duration=7)
        elif wf_id == 32:
            productions = filter_low_production_periods(productions, percentage=0.90, duration=7)
        elif wf_id == 35:
            productions = filter_low_production_periods(productions, percentage=0.85, duration=14)
        elif wf_id == 41:
            productions = filter_low_production_periods(productions, percentage=0.85, duration=7)
        elif wf_id == 42:
            productions = filter_low_production_periods(productions, percentage=0.8, duration=7)
        elif wf_id == 56:
            productions = filter_low_production_periods(productions, percentage=0.4, duration=14)
        elif wf_id == 59:
            productions = filter_low_production_periods(productions, percentage=0.9, duration=14)
        elif wf_id == 62:
            productions = filter_low_production_periods(productions, percentage=0.9, duration=7)
        elif wf_id == 65:
            productions = filter_low_production_periods(productions, percentage=0.4, duration=7)
        elif wf_id == 71:
            productions = filter_low_production_periods(productions, percentage=0.4, duration=14)
        elif wf_id == 87:
            productions = filter_low_production_periods(productions, percentage=0.3, duration=7)
        elif wf_id == 135:
            productions = filter_low_production_periods(productions, percentage=0.85, duration=14)
        elif wf_id == 144:
            productions = filter_low_production_periods(productions, percentage=0.4, duration=14)
        elif wf_id == 148:
            productions = filter_low_production_periods(productions, percentage=0.6, duration=14)
        elif wf_id == 187:
            productions = filter_low_production_periods(productions, percentage=0.85, duration=14)
        elif wf_id == 236:
            productions = filter_low_production_periods(productions, percentage=0.6, duration=14)
        elif wf_id == 242:
            productions = filter_low_production_periods(productions, percentage=0.6, duration=14)
        elif wf_id == 264:
            productions = filter_low_production_periods(productions, percentage=0.4, duration=7)

        else:
            productions = filter_low_production_periods(productions)

        # Run it again to remove periods with 0 production for a minimum of 1 day - Why didn't I think this before...
        productions = filter_low_production_periods(productions, percentage=0.001, duration=1)


        productions_before_outliers = productions.copy()

        try:
            # 2. Outlier Removal
            # Prepare data for DBSCAN
            data = productions[['ws100', 'production']].values

            # choose epsilon based on the size of the installed capacity
            # if it's a small farm, choose a smaller epsilon
            if productions['installed_capacity'].max() < 100:
                epsilon = 0.6
                min_samples = 25
            else:
                epsilon = 0.8
                min_samples = 15

            # Also needed to change these based on the left available data points,
            # but I'll have to go with manual adjustments againg
            # Meter data cleaning is a still a manual work...
            if wf_id == 5:
                epsilon = 0.8
                min_samples = 20
            elif wf_id == 209: # removed anyway
                epsilon = 0.85
                min_samples = 10
            elif wf_id == 226:
                epsilon = 0.8
                min_samples = 10
            elif wf_id == 29:
                epsilon = 0.85
                min_samples = 10


            #print(f"epsilon: {epsilon}, min_samples: {min_samples}")

            # DBSCAN for power curve outliers
            dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)  # Adjust eps and min_samples as needed
            clusters = dbscan.fit_predict(data)

            # Identify the main cluster (largest cluster)
            main_cluster_label = np.argmax(np.bincount(clusters[clusters != -1]))
            mask_dbscan = clusters == main_cluster_label

            # Calculate covariance matrix and inverse covariance
            covariance = np.cov(data[mask_dbscan].T)
            inv_cov = np.linalg.inv(covariance)

            # Combine masks from DBSCAN
            mask = mask_dbscan

            # Apply mask to filter outliers
            productions_filtered = productions[mask]

            # Reset index to ensure continuous index values
            productions_filtered = productions_filtered.reset_index(drop=True)

        except Exception as e:
            print(e)
            productions_filtered = productions.copy()

        if plot:
            p_scatter = figure(width=1800, height=1800,
                               x_axis_label='Wind Speed (m/s)',
                               y_axis_label='Production (MWh)',
                               title="Production vs Wind Speed")

            r = p_scatter.scatter(productions_['ws100'], productions_['production'],
                                  color='red', legend_label='Outliers', size=1)
            o = p_scatter.scatter(productions_before_outliers['ws100'], productions_before_outliers['production'],
                                  color='orange', legend_label='Outliers after pre-filter', size=1)
            b = p_scatter.scatter(productions_filtered['ws100'], productions_filtered['production'],
                                  color='blue', legend_label='Production Data', size=1.6)

            legend = Legend(items=[("Outliers", [r]),
                                  ("Outliers after pre-filter", [o]),
                                  ("Production Data", [b])])
            p_scatter.add_layout(legend)

            p_scatter.output_backend="svg"

            p_scatter.y_range = Range1d(productions_['production'].min(),
                                       productions_['production'].max() + productions_['production'].std()/4)

            p_scatter.legend.nrows=1
            # put a little space between legend entries
            p_scatter.legend.spacing = 30

            p_scatter.toolbar.logo = None
            p_scatter.toolbar_location = None

            p_scatter.title.text_font_size = "36pt"
            p_scatter.legend.label_text_font_size = "26pt"

            p_scatter.xaxis.axis_label_text_font_size = "24pt"
            p_scatter.xaxis.major_label_text_font_size = "18pt"

            p_scatter.yaxis.axis_label_text_font_size = "24pt"
            p_scatter.yaxis.major_label_text_font_size = "18pt"

            export_png(p_scatter, filename="prod_vs_ws100({}).png".format(wf_id), webdriver=driver)
            #show(p_scatter)

        return productions_, productions_filtered

    productions, productions_filtered = find_good_quality_intervals(productions, era5_ws100)

    if productions_filtered is None:
        print("No good quality intervals found for the ID {}".format(wf_id))
        if plot == True:
            os.chdir("../..")
        driver.close()
        return None, None


    def plot_good_quality_intervals(productions, productions_filtered):

        # Create the figure with an extra y-range for each new axis
        p = figure(x_axis_type="datetime",
                   title="Good Quality Intervals and Production Data for {}({})".format(farm_name, wf_id),
                   width=3600, height=1800,
                   x_range=(productions['timestamp'].min() - pd.DateOffset(days=30),
                            productions['timestamp'].max() + pd.DateOffset(days=30)),
                   y_range=(productions['production'].min() - productions['production'].std(),
                            productions['production'].max() + productions['production'].std()),
                   output_backend="svg", sizing_mode="fixed")

        p.axis.axis_label = "Production (MWh)"

        # Define extra y-ranges
        p.extra_y_ranges["availability"] = Range1d(productions['production'].min() - productions['production'].std(), productions['production'].max() + productions['production'].std())
        p.extra_y_ranges["installed_capacity"] = Range1d(start=productions['production'].min() - productions['production'].std(), end=productions['production'].max() + productions['production'].std())

        # Add new axes
        p.add_layout(LinearAxis(y_range_name="availability", axis_label="Availability (MWe)"), 'right')
        p.add_layout(LinearAxis(y_range_name="installed_capacity", axis_label="Installed Capacity (MWe)"), 'left')

        # put the exact values in the y axis for the installed capacity axis
        p.yaxis[1].ticker = productions['installed_capacity'].unique()


        # set the timestamps in production data to NaN that are in the filtered data
        productions.loc[productions['timestamp'].isin(productions_filtered['timestamp']), 'production'] = np.nan

        # fill the non existent date gaps in the dataframe with null values
        try:
            productions_filtered = productions_filtered.set_index('timestamp').resample('h').asfreq().reset_index()
        except Exception as e:
            print(e)

        # Plot production data (on the primary y-axis)
        p.line(x='timestamp', y='production', source=ColumnDataSource(productions),
               legend_label="Production (MWh)", line_color="red", line_width=1)

        p.line(x='timestamp', y='production', source=ColumnDataSource(productions_filtered),
                legend_label="Production (MWh)", line_color="blue", line_width=1.4)

        # Plot availability data (on its own axis)
        p.line(x='timestamp', y='availability', source=ColumnDataSource(productions),
               legend_label="Availability (MWe)", line_color="orange", line_width=1.4,
               y_range_name="availability")  # Specify the y-range

                # Plot installed capacity data (on its own axis)
        p.step(x='timestamp', y='installed_capacity', source=ColumnDataSource(productions),
               legend_label="Installed Capacity (MWe)", line_color="green", line_width=2, mode="after",
               y_range_name="installed_capacity")  # Specify the y-range

        # Plot good quality intervals with alternating colors
        colors = ["orange", "orange"]  # Define the two colors to alternate between

        # Add hover tooltips to the plot
        p.add_tools(HoverTool(
            tooltips=[
                ("Timestamp", "@timestamp{%F}"),
                ("Production", "@production{0.2f} MWh"),
                ("Installed Capacity", "@installed_capacity{0.2f} MWe"),
                ("Availability", "@availability{0.2f}")
            ],
            formatters={
                '@timestamp': 'datetime'
            },
            mode='vline'
        ))

        p.output_backend="svg"

        p.toolbar.logo = None
        p.toolbar_location = None

        p.title.text_font_size = "36pt"
        p.legend.label_text_font_size = "28pt"

        p.xaxis.axis_label = "Timestamp"
        p.xaxis.axis_label_text_font_size = "24pt"
        p.xaxis.major_label_text_font_size = "18pt"
        p.xaxis[0].formatter = DatetimeTickFormatter(months="%b %Y")

        p.yaxis.axis_label_text_font_size = "24pt"
        p.yaxis.major_label_text_font_size = "18pt"

        p.legend.nrows=1
        p.legend.spacing = 30
        export_png(p, filename="intervals_and_production({}).png".format(wf_id), webdriver=driver)
        #show(p)
        return p

    def plot_good_quality_intervals_cf(productions_filtered):

        p = figure(x_axis_type="datetime", title="Averaged Capacity Factor for the Selected Time Frames for {}({})".format(farm_name, wf_id),
                   width=3600, height=1800,
                   x_range=(productions_filtered['timestamp'].min() - pd.DateOffset(days=90), productions_filtered['timestamp'].max() + pd.DateOffset(days=180)),
                   y_range=(productions_filtered['capacity_factor'].min() - productions_filtered['capacity_factor'].std()/2, productions_filtered['capacity_factor'].max() + productions_filtered['capacity_factor'].std()/2),
                   output_backend="svg", sizing_mode="fixed")

        avg_cf_df = productions_filtered.groupby(pd.Grouper(key='timestamp', freq='QS'))['capacity_factor'].mean().reset_index()

        # interpolate the NaN values in the capacity factor
        avg_cf_df['capacity_factor'] = avg_cf_df['capacity_factor'].interpolate(method='linear')

        if avg_cf_df is not None:
            # Calculate the width of each bar based on the time difference to the next interval
            widths = [(avg_cf_df['timestamp'][i+1] - avg_cf_df['timestamp'][i]) / 2
                      for i in range(len(avg_cf_df)-1)]
            # Add a dummy width for the last bar to avoid error
            widths.append(widths[-1])
            avg_cf_df['width'] = widths

        p.extra_y_ranges["CF_availability"] = Range1d(productions_filtered['capacity_factor'].min() - productions_filtered['capacity_factor'].std()/2, productions_filtered['capacity_factor'].max() + productions_filtered['capacity_factor'].std()/2)
        p.add_layout(LinearAxis(y_range_name="CF_availability", axis_label="Capacity Factor Availability"), 'right')

        # Plot average capacity factor for each interval
        p.varea(x='timestamp', y1=0, y2='capacity_factor', source=ColumnDataSource(avg_cf_df),
                color="green", legend_label="Average Capacity Factor")
        #p.varea(x='timestamp', y1=0, y2='capacity_factor_ava', source=ColumnDataSource(avg_cf_df_ava),
        #        color="yellow", legend_label="Average Capacity Factor Ava", y_range_name="CF_availability", alpha=0.5)

        # make the values 2 decimal points (i.e. 0.00)
        avg_cf_df['capacity_factor_rounded'] = avg_cf_df['capacity_factor'].apply(lambda x: round(x, 2)).astype(str)
        #avg_cf_df_ava['capacity_factor_rounded'] = avg_cf_df_ava['capacity_factor_ava'].apply(lambda x: round(x, 2)).astype(str)
        def get_text_baseline(row):
            if row['capacity_factor'] > row['previous_capacity_factor']:
                return 'bottom'  # Convex
            else:
                return 'top'  # Concave
        def get_text_baseline_ava(row):
            if row['capacity_factor_ava'] > row['previous_capacity_factor']:
                return 'bottom'  # Convex
            else:
                return 'top'  # Concave

        # Calculate previous values for comparison (assuming your data is sorted by timestamp)
        avg_cf_df['previous_capacity_factor'] = avg_cf_df['capacity_factor'].shift(1)
        #avg_cf_df_ava['previous_capacity_factor'] = avg_cf_df_ava['capacity_factor_ava'].shift(1)

        # Apply the function to determine baselines
        avg_cf_df['text_baseline'] = avg_cf_df.apply(get_text_baseline, axis=1)
        #avg_cf_df_ava['text_baseline'] = avg_cf_df_ava.apply(get_text_baseline_ava, axis=1)

        # Update the plot
        p.text(x='timestamp', y='capacity_factor', text='capacity_factor_rounded',
               source=ColumnDataSource(avg_cf_df),
               text_font_size="20pt", text_baseline='text_baseline',
               text_align="right", text_color="black")


        # # Calculate and plot trendline
        # z = np.polyfit(avg_cf_df['timestamp'].astype(np.int64) / 1e9, avg_cf_df['capacity_factor'], 1)
        # p_trend = np.poly1d(z)
        # avg_cf_df['trendline'] = p_trend(avg_cf_df['timestamp'].astype(np.int64) / 1e9)
        # p.line(x='timestamp', y='trendline', source=ColumnDataSource(avg_cf_df),
        #        line_color="red", line_dash=[4, 4], line_width=3, legend_label="Trendline")
        #
        # # Calculate downward trend percentage
        # y = p_trend(avg_cf_df['timestamp'].astype(np.int64) / 1e9)
        # y1 = y[0]
        # y2 = y[-1]
        # x = avg_cf_df['timestamp'].values
        # x1 = x[0]
        # x2 = x[-1]
        # dy = y2 - y1
        # dx = (x2 - x1).astype('timedelta64[s]').astype(float)
        # dx = (dx / 60 / 60 / 24 / 365)
        # downward_trend = (dy / dx) * 100
        #
        # # Add downward trend as text annotation
        # trend_label = Label(x=avg_cf_df['timestamp'].iloc[-1], y=avg_cf_df['trendline'].iloc[-1],
        #                     text=f"Trend: {downward_trend:.2f}%\nper year", text_color="orange", text_font_size="26pt",
        #                     x_offset=10, y_offset=10)
        # p.add_layout(trend_label)




        # cf_trend_df = productions_filtered.copy().dropna(subset=['capacity_factor'])

        cf_trend_df = productions_filtered.copy()
        cf_trend_df['capacity_factor'] = cf_trend_df['capacity_factor'].interpolate(method='linear')





        # Calculate and plot trendline
        z = np.polyfit(cf_trend_df['timestamp'].astype(np.int64) / 1e9, cf_trend_df['capacity_factor'], 1)
        p_trend = np.poly1d(z)
        # cf_trend_df['trendline'] = p_trend(cf_trend_df['timestamp'].astype(np.int64) / 1e9)
        cf_trend_df.loc[:, 'trendline'] = p_trend(cf_trend_df['timestamp'].astype(np.int64) / 1e9)

        p.line(x='timestamp', y='trendline', source=ColumnDataSource(cf_trend_df),
               line_color="red", line_dash=[4, 4], line_width=3, legend_label="Trendline")

        # Calculate downward trend percentage
        y = p_trend(cf_trend_df['timestamp'].astype(np.int64) / 1e9)
        y1 = y[0]
        y2 = y[-1]
        x = cf_trend_df['timestamp'].values
        x1 = x[0]
        x2 = x[-1]
        dy = y2 - y1
        dx = (x2 - x1).astype('timedelta64[s]').astype(float)
        dx = (dx / 60 / 60 / 24 / 365)
        downward_trend = (dy / dx) * 100
        # print(downward_trend)

        # Add downward trend as text annotation
        trend_label = Label(x=cf_trend_df['timestamp'].iloc[-1], y=cf_trend_df['trendline'].iloc[-1],
                            text=f"Trend: {downward_trend:.2f}%\nper year", text_color="orange", text_font_size="26pt",
                            x_offset=10, y_offset=10)
        p.add_layout(trend_label)





        # Add hover tooltips to the plot
        p.add_tools(HoverTool(
            tooltips=[
                ("Interval Start", "@timestamp{%F}"),
                ("Average Capacity Factor", "@capacity_factor{0.2f}")
            ],
            formatters={
                '@timestamp': 'datetime'
            },
            mode='vline'
        ))
        p.output_backend="svg"
        
        p.toolbar.logo = None
        p.toolbar_location = None

        p.title.text_font_size = "36pt"
        p.legend.label_text_font_size = "28pt"

        p.xaxis.axis_label = "Timestamp"
        p.xaxis.axis_label_text_font_size = "24pt"
        p.xaxis.major_label_text_font_size = "18pt"
        p.xaxis[0].formatter = DatetimeTickFormatter(months="%b %Y")

        p.yaxis.axis_label = "Average Capacity Factors"
        p.yaxis.axis_label_text_font_size = "24pt"
        p.yaxis.major_label_text_font_size = "18pt"

        p.legend.nrows=1
        p.legend.spacing = 30

        export_png(p, filename="averaged_cf({}).png".format(wf_id), webdriver=driver)
        #show(p)
        return p, downward_trend

    if plot:
        plot_good_quality_intervals(productions, productions_filtered)
        p, downwards_trend = plot_good_quality_intervals_cf(productions_filtered)
        del p

    import gc
    gc.collect()
    driver.close()
    if plot == True:
        os.chdir("../..")
    if plot:
        return productions_filtered, downwards_trend
    else:
        return productions_filtered