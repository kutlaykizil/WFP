import os
from src.DatabaseGetInfo import DatabaseAnalyzer
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Database path
path = os.path.abspath('/home/wheatley/WFD/wfd.db')
analyzer = DatabaseAnalyzer.WindFarmAnalyzer(path)

start_date = '2020-01-01'
end_date = '2024-01-01'

def print_capacity_info(analyzer, wf_id):
    """
    Returns capacity and power information using output from Analyzer as a list of strings.
    """
    moe = analyzer.get_moe_data(wf_id)
    wf_turbines = analyzer.get_wf_turbines_data(wf_id)
    wf = analyzer.get_wf_data(wf_id)

    if moe is not None and wf_turbines is not None and wf is not None:
        # Sort moe data by acceptance_date
        moe = moe.sort_values(by='acceptance_date')
        moe['agg_installed_power'] = moe['additional_unit_power_electrical'].cumsum()

        capacity_info = []

        # Add turbine brands, models and quantities
        capacity_info.append("--- Turbine Information ---")
        capacity_info.append(f"{'Brand':<15} | {'Model':<15} | {'Power (MW)':<15} | {'Start Date':<15} | {'Quantity':<10}")
        capacity_info.append("-" * 80)
        for _, row in wf_turbines[['turbine_brand', 'turbine_model', 'turbine_power', 'start_date_of_operation',
                                   'turbine_number']].iterrows():
            turbine_power_value = row['turbine_power']
            if pd.isna(turbine_power_value) or not isinstance(turbine_power_value, (int, float)):
                try:
                    turbine_power_mw = float(turbine_power_value) if turbine_power_value is not None else None
                except (ValueError, TypeError):
                    turbine_power_mw = None  # Fallback if conversion also fails
            else:
                turbine_power_mw = float(turbine_power_value)

            power_display = f"{turbine_power_mw:.3f}" if turbine_power_mw is not None else "N/A"
            # Ensure start_date_of_operation is string, handling potential None
            start_date_display = str(row['start_date_of_operation']) if pd.notna(
                row['start_date_of_operation']) else "N/A"
            turbine_number_display = str(row['turbine_number']) if pd.notna(
                row['turbine_number']) else "N/A"
            turbine_model_display = row['turbine_model'] if pd.notna(
                row['turbine_model']) else "N/A"
            turbine_brand_display = row['turbine_brand'] if pd.notna(
                row['turbine_brand']) else "N/A"

            capacity_info.append(
                f"{turbine_brand_display:<15} | {turbine_model_display:<15} | {power_display:<15} | {start_date_display:<15} | {turbine_number_display:<10}")

        # Add capacity information
        capacity_info.append("\n--- License and Installed Capacity ---")
        capacity_info.append(f"Production License Capacity (MWe) from EPDK: {wf['capacity_electrical'].sum():.3f}")
        capacity_info.append(f"Production License Capacity (MWe) from MOE: {moe['additional_unit_power_electrical'].sum():.3f}")
        capacity_info.append(f"Mechanical Installed Power (MWm) from EPDK: {wf['installed_power_mechanical'].sum():.3f}")
        capacity_info.append(f"Mechanical Installed Power (MWm) from Turbines: {wf_turbines['installed_power'].sum():.3f}")

        # Add changes throughout the years
        capacity_info.append("\n--- Capacity Change Throughout the Years ---")
        capacity_info.append(f"{'Acceptance Date':<20} | {'Additional Power (MWe)':<25} | {'Aggregated Installed Power (MWe)':<30}")
        capacity_info.append("-" * 80)
        for _, row in moe[['acceptance_date', 'additional_unit_power_electrical', 'agg_installed_power']].iterrows():
            row['acceptance_date'] = row['acceptance_date'].split(" ")[0]
            capacity_info.append(f"{row['acceptance_date']:<20} | {row['additional_unit_power_electrical']:<25.3f} | {row['agg_installed_power']:<30.3f}")

        return capacity_info
    else:
        return None

def print_farm_info(analyzer, wf_id, start_date, end_date):
    """
    Returns formatted site information using output from Analyzer.
    """
    farm_info = analyzer.get_farm_info(wf_id)

    # check the database if the wf_id has and solar production values in the production table and return true or false
    solar = analyzer.check_solar(wf_id, start_date, end_date)
    farm_info['solar'] = 'Yes' if solar else 'No'


    if farm_info:
        # only select the following keys
        farm_info = {key: farm_info[key] for key in ['license_number', 'plant_name', 'wf_id', 'license_holder', 'city', 'district', 'solar']}
        # change keys for formatting
        farm_info['License Number'] = farm_info.pop('license_number')
        farm_info['Plant Name'] = farm_info.pop('plant_name')
        farm_info['Wind Farm ID'] = farm_info.pop('wf_id')
        farm_info['License Holder'] = farm_info.pop('license_holder')
        farm_info['City'] = farm_info.pop('city')
        farm_info['District'] = farm_info.pop('district')
        farm_info['Solar Co-located'] = farm_info.pop('solar') # Renamed for clarity
        return farm_info
    else:
        return None


def wf_analysis_txt(plots_folder, analyzer):
    """
    Generates a TXT report for each farm in the plots_folder.

    Args:
        plots_folder (str): Path to the folder containing farm folders.
        analyzer (object): Analyzer object with methods to get farm data.
    """

    for farm_name in os.listdir(plots_folder):
        farm_folder = os.path.join(plots_folder, farm_name)
        if os.path.isdir(farm_folder):
            # Extract farm ID (assuming the folder name follows the pattern "...(###)...")
            farm_id_from_folder = farm_name.split(")")[0][-3:]
            farm_id_from_folder = ''.join(filter(str.isdigit, farm_id_from_folder))
            farm_id_from_folder = int(farm_id_from_folder)

            # if farm_id_from_folder <240:
            #     continue

            print(f"Processing Wind Farm ID: {farm_id_from_folder}")

            txt_file_path = os.path.join(farm_folder, f"{farm_name}_report.txt")
            with open(txt_file_path, 'w', encoding='utf-8') as f:
                f.write(f"--- Wind Farm Report for {farm_name} ---\n\n")

                # Get and add farm information to the TXT
                farm_info = print_farm_info(analyzer, farm_id_from_folder, start_date, end_date)
                if farm_info:
                    f.write("--- General Farm Information ---\n")
                    for key, value in farm_info.items():
                        f.write(f"{key}: {value}\n")
                    f.write("\n")

                # Get and add capacity information to the TXT
                capacity_info = print_capacity_info(analyzer, farm_id_from_folder)
                if capacity_info:
                    for line in capacity_info:
                        f.write(line + "\n")
                    f.write("\n")


# --- Main execution ---
plots_folder = "plots"
wf_analysis_txt(plots_folder, analyzer)