import os
from selenium import webdriver
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from src.DatabaseGetInfo import DatabasePlotter as dbPlot
from src.DatabaseGetInfo import DatabaseAnalyzer
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Database path
path = os.path.abspath('/home/wheatley/WFD/wfd.db')
analyzer = DatabaseAnalyzer.WindFarmAnalyzer(path)
plotter = dbPlot

start_date = '2020-01-01'
end_date = '2024-01-01'

def save_folium_map_as_image(map, image_path):
    """Saves a Folium map as an image."""

    # Temporarily save the map as an HTML file
    temp_html_path = "temp_map.html"
    map.save(temp_html_path)

    # Set up the webdriver (using Firefox in this example)
    options = webdriver.FirefoxOptions()
    options.add_argument("--headless")  # Run Firefox in headless mode (no GUI)
    driver = webdriver.Firefox(options=options)

    # Load the HTML file in the browser
    driver.get("file://" + os.path.abspath(temp_html_path))

    # Give the map some time to load (adjust the sleep time if needed)
    time.sleep(2)

    # Take a screenshot of the entire page
    driver.save_screenshot(image_path)

    # Close the browser
    driver.quit()

    # Remove the temporary HTML file
    os.remove(temp_html_path)


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
        capacity_info.append("Turbine Brands, Models, and Quantities:")
        for _, row in wf_turbines[['turbine_brand', 'turbine_model', 'turbine_power', 'start_date_of_operation', 'turbine_number']].iterrows():
            capacity_info.append(f"{row['turbine_brand']}       |     {row['turbine_model']}     |   {row['turbine_power']}     |     {row['start_date_of_operation']}     |     quantity: {row['turbine_number']}")

        # Add capacity information
        capacity_info.append(f"Production License Capacity (MWe) from EPDK: {wf['capacity_electrical'].sum()}")
        capacity_info.append(f"Production License Capacity (MWe) from MOE: {moe['additional_unit_power_electrical'].sum()}")
        capacity_info.append(f"Mechanical Installed Power (MWm) from EPDK: {wf['installed_power_mechanical'].sum()}")
        capacity_info.append(f"Mechanical Installed Power (MWm) from Turbines: {wf_turbines['installed_power'].sum()}")

        # Add changes throughout the years
        capacity_info.append("Capacity Change Throughout the Years:")
        capacity_info.append("Acceptance Date | Additional Power (MWe) | Aggregated Installed Power (MWe)")

        for _, row in moe[['acceptance_date', 'additional_unit_power_electrical', 'agg_installed_power']].iterrows():
            row['acceptance_date'] = row['acceptance_date'].split(" ")[0]
            capacity_info.append(f"{row['acceptance_date']} |   {row['additional_unit_power_electrical']}   |   {row['agg_installed_power']}")

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
        farm_info['Solar'] = farm_info.pop('solar')
        return farm_info
    else:
        return None


def wf_analysis_pdf(plots_folder, pdf_file, analyzer):
    """
    Generates a PDF report with two pages for each farm in the plots_folder.

    Page 1: Farm information, capacity information, and a map of turbine locations.
    Page 2: PNG plots of intervals_and_production and averaged_cf.

    Args:
        plots_folder (str): Path to the folder containing farm folders with PNG plots.
        pdf_file (str): Path to the output PDF file.
        analyzer (object): Analyzer object with methods to get farm data.
    """
    c = canvas.Canvas(pdf_file, pagesize=A4)
    c.setPageCompression(1)

    for farm_name in os.listdir(plots_folder):
        margin_left = 1 * cm  # Left margin in centimeters
        margin_bottom = 2 * cm  # Bottom margin in centimeters
        c.translate(margin_left, margin_bottom)
        page_width, page_height = A4
        page_width -= 2 * margin_left
        page_height -= 2 * margin_bottom
        pdfmetrics.registerFont(TTFont('NotoSans-VF', '/home/wheatley/.conda/envs/WFP/fonts/NotoSans-VF.ttf'))
        c.setFont("NotoSans-VF", 8)

        farm_folder = os.path.join(plots_folder, farm_name)
        if os.path.isdir(farm_folder):
            # Extract farm ID (assuming the folder name follows the pattern "...(###)...")
            farm_id_from_folder = farm_name.split(")")[0][-3:]
            farm_id_from_folder = ''.join(filter(str.isdigit, farm_id_from_folder))
            farm_id_from_folder = int(farm_id_from_folder)
            print(farm_id_from_folder)
            # --- Page 1 ---
            # Get and add farm information to the PDF
            farm_info = print_farm_info(analyzer, farm_id_from_folder, start_date, end_date)
            y_position = page_height
            if farm_info:
                for key, value in farm_info.items():
                    c.drawString(0.5 * cm, y_position, f"{key}: {value}")
                    y_position -= 0.4 * cm  # Increased spacing between farm info lines

            # Get and add capacity information to the PDF
            capacity_info = print_capacity_info(analyzer, farm_id_from_folder)

            y_position -= 0.6 * cm  # Increased spacing before capacity info
            if capacity_info:
                # Add turbine brands and models
                c.drawString(0.5 * cm, y_position, capacity_info.pop(0))  # Draw the header and remove it from the list
                y_position -= 0.5 * cm
                for item in capacity_info[:]:
                    if item.startswith("Production"):
                        break
                    c.drawString(1.5 * cm, y_position, item)
                    y_position -= 0.5 * cm
                    capacity_info.pop(0)
                y_position -= 0.5 * cm  # Add extra space after turbine models

                # Add capacity information as strings
                c.drawString(0.5 * cm, y_position, capacity_info.pop(0))  # Draw the first capacity info and remove it
                y_position -= 0.5 * cm  # Increased spacing between lines
                c.drawString(0.5 * cm, y_position, capacity_info.pop(0))
                y_position -= 0.5 * cm
                c.drawString(0.5 * cm, y_position, capacity_info.pop(0))
                y_position -= 0.5 * cm
                c.drawString(0.5 * cm, y_position, capacity_info.pop(0))
                y_position -= 1 * cm  # Increased spacing before changes

                # Add changes throughout the years
                c.drawString(0.5 * cm, y_position, capacity_info.pop(0))  # Draw the header and remove it
                y_position -= 0.5 * cm
                for item in capacity_info:
                    c.drawString(1.5 * cm, y_position, item)
                    y_position -= 0.5 * cm

            # Plot turbine centroids on a Folium map and save it as an image
            map, elevation_data = dbPlot.plot_turbine_centroids(analyzer, farm_id_from_folder)
            map_image_path = os.path.join(farm_folder, "map_image.png")
            save_folium_map_as_image(map, map_image_path)
            y_position -= 10 * cm  # Increased spacing before the map
            # Add the map image to the PDF
            c.drawImage(map_image_path, 0 * cm, y_position, width=19 * cm, height=10 * cm)
            c.showPage()  # End of page 1

            # --- Page 2 ---
            # Add PNGs to PDF, arranged top to bottom
            png_files = [
                os.path.join(farm_folder, "intervals_and_production({}).png".format(farm_id_from_folder)),
                os.path.join(farm_folder, "averaged_cf({}).png".format(farm_id_from_folder)),
            ]

            y_position = page_height - 6.5 * cm  # Initial y-position for the first image
            for png_file in png_files:
                if os.path.exists(png_file):
                    c.drawImage(png_file, 0.5 * cm, y_position, width=20 * cm, height=10 * cm)
                    y_position -= 10 * cm  # Adjust y-position for the next image

            madeup_power_curve = os.path.join(farm_folder, "prod_vs_ws100({}).png".format(farm_id_from_folder))
            if os.path.exists(madeup_power_curve):
                y_position += 1 * cm  # Adjust y-position (this and figure height should be = 10)
                c.drawImage(madeup_power_curve, 0.5 * cm, y_position, width=9 * cm, height=9 * cm)
                y_position -= 10 * cm
            c.showPage()  # End of page 2


    c.save()


plots_folder = "plots"
pdf_file = "CF_report.pdf"

# plots_folder = "plots_all"
# pdf_file = "CF_report_all.pdf"
wf_analysis_pdf(plots_folder, pdf_file, analyzer)