from src.ThesisProject.CF_table.CF_interval_detect import detect_and_plot
import os
from src.DatabaseGetInfo import DatabasePlotter as dbPlot
from src.DatabaseGetInfo import DatabaseAnalyzer
import sqlite3
from src.GLOBAL_VARS import NotTrustedPlants

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Database path
path = os.path.abspath('/home/wheatley/WFD/wfd.db')
analyzer = DatabaseAnalyzer.WindFarmAnalyzer(path)
plotter = dbPlot

#wf_id_query = """select wf_id from wf where wf.city = 'İZMİR' and wf.license_status = 'Yürürlükte' order by wf_id"""
wf_id_query = """select wf_id from wf where wf.license_status = 'Yürürlükte' order by wf_id"""
wf_ids = sqlite3.connect(path).execute(wf_id_query).fetchall()
wf_ids = [wf_id[0] for wf_id in wf_ids]

wf_ids = [30]
#NotTrustedPlants = []

start_date = '2020-01-01'
end_date = '2024-01-01'

trends = {}

for wf_id in wf_ids:
    if wf_id in NotTrustedPlants:
        continue
    productions_filtered, trend = detect_and_plot(wf_id, start_date, end_date, plot=True) # Must be True
    del productions_filtered

    # save trends with wf_id as key:value pairs
    if trend is not None:
        trends[wf_id] = trend

#del file if exists
if os.path.exists('trends_raw.csv'):
    os.remove('trends_raw.csv')
# save trends to csv
with open('trends_raw.csv', 'w') as f:
    for key in trends.keys():
        f.write("%s,%s\n"%(key,trends[key]))

