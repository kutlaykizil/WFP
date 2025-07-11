import json
import requests
from urllib.parse import urlencode
import pandas as pd
from src.DatabaseGetInfo import DatabaseAnalyzer
import os
from datetime import date
from src.credentials import epias_uname, epias_pwd
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

session = requests.Session()

def get_tgt(uname, pwd):
    tgt_url = 'https://giris.epias.com.tr/cas/v1/tickets'

    request_data = {
        "username": epias_uname,
        "password": epias_pwd
    }
    encoded_data = urlencode(request_data)

    headers = {'Accept': 'text/plain', 'Content-Type': 'application/x-www-form-urlencoded'}
    tgt_response = requests.post(tgt_url, data=encoded_data, headers=headers)

    if tgt_response.status_code != 201:
        print(f"Failed to get TGT. Status code: {tgt_response.status_code}")
        print(tgt_response.text)  # Print any error message from the server
        return None

    return tgt_response.text

def get_uevcb(org_id, tgt, start_date=date.today().strftime("%Y-%m-%dT00:00:00+03:00")):
    uevcb = 'https://seffaflik.epias.com.tr/electricity-service/v1/generation/data/uevcb-list'
    request_json = {
        "startDate": start_date,
        "organizationId": org_id,
    }

    headers = {'TGT': f'{tgt}'}
    uevcb = session.post(uevcb, json=request_json, headers=headers)

    return uevcb

def get_rt_gen(rt_gen_id, start_date, end_date, tgt):
    rt_gen = 'https://seffaflik.epias.com.tr/electricity-service/v1/generation/data/realtime-generation'
    request_json = {
        "startDate": start_date,
        "endDate": end_date,
        "powerPlantId": rt_gen_id
    }

    headers = {'TGT': f'{tgt}'}
    rt_gen = session.post(rt_gen, json=request_json, headers=headers)

    return rt_gen

def get_uevm(uevm_id, start_date, end_date, tgt):
    uevm = 'https://seffaflik.epias.com.tr/electricity-service/v1/generation/data/injection-quantity'
    request_json = {
        "startDate": start_date,
        "endDate": end_date,
        "powerplantId": uevm_id
    }

    headers = {'TGT': f'{tgt}'}
    uevm = session.post(uevm, json=request_json, headers=headers)

    return uevm

def get_eak(id, start_date, end_date, org_id, uevcb_id, tgt):
    url = 'https://seffaflik.epias.com.tr/electricity-service/v1/generation/data/aic'

    request_json = {
        "startDate": start_date,
        "endDate": end_date,
        "organizationId": org_id,
        "uevcbId": uevcb_id,
        "region": "TR1"
    }

    headers = {'TGT': f'{tgt}'}
    eak_data = session.post(url, json=request_json, headers=headers)

    if eak_data.status_code != 200:
        return None
    else:
        return eak_data


path = os.path.abspath('/home/kutlay/WFD/wfd.db')
analyzer = DatabaseAnalyzer.WindFarmAnalyzer(path)
connection, cursor = analyzer.connect_to_db()

wf_id_list = analyzer.get_wf_id_list()

sql = """SELECT epias_plant_id, epias_uevm_plant_id, epias_organization_id, epias_uevm_plant_shortname FROM wf WHERE wf_id = ? and license_status is 'Yürürlükte' order by wf_id"""

epias_info = dict()
for wf_id in wf_id_list:
    cursor.execute(sql, (wf_id,))
    epias_info[wf_id] = cursor.fetchone()
# remove the rows with None in values
epias_info = {k: v for k, v in epias_info.items() if v[0] and v[1] and v[2] is not None}

#for key, value in epias_info.items():
#    print(key, value)

# drop the production_test table
#ursor.execute("DROP TABLE IF EXISTS production_test")
import time
tgt_response = get_tgt(epias_uname, epias_pwd)
i = 0

for id in epias_info: # SOMETIMES IT FAILS TO GET THE DATA,
                      # IN THAT CASE, TRY AGAIN FROM WHERE IT LEFT OFF

    #if id < 124 or id > 400:
    #    continue

    i += 1
    if i % 10 == 0:
        tgt_response = get_tgt(epias_uname, epias_pwd)

    time.sleep(5)

    df_combined = pd.DataFrame()
    rt_gen_id = int(epias_info[id][0])
    uevm_id = int(epias_info[id][1])
    org_id = int(epias_info[id][2])
    asset_name = epias_info[id][3]

    uevcb = get_uevcb(org_id, tgt_response)

    # Get the UEVCB ID for the plant from the response by matching the plant name
    try:
        # Sorry not sorry
        if id == 3:
            uevcb_id = 3689
            uevcb_name = 'ANEMON ENERJİ ELEKTRİK ÜRETİM A.Ş'
        elif id == 7:
            uevcb_id = 3779
            uevcb_name = 'DOĞAL ENERJİ ELEKTRİK ÜRETİM A.Ş(SAYALAR)'
        elif id == 11:
            uevcb_id = 3668
            uevcb_name = 'Mare Manastır RES Sanayi ve Ticaret Anonim Şirketi'
        elif id == 12:
            uevcb_id = 3568
            uevcb_name = 'ALİZE ENERJİ ELEKTRİK ÜRETİM A.Ş.'
        elif id == 13:
            uevcb_id = 4070
            uevcb_name = 'MAZI-3 RÜZGAR ENERJİSİ SANTRALİ'
        elif id == 14:
            uevcb_id = 5405
            uevcb_name = 'KORES KOCADAĞ RES'
        elif id == 16:
            uevcb_id = 4421
            uevcb_name = 'BELEN ELEKTRİK ÜRETİM A.Ş.'
        elif id == 26:
            uevcb_id = 3989
            uevcb_name = 'KELTEPE RES'
        elif id == 28:
            uevcb_id = 3988
            uevcb_name = 'ALİZE ENERJİ ELEKTRİK ÜRETİM A.Ş.(ÇAMSEKİ)'
        elif id == 29:
            uevcb_id = 4071
            uevcb_name = 'Ütopya Elektrik Üretim San. ve Tic. A.Ş.(Düzova RES)'
        elif id == 43:
            uevcb_id = 9810
            uevcb_name = 'SENBUK RES(BAKRAS ENR.)'
        elif id == 46:
            uevcb_id = 340575
            uevcb_name = 'AL-YEL ELEKTRİK ÜRETİM A.Ş.'
        elif id == 62:
            uevcb_id = 3648
            uevcb_name = 'TEPERES Elektrik Üretim A.Ş.(Tepe Res)'
        elif id == 64:
            uevcb_id = 2813609
            uevcb_name = 'ÇEŞME RES'
        elif id == 89:
            uevcb_id = 3006558
            uevcb_name = 'ÖDEMİŞ RES'
        elif id == 131:
            uevcb_id = 3194130
            uevcb_name = 'YAHYALI RES(SE)'
        elif id == 154:
            uevcb_id = 3204489
            uevcb_name = 'BERGAMA RES ( VENTO ELK ÜRETİM )'
        elif id == 157:
            uevcb_id = 1624872
            uevcb_name = 'KAVAKLI RES'
        elif id == 162:
            uevcb_id = 804475
            uevcb_name = 'Şadıllı RES'
        elif id == 199:
            uevcb_id = 3194923
            uevcb_name = 'SİNCİKRES'
        elif id == 217:
            uevcb_id = 3217232
            uevcb_name = 'ERİMEZ RES' # todo: check this, name is different
        elif id == 253:
            uevcb_id = 3217830
            uevcb_name = 'Kayadüzü RES'
        elif id == 306:
            uevcb_id = 1235449
            uevcb_name = 'Balabanlı RES'

        else:
            uevcb_id = next((item for item in uevcb.json()['items'] if asset_name in item['name']), None)['id']
            uevcb_name = next((item for item in uevcb.json()['items'] if asset_name in item['name']), None)['name']
    except:
        print(f"\033[91mUEVCB ID not found for {asset_name}.\033[0m")
        if uevcb.json()['items']:
            uevcb_id = input("Enter UEVCB ID   "+ (json.dumps(uevcb.json()['items'], indent=4, ensure_ascii=False)) + " : ")
            uevcb_name = input("Enter UEVCB Name:")

    start_date = '2014-01-01T00:00:00+03:00'
    #end_date = date.today() - datetime.timedelta(days=1)
    #end_date_uevm = (datetime.datetime.now().replace(day=1) - datetime.timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S") + "+03:00"
    end_date_uevm = date.today().replace(day=1).strftime("%Y-%m-%dT%H:%M:%S") + "+03:00"
    end_date = end_date_uevm

    print(f"UEVM ID: {uevm_id}", f"RT Gen ID: {rt_gen_id}", f"Organization ID: {org_id}", f"Asset Name: {asset_name}", f"UEVCB ID: {uevcb_id}", f"UEVCB Name: {uevcb_name}", f"Start Date: {start_date}", f"End Date: {end_date}", f"wf_id: {id}", sep='\n')

    id_matches = pd.DataFrame({'epias_id': [id], 'id': [1]}) # IDK what was this lol but should have no effect

    rt_gen_data = pd.DataFrame()
    uevm_data = pd.DataFrame()
    eak_data = pd.DataFrame()

    print("Retrieving production data...")

    for _, row in id_matches.iterrows():
        epias_id = row['epias_id']
        db_id = row['id']

        if not pd.isna(epias_id):

            while pd.to_datetime(start_date) <= pd.to_datetime(end_date):
                _rt_gen_data = None
                _uevm_data = None
                _eak_data = None

                time.sleep(0.5)

                chunk_end_date = pd.to_datetime(min(str(pd.to_datetime(start_date) + timedelta(days=90)), str(end_date))).strftime("%Y-%m-%dT%H:%M:%S+03:00")
                #print(f"Getting data for {start_date} to {chunk_end_date}")

                try:
                    _rt_gen_data = get_rt_gen(rt_gen_id, str(start_date), str(chunk_end_date), tgt_response)
                except Exception as e:
                    print(f"Failed to get RT Gen data: {e}")
                try:
                    _uevm_data = get_uevm(uevm_id, str(start_date), str(chunk_end_date), tgt_response)
                except Exception as e:
                    print(f"Failed to get UEVM data: {e}")
                try:
                    _eak_data = get_eak(uevm_id, str(start_date), str(chunk_end_date), org_id, uevcb_id, tgt_response)
                except Exception as e:
                    print(f"Failed to get EAK data: {e}")

                if _rt_gen_data:
                    #saveProd(str(db_id), start_date, chunk_end_date, rt_gen_data)
                    _rt_gen_data = pd.DataFrame(_rt_gen_data.json()['items'])
                    rt_gen_data = pd.concat([rt_gen_data, _rt_gen_data], ignore_index=True)
                if _uevm_data:
                    #saveProd(str(db_id), start_date, chunk_end_date, uevm_data)
                    _uevm_data = pd.DataFrame(_uevm_data.json()['items'])
                    uevm_data = pd.concat([uevm_data, _uevm_data], ignore_index=True)
                if _eak_data:
                    #saveProd(str(db_id) + "_eak", start_date, chunk_end_date, eak_data)
                    _eak_data = pd.DataFrame(_eak_data.json()['items'])
                    eak_data = pd.concat([eak_data, _eak_data], ignore_index=True)

                start_date = (pd.to_datetime(chunk_end_date) + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S+03:00")

    print("Data retrieval complete.")
    rt = rt_gen_data.copy()
    uevm = uevm_data.copy()
    eak = eak_data.copy()

    solar = False

    # check if the data have solar values other than 0 and if there is set solar = True
    if 'sun' in uevm.columns:
        if uevm['sun'].sum() > 0:
            solar = True
    if 'sun' in rt.columns:
        if rt['sun'].sum() > 0:
            solar = True
    if 'diger' in eak.columns:
        if eak['diger'].sum() > 0:
            solar = True

    if solar == True:
        rt = rt[['date', 'wind', 'sun']]
        uevm = uevm[['date', 'wind', 'sun']]
        eak = eak[['date', 'ruzgar', 'diger']]

        rt = rt.rename(columns={'date': 'DateTime', 'wind': 'Wind', 'sun': 'Solar'})
        uevm = uevm.rename(columns={'date': 'DateTime', 'wind': 'Wind', 'sun': 'Solar'})
        eak = eak.rename(columns={'date': 'DateTime', 'ruzgar': 'Wind', 'diger': 'Solar'})

        rt = rt.dropna(subset=['Wind', 'Solar'], how='all')
        uevm = uevm.dropna(subset=['Wind', 'Solar'], how='all')
        eak = eak.dropna(subset=['Wind', 'Solar'], how='all')
    if solar == False:
        rt = rt[['date', 'wind']]
        uevm = uevm[['date', 'wind']]
        eak = eak[['date', 'ruzgar']]

        rt = rt.rename(columns={'date': 'DateTime', 'wind': 'Wind'})
        uevm = uevm.rename(columns={'date': 'DateTime', 'wind': 'Wind'})
        eak = eak.rename(columns={'date': 'DateTime', 'ruzgar': 'Wind'})

        rt = rt.dropna(subset=['Wind'], how='all')
        uevm = uevm.dropna(subset=['Wind'], how='all')
        eak = eak.dropna(subset=['Wind'], how='all')


    rt['DateTime'] = pd.to_datetime(rt['DateTime'], format='%Y-%m-%dT%H:%M:%S%z', utc=True).dt.tz_convert('Europe/Istanbul')
    uevm['DateTime'] = pd.to_datetime(uevm['DateTime'], format='%Y-%m-%dT%H:%M:%S%z', utc=True).dt.tz_convert('Europe/Istanbul')
    eak['DateTime'] = pd.to_datetime(eak['DateTime'], format='%Y-%m-%dT%H:%M:%S%z', utc=True).dt.tz_convert('Europe/Istanbul')


    # combine the dataframes on the DateTime column with names wind_rt and wind_uevm and wind_eak
    if solar == False:
        df_combined = pd.merge(rt, uevm, on='DateTime', how='outer', suffixes=('_rt', '_uevm'))
        df_combined = pd.merge(df_combined, eak, on='DateTime', how='outer')
    if solar == True:
        df_combined = pd.merge(rt, uevm, on='DateTime', how='outer', suffixes=('_rt', '_uevm'))
        df_combined = pd.merge(df_combined, eak, on='DateTime', how='outer', suffixes=('_uevm', '_eak'))

    df_combined = df_combined.sort_values(by='DateTime').reset_index(drop=True)

    # find dates without timezone
    df_combined['DateTime'] = df_combined['DateTime'].dt.tz_localize(None)

    # save the data into the database
    if solar == False:
        df_combined = df_combined.rename(columns={'DateTime': 'date', 'Wind_rt': 'wind_rt', 'Wind_uevm': 'wind_uevm', 'Wind': 'wind_eak'})
        df_combined = df_combined.dropna(subset=['wind_rt', 'wind_uevm', 'wind_eak'], how='all')
        # add solar columns and set them to null
        df_combined['solar_rt'] = None
        df_combined['solar_uevm'] = None
        df_combined['solar_eak'] = None
    if solar == True:
        df_combined = df_combined.rename(columns={'DateTime': 'date', 'Wind_rt': 'wind_rt', 'Solar_rt': 'solar_rt', 'Wind_uevm': 'wind_uevm', 'Solar_uevm': 'solar_uevm', 'Wind': 'wind_eak', 'Solar': 'solar_eak'})
        df_combined = df_combined.dropna(subset=['wind_rt', 'solar_rt', 'wind_uevm', 'solar_uevm', 'wind_eak', 'solar_eak'], how='all')

    # set the date column to ts_id for the database
    df_combined['ts_id'] = df_combined['date'].apply(lambda x: 3682081 + int((x - pd.Timestamp('2017-01-01 00:00:00')).total_seconds() / 60))

    # add wf_id column
    df_combined['wf_id'] = id
    # drop date column
    df_combined = df_combined.drop(columns=['date'])

    df_combined.to_sql('production_test', connection, if_exists='append', index=False)

    print("Data saved to database.\n\n")

print("\n\n ALL DATA SAVED TO DATABASE.")
connection.close()