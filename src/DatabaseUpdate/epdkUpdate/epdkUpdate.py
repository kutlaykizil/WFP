from src.Obsolete.sshDb import ssh_db
import pandas as pd

# Read excel file to dataframe.
# File must be inside 'files-to-import' folder with the name 'Elektrik Üretim Lisanslar_.xls'
epdk_list = pd.read_excel('../files-to-import/Elektrik Üretim Lisanslar_.xls', header=[0, 1])

# Filter only important rows to reduce memory for later.
epdk_list = epdk_list.loc[
    (epdk_list['Lisans Durumu', 'Unnamed: 3_level_1'] == 'Sonlandırıldı') |
    (epdk_list['Lisans Durumu', 'Unnamed: 3_level_1'] == 'Yürürlükte') |
    (epdk_list['Lisans Durumu', 'Unnamed: 3_level_1'] == 'İptal Edildi')
    ]

# Split multiindex columns for convenience
epdk_list_l0 = epdk_list.droplevel(1, axis=1)
epdk_list_l1 = epdk_list.droplevel(0, axis=1)

# Filter only important rows and remove multiindex, again for the same reasons
epdk_list_new = epdk_list.loc[epdk_list['Lisans Durumu', 'Unnamed: 3_level_1'] == 'Yürürlükte']
epdk_list_new = epdk_list_new.droplevel(1, axis=1)
# For some reason a list doesn't work
epdk_list_new = epdk_list_new[['Lisans No', 'Tesis Adı']]

# Start SSH tunnel and connect to DB
ssh_db.connect_ssh_db()
# Get table wf from database
wfPostgre = pd.read_sql_table('wf', ssh_db.engine)

# '~' means not in
df1_filtered = epdk_list_new[~epdk_list_new['Lisans No'].isin(wfPostgre['Lisans No'])]

wfPostgre = pd.concat([df1_filtered, wfPostgre], axis=0, ignore_index=True, sort=False)

# UPDATE EXISTING ROWS IN TABLE WF
# set index columns to match different tables
wfPostgre = wfPostgre.set_index('Lisans No')
epdk_list_l0 = epdk_list_l0.set_index('Lisans No')
epdk_list_l1 = epdk_list_l1.set_index('Unnamed: 4_level_1')

# update the existing licenses in table wf with newer info.
wfPostgre.update(epdk_list_l0)
wfPostgre.update(epdk_list_l1)

# reset the indexes (code will not run a 2. time if this is not done)
wfPostgre.reset_index(inplace=True)
epdk_list_l0.reset_index(inplace=True)
epdk_list_l1.reset_index(inplace=True)

# write the dataframe to sql table wf
wfPostgre.to_sql('wf', ssh_db.engine, if_exists='replace', index=False)

# Stop the SSH tunnel
ssh_db.stop_ssh_db()

