import pandas as pd
from src.Obsolete.sshDb import ssh_db

# Delete the first parts of the file that is about to be imported, since it messes with the import. Part to be
# deleted is: "{"resultCode":"0","resultDescription":"success","body":{"powerPlantList":[" and its closing
# parentheses at the end of the file Open the .json file
with open('../files4Import/nonBackup/modeified.json', 'r') as f:
    json_str = f.read()
# Replace all occurrences of null with None because 'pYtHoN'
json_str = json_str.replace('null', 'None')
# Convert the JSON string to a Python dictionary
json_dict = eval(json_str)
# Create a Pandas DataFrame from the dictionary
pp_list = pd.DataFrame(json_dict)
# Drop the name column because we don't need it
pp_list.drop('name', axis=1, inplace=True)
# Rename the columns to more convenient names
pp_list.rename(columns={'id': 'epias_id', 'eic': 'epias_eic', 'shortName': 'epias_name'}, inplace=True)

### NEED TO CHANGE THIS TO SOMETHING ELSE BUT IT WORKS FOR NOW
# Filter the list to rows that contain 'RES' in their name
df_filtered = pp_list[pp_list['epias_name'].str.contains('RES')]

# Select and delete all rows with the id numbers written in the code below
unwanted = df_filtered[
    (df_filtered['epias_id'] == 1779) |
    (df_filtered['epias_id'] == 3014) |
    (df_filtered['epias_id'] == 1025) |
    (df_filtered['epias_id'] == 2746) |
    (df_filtered['epias_id'] == 2776)].index
# Actual deletion part
df_filtered.drop(unwanted, inplace=True)

ssh_db.connect_ssh_db()

# Import the DataFrame into the PostgreSQL database
df_filtered.to_sql('epias_list', ssh_db.engine, if_exists='replace', index=False)

# Stop the SSH tunnel
ssh_db.stop_ssh_db()
