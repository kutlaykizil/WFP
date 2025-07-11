import sympy as sp

def convert_to_float(text):
    text = str(text)
    # Replace all variables with symbols
    text = text.replace("x", "*")
    text = text.replace("X", "*")
    text = text.replace(",", ".")

    if text == "-" or text == "":
        return text
    # Convert the expression to a sympy expression
    expr = sp.sympify(text)

    # Evaluate the expression numerically
    result = sp.N(expr)
    result = float(result)
    return result


import pandas as pd
bk_res = pd.DataFrame()
bk = pd.read_excel(f'../files-to-import/enerji-bakanligi/2024.xlsx', header=1)
bk = bk[bk['KAYNAK'] == 'RES']
bk_res = pd.concat([bk_res, bk])
bk_res.drop(columns=['SIRA NO','KAYNAK'], inplace=True)

from src.Obsolete.sshDb import ssh_db

# Start SSH tunnel and connect to DB
ssh_db.connect_ssh_db()
# Get table wf from database
wfPostgre = pd.read_sql_table('wf', ssh_db.engine)

ministryOfEnergy = pd.merge(wfPostgre[['id']], bk_res, right_on="LİSANS SAYISI", left_on=wfPostgre["Lisans No"], how="right", suffixes=('', ''))
ministryOfEnergy = ministryOfEnergy.set_index('id')
ministryOfEnergy['LİSANS TARİHİ']= pd.to_datetime(ministryOfEnergy['LİSANS TARİHİ']) # Turn to UTC ???????

ministryOfEnergy["Calculated ÜNİTE GÜCÜ (MWe)"] = ministryOfEnergy["ÜNİTE GÜCÜ (MWe)"].apply(convert_to_float)

ministryOfEnergy.to_sql('ministry_of_energy', ssh_db.engine, if_exists='replace', index=True)