import sqlalchemy
import os
os.environ['SQLITE_TMPDIR']="/home/wfd/tmp"
path = '/home/wfd/Backups/wfd.db'
engine = sqlalchemy.create_engine('sqlite:///' + path)

from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import text

# vacuum the database
print("Vacuuming the database")
with engine.connect() as con:
    con.execute(text("VACUUM"))

