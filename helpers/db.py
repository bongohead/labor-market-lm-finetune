import psycopg2
from sqlalchemy import create_engine
import pandas as pd
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv(Path().absolute() / '.env', override = True)

def get_postgres_query(query: str) -> pd.DataFrame: 
    """
    Get query result from Postgres
    """
    engine = create_engine(
        "postgresql+psycopg2://{user}:{password}@{host}/{dbname}".format(
           dbname = os.getenv('DB_DATABASE'),
           user = os.getenv('DB_USERNAME'),
           password = os.getenv('DB_PASSWORD'),
           host = os.getenv('DB_SERVER')
        )
    )
    
    pg = engine.connect()
    res = pd.read_sql(query, con = pg)
    pg.close()
    
    return res
