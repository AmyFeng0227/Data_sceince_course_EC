from sqlalchemy import create_engine
from config import server, database

#saving the dataframe to sql server
def save_to_sql(dataframe_name, table_name):
    connection_string = f'mssql+pyodbc://@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes'
    engine = create_engine(connection_string)
    dataframe_name.to_sql(name=table_name, con=engine, if_exists='replace', index=False)
    print(f"Dataframe saved succesfully to sql server table: {table_name}")
    
