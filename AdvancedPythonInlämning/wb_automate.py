from worldbank_api import fetch_and_transform
from database import save_to_sql
import logging

logging.basicConfig(
    level=logging.INFO,
    filename='process.log',
    filemode='a', # "w" to rewrite
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    
    logging.info("Starting to fetch and transform world bank indicator data.")
    try:
        indicator_fact = fetch_and_transform()
    
        if indicator_fact.empty:
            logging.warning("No data fetched.")
            return
   
        logging.info("Data successfully fetched and transformed.")
        
    except Exception as e:
        logging.error(f"Error fetchning and transforming data: {e}")
        return
    
    logging.info("Starting to save data to sql server")
    try:
        save_to_sql(dataframe_name=indicator_fact, table_name="indicator_fact")
        logging.info(f"Data successfully saved in SQL server as table indicator_fact.")
    
    except Exception as f:
        logging.error(f'Error saving data to SQL server: {f}')
    
if __name__ == "__main__":
    main()