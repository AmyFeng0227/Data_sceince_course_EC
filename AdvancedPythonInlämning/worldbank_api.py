import pandas as pd
import wbgapi as wb
from config import selected_indicators


#observe data from World Bank api
def observe_data(search_word=""):
    result = wb.series.info(q=search_word)
    return result

#fetch and transform world bank indicator fact data
def fetch_and_transform():
    
    data = []
    
    for row in wb.data.fetch(selected_indicators, mrnev=1):
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Convert 'YR2011' to 2011
    df['time'] = df['time'].apply(lambda x: int(x[2:]))
    
    # drop aggregate column
    df.drop(columns=['aggregate'], inplace=True)
    
    # rename columns
    df.rename(columns={'time': 'year', 
                       "series": "indicator_id", 
                       "economy": "country_code"}, 
              inplace=True)
    
    return df

# update indicator_dim data in a dataframe, and change column names
def update_indicator_list(selected_indicators):
    data = []
    for indicator_id in selected_indicators:
        info = wb.series.info(indicator_id)
        data.append(info.items[0])
        
    df = pd.DataFrame(data)
    df.rename(columns={"id":"indicator_id", "value":"indicator_info"}, inplace=True)
    return df