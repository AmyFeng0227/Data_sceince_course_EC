import pytest
import pandas as pd
from unittest.mock import patch, MagicMock 
from worldbank_api import fetch_and_transform
from database import save_to_sql

@patch('worldbank_api.wb.data.fetch') 
def test_fetch_and_transform_success(mock_fetch):  
    
    mock_fetch.return_value = [
        {'time': 'YR2011', 'series': 'SL.TLF.0714.ZS', 'economy': 'USA', 'aggregate': None},
        {'time': 'YR2012', 'series': 'SI.POV.MPUN', 'economy': 'USA', 'aggregate': None}
    ]

    df = fetch_and_transform()

    assert list(df.columns) == ['year', 'indicator_id', 'country_code']
    assert df.iloc[0]['year'] == 2011
    assert df.iloc[0]['indicator_id'] == 'SL.TLF.0714.ZS'
    
    
@patch('database.create_engine')
def test_save_to_sql_success(mock_engine):
    mock_engine.return_value = MagicMock()
    
    df = pd.DataFrame({
        'indicator_id': ['SL.WAG.0714.ZS'],
        'indicator_info': ['Children in employment, wage workers']
    })
    
    try:
        save_to_sql(dataframe_name=df, table_name="indicator_dim")
    except Exception:
        pytest.fail("save_to_sql raised an Exception!")
        

@patch('database.create_engine')
def test_save_to_sql_fail(mock_engine):
    #simulating a failure in database connection
    mock_engine.side_effect = Exception("Database connection failure!!!!")
    
    df = pd.DataFrame({
        'indicator_id': ['SL.WAG.0714.ZS'],
        'indicator_info': ['Children in employment, wage workers']
    })
    
    with pytest.raises(Exception):
        save_to_sql(dataframe_name=df, table_name="indicator_dim")