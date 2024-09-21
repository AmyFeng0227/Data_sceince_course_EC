# World Bank Indicator Automation Project
This project automates the process of fetching data from the World Bank API, storing it in a SQL Server database, and creating a Power BI report based on that data. The data is updated daily using Windows Task Scheduler, and the Power BI report is refreshed daily automatically too. 

## Features
 - Fetches World Bank data automatically via API.
 - Process the data and stores it in an SQL Server database.
 - Updates a Power BI report with the latest data. 

## Project files
 - wb_automate.py: The main automation script that runs daily through Windows Task Scheduler. This script fetches and transforms data from the World Bank API and stores it in SQL Server.
 - config.py: This script contains necessary configuration information for SQL Server, as well as the selected_indicator list that can be updated by the user.
 - wb_prepare.ipynb: this jupyter notebook is used to explore the world bank api, and to update the selected_indicator list.
 - worldbank_api.py: This script contains functions to fetch and transform data from the World Bank API. This module handles data cleaning, column renaming, and structuring the data for storage.
 - database.py: This script manages the connection to the SQL Server and handles inserting data into the database using SQLAlchemy.
 - test_pipeline.py: Contains unit tests to verify the functionality of the data pipeline. This includes tests for fetching data, transforming it, and inserting it into SQL Server.
 - process.log: example of how the logging result looks like

## Requirements
 - Python 3.x
 - SQL Server
 - Power BI
 - Python Libraries:
 - pandas
 - wbgapi
 - SQLAlchemy
 - pyodbc

## Usage
Automate the script using Windows Task Scheduler to run daily.
The main script for automation is wb_automate.py.
View the updated Power BI report after the database is refreshed. 
Contact fengshangchanhui@gmail.com for access to PowerBI report.

## Testing
Run the automated tests using pytest
