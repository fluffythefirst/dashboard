import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, Dash, State, callback, ctx, dash_table
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import plotly.express as px
import plotly.io as pio

import re

class DataModel:
    def __init__(self, csv_read, order_date_format):
        self.csv_read = csv_read
        self.order_date_format = order_date_format
    
    def read_and_transform_agg_model(self):
        df = pd.read_csv(self.csv_read)
        df.rename(columns={'LOCATION_OF_PATIENT_AT_THE_TIME_OF_ORDER': 'WARD'}, inplace=True)
        df.rename(columns={'ATTENDING_MEDICAL_OFFICER': 'DOCTOR'}, inplace=True)
        df.rename(columns={'total_DDD': 'TOTAL_DDD'}, inplace=True)
        df.rename(columns={'total_dosage': 'TOTAL_DOSAGE'}, inplace=True)
        df.rename(columns={'ORDER_PLACED_DATE': 'ORDER_DATE'}, inplace=True)
        df.rename(columns={'DAYS_OF_THERAPY': 'TOTAL_DOT'}, inplace=True)
        
        df['DOCTOR'] = df['DOCTOR'].str.replace(r'\s*\([^)]*\)$', '', regex=True)
        
        filtered_df = df.loc[:, ['MRN', 'ORDER_DATE', 'ORDER_STATUS','ORDER_GENERIC','MEDICAL_SERVICE','WARD','AMS_INDICATION','DOCTOR','TOTAL_DDD','TOTAL_DOSAGE','TOTAL_DOT', 'AGE', 'ACTUAL_WEIGHT', 'dASC', 'PATIENT_NAME']] # select relevant attributes only
        
        filtered_df['ORDER_DATE'] = pd.to_datetime(filtered_df['ORDER_DATE'], format = self.order_date_format)
         
        filtered_df['ORDER_MONTH_YEAR'] = pd.to_datetime(filtered_df['ORDER_DATE']).dt.strftime('%Y-%m')

        return filtered_df.copy(deep = True)
