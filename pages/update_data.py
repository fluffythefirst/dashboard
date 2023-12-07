import dash
from dash import dcc, html, Input, Output, Dash, State, callback, ctx, dash_table
import base64
import datetime
import io

import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np

import os

from cleaning_function import cleaning, update
from dot_ddd_dasc import dot_ddd_dasc
from transform_ddd_v2 import transform_ddd

dash.register_page(__name__, path='/', order = 0)

## -- START HELPER FUNCTIONS ---

def read_and_concat_orders_file():
    '''
    Read from ./orders_data_dump, concat, clean, then save it to ./orders_data_clean
    '''
    path = './orders_data_dump'
    extension = '.csv'

    files = [file for file in os.listdir(path) if file.endswith(extension)]

    # Reading from ./orders_data_dump, clean it, and save it to ./orders_data_clean
    print("read and concat orders file get executed")

    for file in files:
        print("one month of order cleaning file start")
        df = pd.read_csv(os.path.join(path, file))

        # Read and concat
        if os.path.exists('./orders_data_clean/clean.csv'):
            existing_file = pd.read_csv('./orders_data_clean/clean.csv')
            df = cleaning(df)
            df_copy=df.copy(deep=True)
            print("TRANSFORM DDD START")
            ddd = transform_ddd(df_copy)
            print("TRANSFORM DDD END")
            
            df_new = update(existing_file, df)
            df_new.to_csv('./orders_data_clean/clean.csv', index = False, quoting = 1)
            print("update clean.csv.")

        if not os.path.exists('./orders_data_clean/clean.csv'):
            df = cleaning(df)
            df_copy=df.copy(deep=True)
            print("TRANSFORM DDD START")
            ddd = transform_ddd(df_copy)
            print("TRANSFORM DDD END")

            final_df = pd.read_csv('./orders_data_clean/final_df.csv')
            print("reset final_df.csv")
            final_df = final_df[0:0]
            final_df.to_csv('./orders_data_clean/final_df.csv', index = False, quoting = 1)

            print("create clean.csv for the first time.")
            df_copy.to_csv('./orders_data_clean/clean.csv', index = False, quoting = 1)

        print("one month of order cleaning file end")
    
    print("transform dot ddd dasc")
    final_df = dot_ddd_dasc(ddd)

    final_df.to_csv("./orders_data_clean/final_df.csv", index = False, quoting = 1)

    print("finish transform dot ddd dasc")

# Execute read_and_concat_orders_file()

# if os.path.exists('./orders_data_dump'):
#     path = './orders_data_dump'
#     extension = '.csv'
#     files = [file for file in os.listdir(path) if file.endswith(extension)]

#     # If there are more than 1 files
#     if len(files) >= 1:
#         read_and_concat_orders_file()

def read_and_concat_resistance_file():
    path = './resistance_data_dump'
    extension = '.csv'
    files = [file for file in os.listdir(path) if file.endswith(extension)]

    dfs = []
    for file in files:
        df = pd.read_csv(os.path.join(path, file),dtype = {'MRN':str}, low_memory = False)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    return df


def read_and_transform_resistance_data_model():
    print("read_and_transform_resistance_data_model() called ...")

    df = pd.read_csv("./resistance_data_clean/resistance_df.csv", low_memory = False)
    
    # Cleaning
    print("start cleaning ...")
    df['COLLECT_DT_TM'] = pd.to_datetime(df['COLLECT_DT_TM'], format = '%d/%m/%Y %H:%M').dt.date # convert to datetime format
    df['COMPLETE_DT_TM'] = pd.to_datetime(df['COMPLETE_DT_TM'], format = '%d/%m/%Y %H:%M').dt.date # convert to datetime format
    df = df[df['CLIENT'] == 'Westmead Hospital'] # select only records from westmead hospital
    columns_to_keep = ['ORDERABLE','ACCESSION','MRN','DATE_OF_BIRTH','COLLECT_DT_TM','ORGANISM_NAME','CLIENT','NURSE_LOC','ADMITTING_MO','MED_SERVICE','ANTIBIOTIC','INTERP']
    df = df[columns_to_keep] # apply filter selection to df
    df.dropna(subset = ['INTERP'], inplace = True) # drop missing values in INTERP and reset index
    df.reset_index(drop = True, inplace = True)
    df = df[(df['INTERP'] == 'I') | (df['INTERP'] == 'R') | (df['INTERP'] == 'S') | (df['INTERP'] == 'S-DD') | (df['INTERP'] == 'S-IE')] # select only values equal to I, R, S, S-DD, S-IE
    df['ADMITTING_MO'] = df['ADMITTING_MO'].str.replace(r'\s?\(.*\)', '', regex=True) # remove medical officer and specialist medical officer
    condition = (~df['ORDERABLE'].isin(['Blood Culture', 'Culture Urine']))
    replacement = 'Culture Other'
    df['ORDERABLE'] = np.where(condition, replacement, df['ORDERABLE'])
    df = df.drop_duplicates(keep='first')

    print("finished cleaning ...")

    print("start pivot ...")

    pivot_df = df.pivot_table(index=['MRN','DATE_OF_BIRTH','ACCESSION','ORDERABLE','MED_SERVICE','NURSE_LOC','ADMITTING_MO','ORGANISM_NAME'], columns='ANTIBIOTIC', values='INTERP', aggfunc='first')

    # Reset the index if you want 'MRN' and 'ORGANISM_NAME' to be regular columns
    pivot_df.reset_index(inplace=True)

    # Replace NaN values with an empty string or any other desired value
    pivot_df.fillna('', inplace=True)

    # Optional: Rename the columns to remove the name of the index (i.e., 'ANTIBIOTIC')
    pivot_df.columns.name = None

    pivot_df.to_csv("./resistance_data_clean/susceptibility_download.csv", index = False)

    print("finished pivot ...")
    
    df.fillna({'MRN': -1,'ANTIBIOTIC': -1, 'ORGANISM_NAME': -1, 'ORDERABLE': -1, 'NURSE_LOC': -1, 'ADMITTING_MO': -1, 'MED_SERVICE': -1}, inplace=True) #Replace nulls to ensure nothing dropped

    print("start summing the subsceptible, resistance, and intermediate count ...")
    # Sum the subsceptible, resistance, intermediate count
    S = df.groupby(by = ['MRN','ANTIBIOTIC','ORGANISM_NAME','ORDERABLE','NURSE_LOC','ADMITTING_MO','MED_SERVICE','COLLECT_DT_TM'])['INTERP'].apply(lambda x: (x == 'S').sum()).reset_index().rename(columns = {'INTERP':'Count_S'})
    R = df.groupby(by = ['MRN','ANTIBIOTIC','ORGANISM_NAME','ORDERABLE','NURSE_LOC','ADMITTING_MO','MED_SERVICE','COLLECT_DT_TM'])['INTERP'].apply(lambda x: (x == 'R').sum()).reset_index().rename(columns = {'INTERP':'Count_R'})
    I = df.groupby(by = ['MRN','ANTIBIOTIC','ORGANISM_NAME','ORDERABLE','NURSE_LOC','ADMITTING_MO','MED_SERVICE','COLLECT_DT_TM'])['INTERP'].apply(lambda x: (x == 'I').sum()).reset_index().rename(columns = {'INTERP':'Count_I'})
    S_IE = df.groupby(by = ['MRN','ANTIBIOTIC','ORGANISM_NAME','ORDERABLE','NURSE_LOC','ADMITTING_MO','MED_SERVICE','COLLECT_DT_TM'])['INTERP'].apply(lambda x: (x == 'S-IE').sum()).reset_index().rename(columns = {'INTERP':'Count_S-IE'})
    S_DD = df.groupby(by = ['MRN','ANTIBIOTIC','ORGANISM_NAME','ORDERABLE','NURSE_LOC','ADMITTING_MO','MED_SERVICE','COLLECT_DT_TM'])['INTERP'].apply(lambda x: (x == 'S-DD').sum()).reset_index().rename(columns = {'INTERP':'Count_S-DD'})

    print("finished summing the subsceptible, resistance, and intermediate count ...")

    # Return Nan values
    dataframes = [S, R, I, S_IE, S_DD]

    for df in dataframes:
        df.replace(-1, np.nan, inplace=True)

    print("Concat S, R, I dataset into one, and remove duplicate columns ...")

    # Concat S, R, I dataset into one, and remove duplicate columns
    count = pd.concat([S,R,I,S_IE,S_DD], axis = 1).drop_duplicates()
    count = count.loc[:,~count.T.duplicated(keep='first')]
    count['COLLECT_DT_TM'] = pd.to_datetime(count['COLLECT_DT_TM'], format = '%Y-%m-%d')

    print("Finish concat S, R, I dataset into one, and remove duplicate columns ...")

    # Calculate the total of test for each antibiotic in one day
    count['sum'] = count['Count_S'] + count['Count_R'] + count['Count_I'] + count['Count_S-IE'] + count['Count_S-DD']

    def normalize_row(row):
        if row['sum'] > 1:
            if row['Count_R'] >= 1:
                row['sum'] = 1
                row['Count_S'] = 0
                row['Count_I'] = 0
                row['Count_S-IE'] = 0
                row['Count_S-DD'] = 0
            elif row['Count_S-IE'] >= 1:
                row['sum'] = 1
                row['Count_S'] = 0
                row['Count_I'] = 0
                row['Count_R'] = 0
                row['Count_S-DD'] = 0
            elif row['Count_S'] >= 1:
                row['sum'] = 1
                row['Count_R'] = 0
                row['Count_I'] = 0
                row['Count_S-IE'] = 0
                row['Count_S-DD'] = 0
        return row

    print("normalizing rows ...")
    # Apply the function to each row
    count = count.apply(normalize_row, axis=1)

    print("finished normalizing rows ...")

    count.fillna({'MED_SERVICE': 'N/A'}, inplace=True)
    count.sort_values(by='sum', ascending=False)

    count.to_csv("./resistance_data_clean/resistance_count_df.csv", index = False)

    print("read_and_transform_resistance_data_model() finished ...")

    return count.copy(deep = True)

def load_resistance_data_model():
    print("LOAD RESISTANCE DATA MODEL FROM UPDATE_DATA")
    count_df = pd.read_csv("./resistance_data_clean/resistance_count_df.csv")
    count_df['COLLECT_DT_TM'] = pd.to_datetime(count_df['COLLECT_DT_TM'], format = '%Y-%m-%d')

    return count_df.copy(deep = True)

def create_col_df(count_df):
    print("create_col_df() called ...")

    # Get col names
    col_df = count_df.groupby(['ORGANISM_NAME', 'ANTIBIOTIC']).apply(
            lambda x: round((x['Count_S'].sum() / x['sum'].sum()) * 100, 1)
        ).reset_index(name='S_rate')

    # Calculate the total number of tests ('sum') for each combination
    col_df_test = count_df.groupby(['ORGANISM_NAME', 'ANTIBIOTIC'])['sum'].sum().reset_index(name='tests')
        
    # Calculate the total number of tests ('sum') for each combination
    col_df_total_test = count_df.groupby(['ORGANISM_NAME'])['sum'].sum().reset_index(name='total_tests')

    # Merge the susceptibility_rate and test_count DataFrames
    col_merged_df = pd.merge(col_df, col_df_test, on=['ORGANISM_NAME', 'ANTIBIOTIC'], how='left')

    col_df = col_merged_df.pivot_table(
            index='ORGANISM_NAME',
            columns='ANTIBIOTIC',
            values=['S_rate', 'tests'],
            fill_value=0
        )

    col_df.reset_index(inplace=True)

    col_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in col_df.columns]

    col_df = pd.merge(col_df_total_test, col_df, on=['ORGANISM_NAME'], how='left')

    col_df.to_csv("./resistance_data_clean/resistance_col_df.csv", index = False)

    print("create_col_df() finished ...")

    return col_df.copy(deep = True)

# Function to parse the csv content for order csv
def parse_order_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)

    def read_csv_with_multiple_encodings(data, encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']):
        '''
        Try all encoding format to see which one works
        '''
        for encoding in encodings:
            try:
                return pd.read_csv(io.StringIO(data.decode(encoding)))
            
            except UnicodeDecodeError:
                continue

        raise ValueError("None of the provided encodings worked")

    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file

            df = read_csv_with_multiple_encodings(decoded)
            
            extension = '.csv'
            path = './orders_data_dump'

            files = [file for file in os.listdir(path) if file.endswith(extension)]
            file_len = len(files)

            # Ensure directory exists
            if not os.path.exists('./orders_data_dump'):
                os.makedirs('./orders_data_dump')

            # Save the DataFrame to the specified CSV file to ./resistance_data_dump
            df.to_csv('./orders_data_dump/orders_upload_{}.csv'.format(file_len), index = False)

            # Read and concat
            if os.path.exists('./orders_data_clean/clean.csv'):
                existing_file = pd.read_csv('./orders_data_clean/clean.csv')
                df = cleaning(df)
                df_copy=df.copy(deep=True)
                print("TRANSFORM DDD START")
                ddd = transform_ddd(df_copy)
                print("TRANSFORM DDD END")
                
                df_new = update(existing_file, df)
                df_new.to_csv('./orders_data_clean/clean.csv', index = False, quoting = 1)
                print("update clean.csv.")

            if not os.path.exists('./orders_data_clean/clean.csv'):
                df = cleaning(df)
                df_copy=df.copy(deep=True)
                print("TRANSFORM DDD START")
                ddd = transform_ddd(df_copy)
                print("TRANSFORM DDD END")

                final_df = pd.read_csv('./orders_data_clean/final_df.csv')
                print("reset final_df.csv")
                final_df = final_df[0:0]
                final_df.to_csv('./orders_data_clean/final_df.csv', index = False, quoting = 1)

                print("create clean.csv for the first time.")
                df_copy.to_csv('./orders_data_clean/clean.csv', index = False, quoting = 1)
              
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))

    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    try:
        final_df = dot_ddd_dasc(ddd)
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    
    final_df.to_csv("./orders_data_clean/final_df.csv", index = False, quoting = 1)
    
    alert_text = 'Antibiotic order data has been successfully updated'

    return html.Div([
        html.H3(filename),
        html.Div(dash_table.DataTable(
            data = df.to_dict('records'),
            columns = [{'name': i, 'id': i} for i in df.columns],
            page_size = 10
        )),

        html.Div(dbc.Alert(
            html.Div(alert_text, style = {"color": "black"}), 
            color = "primary")
            )
    ])

# Function to parse the csv content for resistance csv
def parse_resistance_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)

    def read_csv_with_multiple_encodings(data, encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']):
        '''
        Try all encoding format to see which one works
        '''
        for encoding in encodings:
            try:
                return pd.read_csv(io.StringIO(data.decode(encoding)))
            
            except UnicodeDecodeError:
                continue

        raise ValueError("None of the provided encodings worked")
    
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = read_csv_with_multiple_encodings(decoded)
            
            # Check file length
            extension = '.csv'
            path = './resistance_data_dump'
            files = [file for file in os.listdir(path) if file.endswith(extension)]
            file_len = len(files)

            # Ensure directory exists
            if not os.path.exists('./resistance_data_dump'):
                os.makedirs('./resistance_data_dump')

            # Save the DataFrame to the specified CSV file to ./resistance_data_dump
            df.to_csv('./resistance_data_dump/resistance_upload_{}.csv'.format(file_len), index = False)

        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    # Read and concat
    df_resistance = read_and_concat_resistance_file()

    # Save to csv
    df_resistance.to_csv("./resistance_data_clean/resistance_df.csv", index = False)

    count_df_transform = read_and_transform_resistance_data_model()

    count_df = load_resistance_data_model()

    col_df_transform = create_col_df(count_df)

    # Save to csv

    alert_text = 'Antibiotic resistance data has been successfully updated'

    return html.Div([
        html.H3(filename),
        html.Div(dash_table.DataTable(
            data = df.to_dict('records'),
            columns = [{'name': i, 'id': i} for i in df.columns],
            page_size = 10
        )),

        html.Div(dbc.Alert(
            html.Div(alert_text, style = {"color": "black"}), 
            color = "primary")
            )
    ])

## -- END HELPER FUNCTIONS ---

## -- START LAYOUT INITIALISATION VARIABLES --

clean_file_exist = os.path.exists('./orders_data_clean/clean.csv')

if clean_file_exist:
    clean_df = pd.read_csv('./orders_data_clean/clean.csv')
    clean_df["ORDER_PLACED_DATE"] = pd.to_datetime(clean_df["ORDER_PLACED_DATE"], format = '%d/%m/%Y %H:%M')
    clean_df['ORDER_MONTH_YEAR'] = clean_df['ORDER_PLACED_DATE'].dt.strftime('%Y-%m')
    unique_month_years = sorted(clean_df['ORDER_MONTH_YEAR'].dropna().unique())
    unique_month_year_formatted = [pd.to_datetime(date).strftime('%B %Y') for date in unique_month_years]

    ao_month_year_table = dash_table.DataTable(
        columns = [{"name": 'Order Month Year', "id": 'ORDER_MONTH_YEAR'}],
        data = [{'ORDER_MONTH_YEAR': date} for date in unique_month_year_formatted],
        page_action = "native",
        page_current = 0,
        page_size = 5,
        style_cell = {'textAlign': 'left','font-family': 'Helvetica Neue', 'fontSize': 14},
        style_header = {
            'backgroundColor': 'rgb(173,216,230)',
            'fontWeight': 'bold'
        },
        id = 'ao-month-year-table-update-data-page'
    )

else:
    ao_month_year_table = dash_table.DataTable(
        columns = [{"name": 'Order Month Year', "id": 'ORDER_MONTH_YEAR'}],
        data = [],  # Empty data
        page_action = "native",
        page_current = 0,
        page_size = 5,
        style_cell = {'textAlign': 'left', 'font-family': 'Helvetica Neue', 'fontSize': 14},
        style_header = {
            'backgroundColor': 'rgb(173,216,230)',
            'fontWeight': 'bold'
        },
        id = 'ao-month-year-table-update-data-page'
    )

resistance_file_exist = os.path.exists('./resistance_data_clean/resistance_df.csv')

if resistance_file_exist:
    resistance_df = pd.read_csv('./resistance_data_clean/resistance_df.csv')
    resistance_df['COLLECT_DT_TM'] = pd.to_datetime(resistance_df['COLLECT_DT_TM'], format = '%d/%m/%Y %H:%M')
    resistance_df['SUSCEPTIBILITY_MONTH_YEAR'] = resistance_df['COLLECT_DT_TM'].dt.strftime('%Y-%m')
    unique_month_years = sorted(resistance_df['SUSCEPTIBILITY_MONTH_YEAR'].dropna().unique())
    unique_month_year_formatted = [pd.to_datetime(date).strftime('%B %Y') for date in unique_month_years]

    susceptibility_month_year_table = dash_table.DataTable(
        columns = [{"name": 'Susceptibility Month Year', "id": 'SUSCEPTIBILITY_MONTH_YEAR'}],
        data = [{'SUSCEPTIBILITY_MONTH_YEAR': date} for date in unique_month_year_formatted],
        page_action = "native",
        page_current = 0,
        page_size = 5,
        style_cell = {'textAlign': 'left','font-family': 'Helvetica Neue', 'fontSize': 14},
        style_header = {
            'backgroundColor': 'rgb(173,216,230)',
            'fontWeight': 'bold'
        },
        id = 'susceptibility-month-year-table-update-data-page'
    )

else:
    susceptibility_month_year_table = dash_table.DataTable(
        columns = [{"name": 'Susceptibility Month Year', "id": 'SUSCEPTIBILITY_MONTH_YEAR'}],
        data = [], # Empty data
        page_action = "native",
        page_current = 0,
        page_size = 5,
        style_cell = {'textAlign': 'left','font-family': 'Helvetica Neue', 'fontSize': 14},
        style_header = {
            'backgroundColor': 'rgb(173,216,230)',
            'fontWeight': 'bold'
        },
        id = 'susceptibility-month-year-table-update-data-page'
    )

## -- END LAYOUT INITIALISATION VARIABLES --

layout = html.Div([

    # Toast for order data and resistance data
    html.Div([
        dbc.Toast(
            "Antibiotic order data has been successfully transformed and saved",
            id = "order-data-positioned-toast",
            header = "Successful",
            is_open = False,
            dismissable = True,
            icon = "success",
            style={"width": 350},
        ),

        dbc.Toast(
            "Suceptibility data has been successfully transformed and saved",
            id = "resistance-data-positioned-toast",
            header = "Successful",
            is_open = False,
            dismissable = True,
            icon = "success",
            style={"width": 350},
        ),

        dbc.Toast(
            "Order data has been succesfuly refreshed",
            id = "order-data-positioned-toast-refresh",
            header = "Successful",
            is_open = False,
            dismissable = True,
            icon = "success",
            style={"width": 350},
        ),

        dbc.Toast(
            "Susceptibility data has been succesfully refreshed",
            id = "susceptibility-data-positioned-toast-refresh",
            header = "Successful",
            is_open = False,
            dismissable = True,
            icon = "success",
            style={"width": 350},
        ),
        
    ], 
        style = {"position": "fixed", "top": 5, "right": 5, "width": 350, "zIndex": 1}
    ),
    html.H3("Upload Order Data"),

    # Order data upload
    dcc.Upload(
        id = 'upload-order-data',
        children = html.Div([
            'Drag and Drop Antibiotic Order Data or ',
            html.A('Select Antibiotic Order Data File')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'z-index': 1
        },
        # Allow multiple files to be uploaded
        # Do not set multiple to false. If you set it to false, the parse function will work differently
        multiple = True
    ),
    
    # Show output order data
    html.Div(id='output-order-data-upload',  style={
            'width': '100%'
        }),


    # Order data download df
    dcc.Download(id = "download-cleaned-order-data-df"),

    html.Br(),
    html.Label("Uploaded Order Data Records:"),
    html.Br(),
    ao_month_year_table,

    # Order data refresh text
    html.Div([
        html.Br(),
        html.Label("Order data has been successfully refreshed."),
        html.Br()], 
        id = "order-data-refresh-text-update-data-page",
        style = {"display": "none"}
    ),

    # Order data download button and refresh
    dbc.Row([
        dbc.Col(dbc.Button("Download cleaned order data csv", id = "download-cleaned-order-data-button"), width = "auto"),
        dbc.Col(dbc.Button("Refresh order data", id = "refresh-order-data-button-update-data-page"), width = "auto")
        ], 
        justify = "center"
    ),

    html.Br(),

    html.H3("Upload Susceptibility Data"),

    # Resistance data upload
    dcc.Upload(
        id = 'upload-resistance-data',
        children = html.Div([
            'Drag and Drop Antibiotic Susceptibility Data or ',
            html.A('Select Antibiotic Susceptibility Data File')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'z-index': 1
        },
        # Allow multiple files to be uploaded
        # Do not set multiple to false. If you set it to false, the parse function will work differently
        multiple = True
    ),

    # Show output resistance data
    html.Div(id='output-resistance-data-upload',  style={
            'width': '100%'
    }),

    # Resistance data download df
    dcc.Download(id = "download-cleaned-resistance-df"),

    html.Br(),
    html.Label("Uploaded Susceptibility Data Records:"),
    html.Br(),
    susceptibility_month_year_table,

    # Susceptibility data refresh text
    html.Div([
        html.Br(),
        html.Label("Susceptibility data has been successfully refreshed."),
        html.Br()], 
        id = "susceptibility-data-refresh-text-update-data-page",
        style = {"display": "none"}
    ),

    # Resistance data download button and refresh
    dbc.Row([
            dbc.Col(dbc.Button("Download cleaned resistance data csv", id = "download-cleaned-resistance-data-button"), width = "auto"),
            dbc.Col(dbc.Button("Refresh susceptibility data", id = "refresh-susceptibility-data-button-update-data-page"), width = "auto")
            ], 
            justify = "center"
    ),
    html.Br(),
])

# Callback for download functionality
@callback (
        Output('download-cleaned-order-data-df', 'data'),
        Output('download-cleaned-resistance-df', 'data'),
        Input('download-cleaned-order-data-button', 'n_clicks'),
        Input('download-cleaned-resistance-data-button', 'n_clicks')
)

def update_download_function(
                             download_cleaned_order_data_n,
                             download_cleaned_resistance_data_n
                             ):
    cleaned_order_data_df = pd.read_csv("./orders_data_clean/final_df.csv")

    cleaned_resistance_data_df = pd.read_csv("./resistance_data_clean/susceptibility_download.csv")

    download_cleaned_order_data_val = dash.no_update
    if "download-cleaned-order-data-button" == ctx.triggered_id:
        download_cleaned_order_data_val = dcc.send_data_frame(cleaned_order_data_df.to_csv, "download-cleaned-order-data.csv", index = False)
    else:
        download_cleaned_order_data_val = dash.no_update
    
    download_cleaned_resistance_data_val = dash.no_update
    if "download-cleaned-resistance-data-button" == ctx.triggered_id:
        download_cleaned_order_data_val = dcc.send_data_frame(cleaned_resistance_data_df.to_csv, "download-cleaned-susceptibility-data.csv", index = False)
    else:
        download_cleaned_resistance_data_val = dash.no_update

    return download_cleaned_order_data_val, download_cleaned_resistance_data_val


# Callback for antibiotic order data upload

@callback(Output('output-order-data-upload', 'children'),
          Output('memory', 'data'),
          Output("order-data-positioned-toast", "is_open"),
           Input('upload-order-data', 'contents'),
           State('upload-order-data', 'filename'),
           State('upload-order-data', 'last_modified'))

def update_order_data_output(list_of_contents, list_of_names, list_of_dates):
    '''
        Currently, value is always set to "dataset-uploaded," so every time we navigate to another page, 
        it reads the most recent dataframe. This does not affect the performance of the dashboard so far.
    '''
    
    is_open = False

    value = "dataset-uploaded"

    if list_of_contents is None:
        return dash.no_update

    if list_of_contents is not None:
        is_open = True

        children = [
            parse_order_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]

        return children, value, is_open
    

# Callback for antibiotic resistance data upload

@callback(Output('output-resistance-data-upload', 'children'),
          Output('memory2', 'data'),
          Output("resistance-data-positioned-toast", "is_open"),
           Input('upload-resistance-data', 'contents'),
           State('upload-resistance-data', 'filename'),
           State('upload-resistance-data', 'last_modified'))

def update_resistance_data_output(list_of_contents, list_of_names, list_of_dates):
    '''
        Currently, value is always set to "dataset-uploaded," so every time we navigate to another page, 
        it reads the most recent dataframe. This does not affect the performance of the dashboard so far.
    '''
    
    is_open = False

    value = "dataset-uploaded"

    if list_of_contents is None:
        return dash.no_update

    if list_of_contents is not None:
        is_open = True

        children = [
            parse_resistance_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]

        return children, value, is_open

# Callback when user click refresh orders button it will: 
# go through all orders_data_dump -> clean -> transform -> visualise

@callback(
    Output('order-data-positioned-toast-refresh', 'is_open'),
    Output('order-data-refresh-text-update-data-page', 'style'),
    Input('refresh-order-data-button-update-data-page', 'n_clicks')
)

def update_refresh_order_data(list_of_contents):

    is_open = False

    display_style = {"display": "none"}

    if "refresh-order-data-button-update-data-page" == ctx.triggered_id:
        read_and_concat_orders_file()
        is_open = True
        display_style = {"display": "block"}

    
    return is_open, display_style
    

# Callback when user click refresh susceptibility button it will: 
# go through all resistance_data_dump -> clean -> transform -> visualise

@callback(
    Output('susceptibility-data-positioned-toast-refresh', 'is_open'),
    Output('susceptibility-data-refresh-text-update-data-page', 'style'),
    Input('refresh-susceptibility-data-button-update-data-page', 'n_clicks')
)

def update_refresh_susceptibility_data(list_of_contents):

    is_open = False
    
    display_style = {"display": "none"}

    if "refresh-susceptibility-data-button-update-data-page" == ctx.triggered_id:

        df_resistance = read_and_concat_resistance_file()

        # Save to csv
        df_resistance.to_csv("./resistance_data_clean/resistance_df.csv", index = False)
        count_df_transform = read_and_transform_resistance_data_model()

        count_df = load_resistance_data_model()
        col_df_transform = create_col_df(count_df)
        is_open = True
        display_style = {"display": "block"}
    
    return is_open, display_style

# Callback to check if order dataset is uploaded, refreshed is clicked, or if we are in the home page
#  then update the table for orders

@callback(
    Output('ao-month-year-table-update-data-page', 'data'),
    Input('memory', 'data'),
    Input('refresh-order-data-button-update-data-page', 'n_clicks'),
    Input('url', 'pathname')
)

def update_ao_month_year_table_from_upload(data, refresh_n_clicks, pathname):

    # Check if clean exist
    clean_file_exist = os.path.exists('./orders_data_clean/clean.csv')

    if clean_file_exist:
        if data == "dataset-uploaded" or ctx.triggered_id == 'refresh-order-data-button-update-data-page' or pathname == '/':
            clean_df = pd.read_csv('./orders_data_clean/clean.csv')
            clean_df["ORDER_PLACED_DATE"] = pd.to_datetime(clean_df["ORDER_PLACED_DATE"], format = '%d/%m/%Y %H:%M')
            clean_df['ORDER_MONTH_YEAR'] = clean_df['ORDER_PLACED_DATE'].dt.strftime('%Y-%m')
            unique_month_years = sorted(clean_df['ORDER_MONTH_YEAR'].dropna().unique())
            unique_month_year_formatted = [pd.to_datetime(date).strftime('%B %Y') for date in unique_month_years]
            
            new_data = [{'ORDER_MONTH_YEAR': date} for date in unique_month_year_formatted]
        
            return new_data

# Callback to check if susceptibility dataset is uploaded, refreshed is clicked, or if we are in the home page
#  then update the table for susceptibility

@callback(
    Output('susceptibility-month-year-table-update-data-page', 'data'),
    Input('memory2', 'data'),
    Input('refresh-susceptibility-data-button-update-data-page', 'n_clicks'),
    Input('url', 'pathname')    
)

def update_susceptibility_month_year_table_from_upload(data, refresh_n_clicks, pathname):
    if data == "dataset-uploaded" or ctx.triggered_id == 'refresh-susceptibility-data-button-update-data-page' or pathname == '/':
        resistance_df = pd.read_csv('./resistance_data_clean/resistance_df.csv')
        resistance_df['COLLECT_DT_TM'] = pd.to_datetime(resistance_df['COLLECT_DT_TM'], format = '%d/%m/%Y %H:%M')
        resistance_df['SUSCEPTIBILITY_MONTH_YEAR'] = resistance_df['COLLECT_DT_TM'].dt.strftime('%Y-%m')
        unique_month_years = sorted(resistance_df['SUSCEPTIBILITY_MONTH_YEAR'].dropna().unique())
        unique_month_year_formatted = [pd.to_datetime(date).strftime('%B %Y') for date in unique_month_years]

        new_data = [{'SUSCEPTIBILITY_MONTH_YEAR': date} for date in unique_month_year_formatted]

        return new_data