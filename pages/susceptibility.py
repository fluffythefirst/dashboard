import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, Dash, State, callback, ctx, dash_table
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import os

import re

#
# This is for https://community.plotly.com/t/datetimeproperties-to-pydatetime-is-deprecated/78293
# The problem is from plotly backend
#

import warnings
warnings.simplefilter("ignore", category=FutureWarning)


dash.register_page(__name__, order = 4)

pio.templates.default = "plotly_white"

## -- START READ AND CONCAT RESISTANCE DATASET

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

## -- END READ AND CONCAT RESISTANCE DATASET

df_resistance = read_and_concat_resistance_file()

# Save to csv
df_resistance.to_csv("./resistance_data_clean/resistance_df.csv", index = False)

# -- START RESISTANCE DATA MODEL

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

    df.fillna({'MRN': -1,'ACCESSION': -1,'DATE_OF_BIRTH': -1,'ANTIBIOTIC': -1, 'ORGANISM_NAME': -1, 'ORDERABLE': -1, 'NURSE_LOC': -1, 'ADMITTING_MO': -1, 'MED_SERVICE': -1}, inplace=True)
    print("finished cleaning ...")

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

## -- END RESISTANCE DATA MODEL

## - START LAYOUT HELPER FUNCTIONS

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

def load_resistance_data_model():
    # print("LOAD RESISTANCE DATA MODEL FROM RESISTANCE")
    count_df = pd.read_csv("./resistance_data_clean/resistance_count_df.csv")
    count_df['COLLECT_DT_TM'] = pd.to_datetime(count_df['COLLECT_DT_TM'], format = '%Y-%m-%d')

    return count_df.copy(deep = True)

def load_col_df():
    col_df = pd.read_csv("./resistance_data_clean/resistance_col_df.csv")

    return col_df.copy(deep = True)

def generate_style_data_conditional_logic_for_antibiogram(col_df):
    style_data_conditional_for_antibiogram = [
        {
            'if': {
                'filter_query': '{{{col}}} > 0 && {{{col}}} < 70'.format(col=col),
                'column_id': col,
            },
            'backgroundColor': 'red',
            'color': 'white',
        }
        for col in col_df.columns if 's_rate' in col.lower()
    ] + [
        {
            'if': {
                'filter_query': '{{{col}}} >= 70 && {{{col}}} < 90'.format(col=col),
                'column_id': col,
            },
            'backgroundColor': 'yellow',
            'color': 'black',
        }
        for col in col_df.columns if 's_rate' in col.lower()
    ] + [
        {
            'if': {
                'filter_query': '{{{col}}} >= 90'.format(col=col),
                'column_id': col,
            },
            'backgroundColor': 'green',
            'color': 'white',
        }
        for col in col_df.columns if 's_rate' in col.lower()
    ]+ [
        {
            'if': {
                'filter_query': '{{{col}}} > 0 && {{{col}}} < 30'.format(col=col),
                'column_id': col,
            },
            'backgroundColor': 'orange',
            'color': 'black',
        }
        for col in col_df.columns if 'tests' in col.lower()
    ]

    return style_data_conditional_for_antibiogram    


## - END LAYOUT HELPER FUNCTIONS

## -- START LAYOUT INITIALISATION VARIABLES

print("START LAYOUT INITIALISATION VARIABLES")
count_df_transform = read_and_transform_resistance_data_model()
print("count_df outside function called ...")

count_df = load_resistance_data_model()

min_date = min(count_df['COLLECT_DT_TM'])
max_date = max(count_df['COLLECT_DT_TM'])

unique_orderable = sorted(count_df['ORDERABLE'].dropna().unique())
unique_med_service = sorted(count_df['MED_SERVICE'].dropna().unique())
unique_antibiotic = sorted(count_df['ANTIBIOTIC'].dropna().unique())
unique_nurse_locs = sorted(count_df['NURSE_LOC'].dropna().unique())
unique_mo = sorted(count_df['ADMITTING_MO'].dropna().unique())
unique_organism = sorted(count_df['ORGANISM_NAME'].unique())

col_df_transform = create_col_df(count_df)

col_df = load_col_df()

print("col_df outside function called ...")

# start dedup_data and overall_rates_org
# dedup_data and overall_dates not used?

dedup_data = count_df.sort_values(by='COLLECT_DT_TM').drop_duplicates(
        subset=['MRN', 'ANTIBIOTIC', 'ORGANISM_NAME'],
        keep='first'
    )

overall_rates_org = dedup_data.groupby('ORGANISM_NAME').apply(
    lambda x: round((x['Count_S'].sum() + x['Count_S-IE'].sum() + x['Count_S-DD'].sum()) / x['sum'].sum() * 100)
).reset_index(name='Overall Susceptibility Rate')

# end dedup_data and overall_rates_org

print("END LAYOUT INITIALISATION VARIABLES")
print("FINISHED LOADING")

## -- END LAYOUT INITIALISATION VARIABLES

## -- START LAYOUT

layout = html.Div([
    html.Div(id='intermediate-div-resistance-page', style={'display': 'none'}),
    html.H1('Susceptibility Rates and Antibiogram Datatable'),
    
    html.Br(),
    
    html.Div([
        dbc.Row(
            [
                dbc.Col('Select a date range to filter', style = {'font-weight': 'bold'}),
                dbc.Col(
                    dcc.DatePickerRange(
                        id = 'date-range-picker',
                        start_date = min_date,
                        end_date = max_date,
                        display_format = 'YYYY-MM-DD'
                    )
            )
          ]
        )   
    ]),
    
    html.Div([
    dcc.Dropdown(
        id='test-filter',
        options=[
            {'label': 'All', 'value': 'All'},
            {'label': '30+ Tests Required', 'value': '30_plus_tests'},
        ],
        value='30_plus_tests',  # Set the default value to '30+ Tests Required'
    ),
    ]),
    
    # Dropdowns in a single row
    dbc.Row([
        # Orderable Dropdown
        dbc.Col([
            html.Label('Orderable Dropdown'),
            dcc.Dropdown(
                id='orderable-dropdown',
                options=[{'label': loc, 'value': loc} for loc in unique_orderable],
                multi=True
            ),
        ]),

        # Med Service Dropdown
        dbc.Col([
            html.Label('Med Service Dropdown'),
            dcc.Dropdown(
                id='med-service-dropdown',
                options=[{'label': loc, 'value': loc} for loc in unique_med_service],
                multi=True
            ),
        ]),

        # Antibiotic Dropdown
        dbc.Col([
            html.Label('Antibiotic Dropdown'),
            dcc.Dropdown(
                id='antibiotic-dropdown',
                options=[{'label': loc, 'value': loc} for loc in unique_antibiotic],
                multi=True
            ),
        ]),

        # Organism Dropdown
        dbc.Col([
            html.Label('Organism Dropdown'),
            dcc.Dropdown(
                id='organism-dropdown',
                options=[{'label': loc, 'value': loc} for loc in unique_organism],
                style={'width': '300px'},
                multi=True
            ),
        ]),

        # Nurse Loc Dropdown
        dbc.Col([
            html.Label('Nurse Loc Dropdown'),
            dcc.Dropdown(
                id='nurse-loc-dropdown',
                options=[{'label': loc, 'value': loc} for loc in unique_nurse_locs],
                multi=True
            ),
        ]),

        # MO Dropdown
        dbc.Col([
            html.Label('MO Dropdown'),
            dcc.Dropdown(
                id='mo-dropdown',
                options=[{'label': loc, 'value': loc} for loc in unique_mo],
                multi=True
            ),
        ]),
    ]),

    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id = 'antibiotic-chart',
                config = {'displayModeBar': False}
                ),
        ]),
        dbc.Col([
            dcc.Graph(
                id='organism-chart',
                config={'displayModeBar': False}
            ),
        ]),
    ]),
    # DataTable to display filtered data

    dash_table.DataTable(
        id='antibiogram-table',
        columns=[
            {"name": 'ORGANISM_NAME', "id": 'ORGANISM_NAME', "deletable": True, "selectable": True},
            # Include columns for ANTIBIOTIC values here
        ],
        data=[],  # Initialize with an empty list
        filter_action="native",
        sort_action='native',
        page_size=30,
        style_data_conditional= generate_style_data_conditional_logic_for_antibiogram(col_df = col_df)   
   ),
    html.Br(),

    html.Label("Time Period Filter"),
    dcc.Dropdown(
        id='aggregation-unit-dropdown',
        options=[
            {'label': 'Daily', 'value': 'D'},
            {'label': 'Weekly', 'value': 'W'},
            {'label': 'Monthly', 'value': 'M'},
            {'label': 'Quarterly', 'value': 'Q'},
            {'label': 'Yearly', 'value': 'Y'},
        ],
        value='W'  # Default to weekly aggregation
    ),

    # Line chart for resistance rate over time
    dcc.Graph(id='line-chart')
])

## -- END LAYOUT

## START INITIALISATION CALLBACK FUNCTION

@callback(
    # Initialise min date and max date
    Output('date-range-picker', 'min_date_allowed'),
    Output('date-range-picker','max_date_allowed'),

    # Initialise antibiogram table
    Output('antibiogram-table', 'style_data_conditional'),

    # Trigger other callback to initialise the charts
    Output('intermediate-div-resistance-page', 'children'),

    # From memory
    Input('memory2', 'data')
)

def refresh_and_initialize_resistance_data(data):
    
    # Initialise variables
    #count_df = None
    col_df = None

    min_date = dash.no_update
    max_date = dash.no_update
    
    antibiogram_table_style_data_conditional = dash.no_update

    intermediate_data = dash.no_update
    print("refresh_and_initialize_resistance_data() called ...")

    if data == "dataset-uploaded":
        print("dataset-uploaded from resistance page ...")
        # print("refresh_and_initialize_resistance_data() called ...")

        # count_df = read_and_transform_resistance_data_model()
        
        # min_date = min(count_df['COLLECT_DT_TM'])
        # max_date = max(count_df['COLLECT_DT_TM'])

        # col_df = create_col_df(count_df)

        # antibiogram_table_style_data_conditional = \
        #     generate_style_data_conditional_logic_for_antibiogram(col_df = col_df)
        
        # intermediate_data = str(pd.Timestamp.now())

    return min_date, max_date, \
           antibiogram_table_style_data_conditional, \
           intermediate_data

## END INITIALISATION CALLBACK FUNCTION

# Callback for Antibiotic Dropdown

@callback(
    Output('antibiotic-dropdown', 'options'),
    Input('orderable-dropdown', 'value'),
    Input('med-service-dropdown', 'value'),
    Input('nurse-loc-dropdown', 'value'),
    Input('mo-dropdown', 'value'),
    Input('organism-dropdown', 'value'),
    # Coming from the refresh_and_initialize_resistance_data() function
    Input('intermediate-div-resistance-page', 'children')
)

def update_antibiotic_options(
                              selected_orderable, 
                              selected_med_service, 
                              selected_nurse_loc, 
                              selected_mo,
                              selected_organism,
                              intermediate_data
                              ):
    
    # print("Callback for Antibiotic Dropdown is called")
    
    filtered_df = load_resistance_data_model()

    if selected_orderable:
        if isinstance(selected_orderable, str):  # Check if it's a single string
            selected_orderable = [selected_orderable]  # Convert to a list
        filtered_df = filtered_df[filtered_df['ORDERABLE'].isin(selected_orderable)]

    if selected_med_service:
        if isinstance(selected_med_service, str):
            selected_med_service = [selected_med_service]
        filtered_df = filtered_df[filtered_df['MED_SERVICE'].isin(selected_med_service)]

    if selected_nurse_loc:
        if isinstance(selected_nurse_loc, str):
            selected_nurse_loc = [selected_nurse_loc]
        filtered_df = filtered_df[filtered_df['NURSE_LOC'].isin(selected_nurse_loc)]

    if selected_mo:
        if isinstance(selected_mo, str):
            selected_mo = [selected_mo]
        filtered_df = filtered_df[filtered_df['ADMITTING_MO'].isin(selected_mo)]
    
    if selected_organism:
        if isinstance(selected_organism, str):
            selected_organism = [selected_organism]
        filtered_df = filtered_df[filtered_df['ORGANISM_NAME'].isin(selected_organism)]


    filtered_options = sorted(filtered_df['ANTIBIOTIC'].dropna().unique())
    options = [{'label': option, 'value': option} for option in filtered_options]

    return options


# Callback for Organism Dropdown

@callback(
    Output('organism-dropdown', 'options'),
    Input('orderable-dropdown', 'value'),
    Input('med-service-dropdown', 'value'),
    Input('nurse-loc-dropdown', 'value'),
    Input('antibiotic-dropdown', 'value'),
    Input('mo-dropdown', 'value'),
    # Coming from the refresh_and_initialize_resistance_data() function
    Input('intermediate-div-resistance-page', 'children')
)
def update_organism_options(
                            selected_orderable, 
                            selected_med_service, 
                            selected_nurse_loc, 
                            selected_antibiotic, 
                            selected_mo,
                            intermediate_data
                            ):
    
    filtered_df = load_resistance_data_model()

    if selected_orderable:
        if isinstance(selected_orderable, str):  # Check if it's a single string
            selected_orderable = [selected_orderable]  # Convert to a list
        filtered_df = filtered_df[filtered_df['ORDERABLE'].isin(selected_orderable)]

    if selected_med_service:
        if isinstance(selected_med_service, str):
            selected_med_service = [selected_med_service]
        filtered_df = filtered_df[filtered_df['MED_SERVICE'].isin(selected_med_service)]

    if selected_nurse_loc:
        if isinstance(selected_nurse_loc, str):
            selected_nurse_loc = [selected_nurse_loc]
        filtered_df = filtered_df[filtered_df['NURSE_LOC'].isin(selected_nurse_loc)]
    
    if selected_antibiotic:
        if isinstance(selected_antibiotic, str):
            selected_antibiotic = [selected_antibiotic]
        filtered_df = filtered_df[filtered_df['ANTIBIOTIC'].isin(selected_antibiotic)]

    if selected_mo:
        if isinstance(selected_mo, str):
            selected_mo = [selected_mo]
        filtered_df = filtered_df[filtered_df['ADMITTING_MO'].isin(selected_mo)]

    filtered_options = sorted(filtered_df['ORGANISM_NAME'].unique())
    options = [{'label': option, 'value': option} for option in filtered_options]

    return options


# Callback for Nurse Loc Dropdown

@callback(
    Output('nurse-loc-dropdown', 'options'),
    Input('orderable-dropdown', 'value'),
    Input('med-service-dropdown', 'value'),
    Input('antibiotic-dropdown', 'value'),
    Input('mo-dropdown', 'value'),
    Input('organism-dropdown', 'value'),
    # Coming from the refresh_and_initialize_resistance_data() function
    Input('intermediate-div-resistance-page', 'children')
)

def update_nurse_loc_options(
                             selected_orderable, 
                             selected_med_service, 
                             selected_antibiotic, 
                             selected_mo, 
                             selected_organism,
                             intermediate_data
                             ):
    
    filtered_df = load_resistance_data_model()

    if selected_orderable:
        if isinstance(selected_orderable, str):
            selected_orderable = [selected_orderable]
        filtered_df = filtered_df[filtered_df['ORDERABLE'].isin(selected_orderable)]

    if selected_med_service:
        if isinstance(selected_med_service, str):
            selected_med_service = [selected_med_service]
        filtered_df = filtered_df[filtered_df['MED_SERVICE'].isin(selected_med_service)]

    if selected_antibiotic:
        if isinstance(selected_antibiotic, str):
            selected_antibiotic = [selected_antibiotic]
        filtered_df = filtered_df[filtered_df['ANTIBIOTIC'].isin(selected_antibiotic)]

    if selected_mo:
        if isinstance(selected_mo, str):
            selected_mo = [selected_mo]
        filtered_df = filtered_df[filtered_df['ADMITTING_MO'].isin(selected_mo)]
    
    if selected_organism:
        if isinstance(selected_organism, str):
            selected_organism = [selected_organism]
        filtered_df = filtered_df[filtered_df['ORGANISM_NAME'].isin(selected_organism)]

    filtered_options = sorted(filtered_df['NURSE_LOC'].dropna().unique())
    options = [{'label': option, 'value': option} for option in filtered_options]

    return options

# Callback for MO Dropdown
@callback(
    Output('mo-dropdown', 'options'),
    Input('orderable-dropdown', 'value'),
    Input('med-service-dropdown', 'value'),
    Input('antibiotic-dropdown', 'value'),
    Input('nurse-loc-dropdown', 'value'),
    Input('organism-dropdown', 'value'),
    # Coming from the refresh_and_initialize_resistance_data() function
    Input('intermediate-div-resistance-page', 'children')
)
def update_mo_options(
                     selected_orderable, 
                     selected_med_service, 
                     selected_antibiotic, 
                     selected_nurse_loc, 
                     selected_organism,
                      intermediate_data
                      ):
    filtered_df = load_resistance_data_model()

    if selected_orderable:
        if isinstance(selected_orderable, str):
            selected_orderable = [selected_orderable]
        filtered_df = filtered_df[filtered_df['ORDERABLE'].isin(selected_orderable)]

    if selected_med_service:
        if isinstance(selected_med_service, str):
            selected_med_service = [selected_med_service]
        filtered_df = filtered_df[filtered_df['MED_SERVICE'].isin(selected_med_service)]

    if selected_antibiotic:
        if isinstance(selected_antibiotic, str):
            selected_antibiotic = [selected_antibiotic]
        filtered_df = filtered_df[filtered_df['ANTIBIOTIC'].isin(selected_antibiotic)]

    if selected_nurse_loc:
        if isinstance(selected_nurse_loc, str):
            selected_nurse_loc = [selected_nurse_loc]
        filtered_df = filtered_df[filtered_df['NURSE_LOC'].isin(selected_nurse_loc)]
    
    if selected_organism:
        if isinstance(selected_organism, str):
            selected_organism = [selected_organism]
        filtered_df = filtered_df[filtered_df['ORGANISM_NAME'].isin(selected_organism)]


    filtered_options = sorted(filtered_df['ADMITTING_MO'].dropna().unique())
    options = [{'label': option, 'value': option} for option in filtered_options]

    return options

# Callback for Med Service Dropdown
@callback(
    Output('med-service-dropdown', 'options'),
    Input('orderable-dropdown', 'value'),
    Input('antibiotic-dropdown', 'value'),
    Input('nurse-loc-dropdown', 'value'),
    Input('mo-dropdown', 'value'),
    Input('organism-dropdown', 'value'),
    # Coming from the refresh_and_initialize_resistance_data() function
    Input('intermediate-div-resistance-page', 'children')
)
def update_med_service_options(
                               selected_orderable, 
                               selected_antibiotic, 
                               selected_nurse_loc, 
                               selected_mo, 
                               selected_organism,
                               intermediate_data
                               ):
    
    filtered_df = load_resistance_data_model()

    if selected_orderable:
        if isinstance(selected_orderable, str):
            selected_orderable = [selected_orderable]
        filtered_df = filtered_df[filtered_df['ORDERABLE'].isin(selected_orderable)]

    if selected_antibiotic:
        if isinstance(selected_antibiotic, str):
            selected_antibiotic = [selected_antibiotic]
        filtered_df = filtered_df[filtered_df['ANTIBIOTIC'].isin(selected_antibiotic)]

    if selected_nurse_loc:
        if isinstance(selected_nurse_loc, str):
            selected_nurse_loc = [selected_nurse_loc]
        filtered_df = filtered_df[filtered_df['NURSE_LOC'].isin(selected_nurse_loc)]

    if selected_mo:
        if isinstance(selected_mo, str):
            selected_mo = [selected_mo]
        filtered_df = filtered_df[filtered_df['ADMITTING_MO'].isin(selected_mo)]
    
    if selected_organism:
        if isinstance(selected_organism, str):
            selected_organism = [selected_organism]
        filtered_df = filtered_df[filtered_df['ORGANISM_NAME'].isin(selected_organism)]


    filtered_options = sorted(filtered_df['MED_SERVICE'].dropna().unique())
    options = [{'label': option, 'value': option} for option in filtered_options]

    return options

@callback(
    Output('line-chart', 'figure'),
    Input('antibiotic-dropdown', 'value'),
    Input('orderable-dropdown', 'value'),
    Input('med-service-dropdown', 'value'),
    Input('nurse-loc-dropdown', 'value'),
    Input('mo-dropdown', 'value'),
    Input('organism-dropdown', 'value'),
    Input('date-range-picker', 'start_date'),
    Input('date-range-picker', 'end_date'),
    Input('aggregation-unit-dropdown', 'value')
)
def update_line_chart(selected_antibiotic, selected_orderable, selected_med_service, selected_nurse_locs, selected_mo, selected_organism, start_date, end_date,selected_aggregation_unit):
    
    # print("Callback for Update line chart is called")
    if not selected_antibiotic:
        # Handle case where no antibiotics are selected
#         print("No antibiotics selected")
        return {'data': [], 'layout': {}}
    
    filtered_df = load_resistance_data_model()

    if selected_orderable:
        filtered_df = filtered_df[filtered_df['ORDERABLE'].isin(selected_orderable)]
    
    if selected_antibiotic:
        filtered_df = filtered_df[filtered_df['ANTIBIOTIC'].isin(selected_antibiotic)]

    if selected_med_service:
            filtered_df = filtered_df[filtered_df['MED_SERVICE'].isin(selected_med_service)]
    
    if selected_nurse_locs:
        filtered_df = filtered_df[filtered_df['NURSE_LOC'].isin(selected_nurse_locs)]
    
    if selected_mo:
        filtered_df = filtered_df[filtered_df['ADMITTING_MO'].isin(selected_mo)]

    if selected_organism:
        filtered_df = filtered_df[filtered_df['ORGANISM_NAME'].isin(selected_organism)]
   
    
    # Filter the DataFrame based on the selected date range

    filtered_df = filtered_df[(filtered_df['COLLECT_DT_TM'] >= start_date) & (filtered_df['COLLECT_DT_TM'] <= end_date)]

    if filtered_df.empty:
        # Handle case where no data matches the filter criteria
#         print("No data matches the filter criteria")
        return {'data': [], 'layout': {}}

    # Deduplicate the filtered data
    deduplicated_data = filtered_df.sort_values(by='COLLECT_DT_TM').drop_duplicates(
        subset=['MRN', 'ANTIBIOTIC', 'ORGANISM_NAME'],
        keep='first'
    )
    
    # Convert 'COLLECT_DT_TM' to datetime
    deduplicated_data['COLLECT_DT_TM'] = pd.to_datetime(deduplicated_data['COLLECT_DT_TM'])

    if not selected_aggregation_unit:
        grouped_data = deduplicated_data.groupby(['ANTIBIOTIC', pd.Grouper(key='COLLECT_DT_TM', freq='W')]).apply(
            lambda x: round((x['Count_S'].sum() + x['Count_S-IE'].sum() + x['Count_S-DD'].sum()) / x['sum'].sum() * 100, 1)
        ).reset_index(name='Susceptibility Rate')
    
    if selected_aggregation_unit == 'D':
        # Daily aggregation
        grouped_data = deduplicated_data.groupby(['ANTIBIOTIC', pd.Grouper(key='COLLECT_DT_TM', freq='D')]).apply(
            lambda x: round((x['Count_S'].sum() + x['Count_S-IE'].sum() + x['Count_S-DD'].sum()) / x['sum'].sum() * 100, 1)
        ).reset_index(name='Susceptibility Rate')
    elif selected_aggregation_unit == 'W':
        # Weekly aggregation
        grouped_data = deduplicated_data.groupby(['ANTIBIOTIC', pd.Grouper(key='COLLECT_DT_TM', freq='W')]).apply(
            lambda x: round((x['Count_S'].sum() + x['Count_S-IE'].sum() + x['Count_S-DD'].sum()) / x['sum'].sum() * 100, 1)
        ).reset_index(name='Susceptibility Rate')
    elif selected_aggregation_unit == 'M':
        # Monthly aggregation
        grouped_data = deduplicated_data.groupby(['ANTIBIOTIC', pd.Grouper(key='COLLECT_DT_TM', freq='M')]).apply(
            lambda x: round((x['Count_S'].sum() + x['Count_S-IE'].sum() + x['Count_S-DD'].sum()) / x['sum'].sum() * 100, 1)
        ).reset_index(name='Susceptibility Rate')
    elif selected_aggregation_unit == 'Q':
        # Quarterly aggregation (3-month periods)
        grouped_data = deduplicated_data.groupby(['ANTIBIOTIC', pd.Grouper(key='COLLECT_DT_TM', freq='Q')]).apply(
            lambda x: round((x['Count_S'].sum() + x['Count_S-IE'].sum() + x['Count_S-DD'].sum()) / x['sum'].sum() * 100, 1)
        ).reset_index(name='Susceptibility Rate')
    elif selected_aggregation_unit == 'Y':
        # Yearly aggregation
        grouped_data = deduplicated_data.groupby(['ANTIBIOTIC', pd.Grouper(key='COLLECT_DT_TM', freq='Y')]).apply(
            lambda x: round((x['Count_S'].sum() + x['Count_S-IE'].sum() + x['Count_S-DD'].sum()) / x['sum'].sum() * 100, 1)
        ).reset_index(name='Susceptibility Rate')

    # Create the line chart figure using Plotly Express
    fig = px.line(
        grouped_data,
        x='COLLECT_DT_TM',
        y='Susceptibility Rate',
        color='ANTIBIOTIC',
        title=f'Susceptibility Rate Over Time ({", ".join(selected_antibiotic)})',
        labels={'COLLECT_DT_TM': 'Date', 'Susceptibility Rate': 'Susceptibility Rate'},
    )
    

    x_axis_title = 'Date'
    
    if not selected_aggregation_unit:
    # Default to Weekly if no selection is made
        selected_aggregation_unit = 'W'
    
    if selected_aggregation_unit == 'D':
    # Daily aggregation
        x_axis_title += ' (Daily)'
        x_axis_tickmode = 'auto'  # Automatically adjust ticks
    elif selected_aggregation_unit == 'W':
        # Weekly aggregation
        x_axis_title += ' (Weekly)'
        x_axis_tickmode = 'auto'  # Automatically adjust ticks
    elif selected_aggregation_unit == 'M':
        # Monthly aggregation
        x_axis_title += ' (Monthly)'
        x_axis_tickmode = 'auto'  # Automatically adjust ticks
    elif selected_aggregation_unit == 'Q':
        # Quarterly aggregation
        x_axis_title += ' (Quarterly)'
        x_axis_tickmode = 'auto'  # Automatically adjust ticks
    elif selected_aggregation_unit == 'Y':
        # Yearly aggregation
        x_axis_title += ' (Yearly)'
        x_axis_tickmode = 'auto'  # Automatically adjust ticks

    fig.update_layout(
        xaxis_title=x_axis_title,  # Set x-axis title based on selected aggregation unit
        xaxis=dict(
            tickmode=x_axis_tickmode,  # Set tick mode to auto-adjust
            nticks=5,  # Maximum number of ticks to display (you can adjust this as needed)
        ),
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Set plot background color to transparent
        paper_bgcolor='rgba(0, 0, 0, 0)',  # Set paper background color to transparent
    )

    return fig

@callback(
    Output('antibiogram-table', 'columns'),
    Output('antibiogram-table', 'data'),
    Input('orderable-dropdown', 'value'),
    Input('antibiotic-dropdown', 'value'),
    Input('med-service-dropdown', 'value'),
    Input('nurse-loc-dropdown', 'value'),
    Input('mo-dropdown', 'value'),
    Input('organism-dropdown', 'value'),
    Input('test-filter', 'value'),
    Input('date-range-picker', 'start_date'),
    Input('date-range-picker', 'end_date'))
def update_table(selected_orderable, selected_antibiotic, selected_med_service, selected_nurse_locs, selected_mo, selected_organism, test_filter, start_date, end_date):
    # Convert start_date and end_date to datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    filtered_df = load_resistance_data_model()

    if selected_orderable:
        filtered_df = filtered_df[filtered_df['ORDERABLE'].isin(selected_orderable)]
    
    if selected_antibiotic:
        filtered_df = filtered_df[filtered_df['ANTIBIOTIC'].isin(selected_antibiotic)]

    if selected_med_service:
            filtered_df = filtered_df[filtered_df['MED_SERVICE'].isin(selected_med_service)]
    
    if selected_nurse_locs:
        filtered_df = filtered_df[filtered_df['NURSE_LOC'].isin(selected_nurse_locs)]
    
    if selected_mo:
        filtered_df = filtered_df[filtered_df['ADMITTING_MO'].isin(selected_mo)]

    if selected_organism:
        filtered_df = filtered_df[filtered_df['ORGANISM_NAME'].isin(selected_organism)]
    
    # Filter the DataFrame based on the selected date range

    filtered_df = filtered_df[(filtered_df['COLLECT_DT_TM'] >= start_date) & (filtered_df['COLLECT_DT_TM'] <= end_date)]
    
    deduplicated_df = filtered_df.sort_values(by='COLLECT_DT_TM').drop_duplicates(
        subset=['MRN', 'ANTIBIOTIC', 'ORGANISM_NAME'],
        keep='first'
    )
    
    grouped_df = deduplicated_df.groupby(['ORGANISM_NAME', 'ANTIBIOTIC']).apply(
        lambda x: round((x['Count_S'].sum() + x['Count_S-IE'].sum() + x['Count_S-DD'].sum()) / x['sum'].sum() * 100, 1)
    ).reset_index(name='S_rate')

    # Calculate the total number of tests ('sum') for each combination
    test_count_df = deduplicated_df.groupby(['ORGANISM_NAME', 'ANTIBIOTIC'])['sum'].sum().reset_index(name='tests')
    
    # Calculate the total number of tests ('sum') for each combination
    total_tests = deduplicated_df.groupby(['ORGANISM_NAME'])['sum'].sum().reset_index(name='total_tests')

    # Merge the susceptibility_rate and test_count DataFrames
    merged_df = pd.merge(grouped_df, test_count_df, on=['ORGANISM_NAME', 'ANTIBIOTIC'], how='left')

    # Filter data based on the selected test filter value
    if test_filter == '30_plus_tests':
        merged_df = merged_df[merged_df['tests'] >= 30]


    # Pivot the table to have ORGANISM_NAME as rows and ANTIBIOTIC as columns
    pivot_df = merged_df.pivot_table(
        index='ORGANISM_NAME',
        columns='ANTIBIOTIC',
        values=['S_rate', 'tests'],
        fill_value=0
    )

    # Reset the index to make ORGANISM_NAME a column
    pivot_df.reset_index(inplace=True)

    # Flatten the MultiIndex columns
    pivot_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in pivot_df.columns]

#   # Create a list to hold the column order
    column_order = ['ORGANISM_NAME']

    # Get the list of unique antibiotics from the columns
    antibiotics = [col for col in pivot_df.columns if col != 'ORGANISM_NAME']

    # Iterate through unique antibiotics
    for index, antibiotic in enumerate(antibiotics):
        if index < len(antibiotics) // 2:
            column_order.append(antibiotic)
            paired_antibiotic = antibiotics[index + len(antibiotics) // 2]
            column_order.append(paired_antibiotic)

    # Rearrange the columns in the pivot_df DataFrame based on the column_order list
    pivot_df = pivot_df[column_order]

    final_df = pd.merge(total_tests, pivot_df, on=['ORGANISM_NAME'], how='left')
    
    #Filter out rows that have no data
    final_df = final_df.dropna(subset=[col for col in final_df.columns if col not in ['ORGANISM_NAME', 'total_tests']], how='all')
    
#     # Define the columns for the DataTable
    columns = [{"name": 'ORGANISM_NAME', "id": 'ORGANISM_NAME', "deletable": True, "selectable": True}]
    columns.extend([{"name": col, "id": col} for col in final_df.columns if col != 'ORGANISM_NAME'])

#     # Convert pivot_df to a list of dictionaries
    data = final_df.to_dict('records')
    
    return columns, data


@callback(
    Output('antibiotic-chart', 'figure'),
    Input('orderable-dropdown', 'value'),
    Input('antibiotic-dropdown', 'value'),
    Input('med-service-dropdown', 'value'),
    Input('nurse-loc-dropdown', 'value'),
    Input('mo-dropdown', 'value'),
    Input('organism-dropdown', 'value'),
    Input('test-filter', 'value'),
    Input('date-range-picker', 'start_date'),
    Input('date-range-picker', 'end_date'))
def update_bar_chart(selected_orderable, selected_antibiotic, selected_med_service, selected_nurse_locs, selected_mo, selected_organism, test_filter, start_date, end_date):

    filtered_df = load_resistance_data_model()
    
     # Filter the DataFrame based on the selected date range
    filtered_df = filtered_df[(filtered_df['COLLECT_DT_TM'] >= start_date) & (filtered_df['COLLECT_DT_TM'] <= end_date)]
    
    # Calculate susceptibility rate for each organism
    deduplicated_data_time = filtered_df.sort_values(by='COLLECT_DT_TM').drop_duplicates(
        subset=['MRN', 'ANTIBIOTIC', 'ORGANISM_NAME'],
        keep='first'
    )
    
    overall_susc = deduplicated_data_time.groupby('ANTIBIOTIC').apply(
        lambda x: round((x['Count_S'].sum() + x['Count_S-IE'].sum() + x['Count_S-DD'].sum()) / x['sum'].sum() * 100)
    ).reset_index(name='Susceptibility Rate')
  

    if selected_orderable:
        filtered_df = filtered_df[filtered_df['ORDERABLE'].isin(selected_orderable)]
    
    if selected_antibiotic:
        filtered_df = filtered_df[filtered_df['ANTIBIOTIC'].isin(selected_antibiotic)]

    if selected_med_service:
            filtered_df = filtered_df[filtered_df['MED_SERVICE'].isin(selected_med_service)]
    
    if selected_nurse_locs:
        filtered_df = filtered_df[filtered_df['NURSE_LOC'].isin(selected_nurse_locs)]
    
    if selected_mo:
        filtered_df = filtered_df[filtered_df['ADMITTING_MO'].isin(selected_mo)]
    
    if selected_organism:
        filtered_df = filtered_df[filtered_df['ORGANISM_NAME'].isin(selected_organism)]
    
    if filtered_df.empty:
        # Handle case where no data matches the filter criteria
        empty_figure = {
            'data': [],
            'layout': {
                'title': 'No data to display',
                'xaxis': {'title': 'Susceptibility Rate'},
                'yaxis': {'title': 'ORGANISM_NAME'},
            }
        }
        return empty_figure
    
    # Calculate susceptibility rate for each antibiotic
    deduplicated_data = filtered_df.sort_values(by='COLLECT_DT_TM').drop_duplicates(
        subset=['MRN', 'ANTIBIOTIC', 'ORGANISM_NAME'],
        keep='first'
    )

    # Calculate susceptibility rate for each antibiotic
    grouped_data = deduplicated_data.groupby('ANTIBIOTIC').apply(
        lambda x: round((x['Count_S'].sum() + x['Count_S-IE'].sum() + x['Count_S-DD'].sum()) / x['sum'].sum() * 100)
    ).reset_index(name='Susceptibility Rate')
    
    test_count_df = deduplicated_data.groupby('ANTIBIOTIC')['sum'].sum().reset_index(name='tests')

    # Merge the susceptibility rate data with the total test count data
    merged_data = pd.merge(grouped_data, test_count_df, on='ANTIBIOTIC', how='left')

    # Filter data based on the selected test filter value
    if test_filter == '30_plus_tests':
        merged_data = merged_data[merged_data['tests'] >= 30]

    # Sort data by susceptibility rate in ascending order
    sorted_data = merged_data.sort_values(by='Susceptibility Rate')

    # Select the lowest 10 antibiotics
    lowest_10 = sorted_data.head(10)
    
    lowest_10 = lowest_10.sort_values(by='Susceptibility Rate',ascending=False)
    
    if lowest_10.empty:
        empty_figure = {
            'data': [],
            'layout': {
                'title': 'No data to display',
                'xaxis': {'title': 'Susceptibility Rate'},
                'yaxis': {'title': 'ORGANISM_NAME'},
            }
        }
        return empty_figure
    
    lowest_10['Label'] = lowest_10.apply(lambda row: f"{row['Susceptibility Rate']}% ({row['tests']} tests)", axis=1)

    # Create the horizontal bar chart using Plotly Express
    fig = px.bar(
        lowest_10,
        x='Susceptibility Rate',
        y='ANTIBIOTIC',
        text='Label',
        orientation='h',
        title='Top 10 Antibiotics by Lowest Susceptibility Rate',
        labels={'ANTIBIOTIC': 'Antibiotic', 'Susceptibility Rate': 'Susceptibility Rate'},
    )
    
    fig.update_traces(
    hovertemplate='%{y}: %{text} <extra></extra>',
        textposition='outside'
                    )
    
    for i, row in overall_susc.iterrows():
        if row['ANTIBIOTIC'] in lowest_10['ANTIBIOTIC'].tolist():
            organism_name = row['ANTIBIOTIC']
            susceptibility_rate = row['Susceptibility Rate']
            formatted_rate = f"{susceptibility_rate:.0f}%"
            legend_label = f"Overall Rate: {formatted_rate}"
            fig.add_trace(go.Scatter(
                x=[susceptibility_rate],
                y=[organism_name],
                mode='markers',
                marker=dict(size=10),
                name=legend_label,
                hoverinfo='text',  # Set hoverinfo to show only the text
                text=legend_label,
            ))
    
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    start_date = start_date.date()
    end_date = end_date.date()
    
    # Modify the code to set the legend title
#     fig.update_layout(
#         legend=dict(title=f"Overall Rate for {start_date} to {end_date}",
#                    font=dict(size=10))
#     )
    
     #toggle legend off
    fig.update_layout(
        showlegend=False)
    
    fig.update_xaxes(title='Susceptibility Rate')
    fig.update_yaxes(title='ANTIBIOTIC')
    #fig.update_xaxes(title='Susceptibility Rate', range=[0, 150],fixedrange=True)
    fig.update_xaxes(title='Susceptibility Rate', range=[0, 150], tickvals=[0, 20, 40, 60, 80, 100], ticktext=['0%', '20%', '40%', '60%', '80%', '100%'])


    fig.update_layout(
    width=1000,
    plot_bgcolor='rgba(0, 0, 0, 0)',  # Set plot background color to transparent
    paper_bgcolor='rgba(0, 0, 0, 0)',  # Set paper background color to transparent
)


    return fig


@callback(
    Output('organism-chart', 'figure'),
    Input('orderable-dropdown', 'value'),
    Input('antibiotic-dropdown', 'value'),
    Input('med-service-dropdown', 'value'),
    Input('nurse-loc-dropdown', 'value'),
    Input('mo-dropdown', 'value'),
    Input('organism-dropdown', 'value'),
    Input('test-filter', 'value'),
    Input('date-range-picker', 'start_date'),
    Input('date-range-picker', 'end_date'))
def update_bar_chart(selected_orderable, selected_antibiotic, selected_med_service, selected_nurse_locs, selected_mo, selected_organism, test_filter, start_date, end_date):

    filtered_df = load_resistance_data_model()
    
    # Filter the DataFrame based on the selected date range
    filtered_df = filtered_df[(filtered_df['COLLECT_DT_TM'] >= start_date) & (filtered_df['COLLECT_DT_TM'] <= end_date)]
    
    # Calculate susceptibility rate for each organism
    deduplicated_data_time = filtered_df.sort_values(by='COLLECT_DT_TM').drop_duplicates(
        subset=['MRN', 'ANTIBIOTIC', 'ORGANISM_NAME'],
        keep='first'
    )
    
    overall_susc = deduplicated_data_time.groupby('ORGANISM_NAME').apply(
        lambda x: round((x['Count_S'].sum() + x['Count_S-IE'].sum() + x['Count_S-DD'].sum()) / x['sum'].sum() * 100)
    ).reset_index(name='Susceptibility Rate')
    

    if selected_orderable:
        filtered_df = filtered_df[filtered_df['ORDERABLE'].isin(selected_orderable)]
    
    if selected_antibiotic:
        filtered_df = filtered_df[filtered_df['ANTIBIOTIC'].isin(selected_antibiotic)]

    if selected_med_service:
            filtered_df = filtered_df[filtered_df['MED_SERVICE'].isin(selected_med_service)]
    
    if selected_nurse_locs:
        filtered_df = filtered_df[filtered_df['NURSE_LOC'].isin(selected_nurse_locs)]
    
    if selected_mo:
        filtered_df = filtered_df[filtered_df['ADMITTING_MO'].isin(selected_mo)]

    if selected_organism:
        filtered_df = filtered_df[filtered_df['ORGANISM_NAME'].isin(selected_organism)]
   
   
    if filtered_df.empty:
        # Handle case where no data matches the filter criteria
        empty_figure = {
            'data': [],
            'layout': {
                'title': 'No data to display',
                'xaxis': {'title': 'Susceptibility Rate'},
                'yaxis': {'title': 'ORGANISM_NAME'},
            }
        }
        return empty_figure
    
    # Calculate susceptibility rate for each organism
    deduplicated_data = filtered_df.sort_values(by='COLLECT_DT_TM').drop_duplicates(
        subset=['MRN', 'ANTIBIOTIC', 'ORGANISM_NAME'],
        keep='first'
    )

    # Calculate susceptibility rate for each organism
    grouped_data = deduplicated_data.groupby('ORGANISM_NAME').apply(
        lambda x: round((x['Count_S'].sum() + x['Count_S-IE'].sum() + x['Count_S-DD'].sum()) / x['sum'].sum() * 100)
    ).reset_index(name='Susceptibility Rate')
    
    test_count_df = deduplicated_data.groupby('ORGANISM_NAME')['sum'].sum().reset_index(name='tests')

    # Merge the susceptibility rate data with the total test count data
    merged_data = pd.merge(grouped_data, test_count_df, on='ORGANISM_NAME', how='left')

    # Filter data based on the selected test filter value
    if test_filter == '30_plus_tests':
        merged_data = merged_data[merged_data['tests'] >= 30]

    # Sort data by susceptibility rate in ascending order
    sorted_data = merged_data.sort_values(by='Susceptibility Rate')

    # Select the lowest 10 organisms
    lowest_10 = sorted_data.head(10)
    
    lowest_10 = lowest_10.sort_values(by='Susceptibility Rate', ascending=False)
    
    if lowest_10.empty:
        empty_figure = {
            'data': [],
            'layout': {
                'title': 'No data to display',
                'xaxis': {'title': 'Susceptibility Rate'},
                'yaxis': {'title': 'ORGANISM_NAME'},
            }
        }
        return empty_figure
    
    lowest_10['Label'] = lowest_10.apply(lambda row: f"{row['Susceptibility Rate']}% ({row['tests']} tests)", axis=1)

    # Create the horizontal bar chart using Plotly Express
    fig = px.bar(
        lowest_10,
        x='Susceptibility Rate',
        y='ORGANISM_NAME',
        text='Label',
        orientation='h',
        title='Top 10 Organisms by Lowest Susceptibility Rate',
        labels={'ORGANISM_NAME': 'Organism', 'Susceptibility Rate': 'Susceptibility Rate'},
        width=1000,
    )
    
    fig.update_layout(bargroupgap=0.2) 
    fig.update_layout(bargap=0.2) 
    

    fig.update_traces(
    hovertemplate='%{y}: %{text} <extra></extra>',
        textposition='outside'
                    )
    
    for i, row in overall_susc.iterrows():
        if row['ORGANISM_NAME'] in lowest_10['ORGANISM_NAME'].tolist():
            organism_name = row['ORGANISM_NAME']
            susceptibility_rate = row['Susceptibility Rate']
            formatted_rate = f"{susceptibility_rate:.0f}%"
            legend_label = f"Overall Rate: {formatted_rate}"
            fig.add_trace(go.Scatter(
                x=[susceptibility_rate],
                y=[organism_name],
                mode='markers',
                marker=dict(size=10),
                name=legend_label,
                hoverinfo='text',  # Set hoverinfo to show only the text
                text=legend_label,
            ))
    
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    start_date = start_date.date()
    end_date = end_date.date()
    
  #  Modify the code to set the legend title
#     fig.update_layout(
#         legend=dict(title=f"Overall Rate for {start_date} to {end_date}",
#                    font=dict(size=10),
#                    x=0.5,
#                    y=1)
#     )
    
    #toggle legend off
    fig.update_layout(
        showlegend=False)
    
    

    #fig.update_xaxes(title='Susceptibility Rate', range=[0, 150],fixedrange=True)
    fig.update_xaxes(title='Susceptibility Rate', range=[0, 150], tickvals=[0, 20, 40, 60, 80, 100], ticktext=['0%', '20%', '40%', '60%', '80%', '100%'])

    
    fig.update_layout(
        width=1000,
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Set plot background color to transparent
        paper_bgcolor='rgba(0, 0, 0, 0)',  # Set paper background color to transparent
    )

    return fig