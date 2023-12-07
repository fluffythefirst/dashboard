import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, Dash, State, callback, ctx, dash_table
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

import re

dash.register_page(__name__, order = 2)

## -- SETUP GLOBAL VARIABLES ---

ADMIT_DATE_DATE_FORMAT = '%d/%m/%Y'
DISCHARGE_DATE_DATE_FORMAT = '%d/%m/%Y %H:%M'
ORDER_DATE_DATE_FORMAT = '%d/%m/%Y %H:%M'
CSV_READ = './orders_data_clean/final_df.csv'

## -- END SETUP GLOBAL VARIABLES -- 

## --- START AGG DATA MODEL ---

def read_and_transform_agg_model():
    df = pd.read_csv(CSV_READ)

    # rename for shorter labels
    df.rename(columns={'LOCATION_OF_PATIENT_AT_THE_TIME_OF_ORDER': 'WARD'}, inplace=True)
    df.rename(columns={'ATTENDING_MEDICAL_OFFICER': 'DOCTOR'}, inplace=True)
    df.rename(columns={'total_DDD': 'TOTAL_DDD'}, inplace=True)
    df.rename(columns={'total_dosage': 'TOTAL_DOSAGE'}, inplace=True)
    df.rename(columns={'ORDER_PLACED_DATE': 'ORDER_DATE'}, inplace=True)
    df.rename(columns={'DAYS_OF_THERAPY': 'TOTAL_DOT'}, inplace=True)

    df['DOCTOR'] = df['DOCTOR'].str.replace(r'\s*\([^)]*\)$', '', regex=True) # remove (MO)(SMO)

    filtered_df = df.loc[:, ['MRN', 'ORDER_DATE', 'ORDER_STATUS','ORDER_GENERIC','MEDICAL_SERVICE','WARD','AMS_INDICATION','DOCTOR','TOTAL_DDD','TOTAL_DOSAGE','TOTAL_DOT', 'AGE', 'ACTUAL_WEIGHT', 'dASC', 'EGFR', 'FREQUENCY', 'DOSE', 'PATIENT_NAME']] # select relevant attributes only

    filtered_df['ORDER_DATE'] = pd.to_datetime(filtered_df['ORDER_DATE'], format = ORDER_DATE_DATE_FORMAT)
    
    filtered_df['ORDER_MONTH_YEAR'] = pd.to_datetime(filtered_df['ORDER_DATE']).dt.strftime('%Y-%m')

    return filtered_df.copy(deep = True)

## --- END AGG DATA MODEL ---

## -- START EGFR DOSE FIRST 24 HOURS DATA MODEL --

def read_and_transform_egfr_dose_first_24h_data_model():
    egfr_dosef24 = pd.read_csv(CSV_READ)

    egfr_dosef24.rename(columns={'LOCATION_OF_PATIENT_AT_THE_TIME_OF_ORDER': 'WARD'}, inplace=True)
    egfr_dosef24.rename(columns={'ATTENDING_MEDICAL_OFFICER': 'DOCTOR'}, inplace=True)
    egfr_dosef24.rename(columns={'ORDER_PLACED_DATE': 'ORDER_DATE'}, inplace=True)
    egfr_dosef24.rename(columns={'dosage_f24': 'FIRST 24hr DOSE'}, inplace=True)

    egfr_dosef24['ORDER_DATE'] = pd.to_datetime(egfr_dosef24['ORDER_DATE'], format = ORDER_DATE_DATE_FORMAT)


    return egfr_dosef24.copy(deep = True)

## -- END EGFR DOSE FIRST 24 HOURS DATA MODEL -- 

def read_and_transform_length_of_stay_data_model():
    los_df = pd.read_csv(CSV_READ)
    los_df['ADMIT_DATE'] = pd.to_datetime(los_df['ADMIT_DATE'], format = ADMIT_DATE_DATE_FORMAT)
    los_df['DISCHARGE_DATE'] = pd.to_datetime(los_df['DISCHARGE_DATE'], format = DISCHARGE_DATE_DATE_FORMAT)

    los_df['LENGTH_OF_STAY'] = los_df['DISCHARGE_DATE'] - los_df['ADMIT_DATE']
    los_df['LENGTH_OF_STAY_DAYS'] = los_df['LENGTH_OF_STAY'].dt.days
    los_df['LENGTH_OF_STAY_HOURS'] = los_df['LENGTH_OF_STAY'].dt.components['hours']

    # Remove row that does not have discharge date
    los_df.dropna(subset = ['DISCHARGE_DATE'], inplace = True)
    los_df.reset_index(drop = True, inplace = True)

    # los_df.to_csv("los_df.csv", index = False)

    return los_df.copy(deep = True)


filtered_df = read_and_transform_agg_model()
egfr_dosef24 = read_and_transform_egfr_dose_first_24h_data_model()
los_df = read_and_transform_length_of_stay_data_model()

## --- START DATA MODEL INITIALISATION ---

# Dropdowns for general filters
unique_order_status = filtered_df["ORDER_STATUS"].dropna().unique()
unique_order_generic = filtered_df["ORDER_GENERIC"].dropna().unique()
unique_medical_service = filtered_df["MEDICAL_SERVICE"].dropna().unique()
unique_ward = filtered_df["WARD"].dropna().unique()
unique_doctor = filtered_df["DOCTOR"].dropna().unique()
unique_ams_indication = filtered_df["AMS_INDICATION"].dropna().unique()

# Dropdown for egfr dose first 24 hours
unique_order_generic_egfr_dosef24 = egfr_dosef24['ORDER_GENERIC'].dropna().unique().tolist()

# Dropdowns for length of stay
unique_order_generic_los = los_df['ORDER_GENERIC'].dropna().unique()
unique_ams_indication_los = los_df['AMS_INDICATION'].dropna().unique()
unique_frequency_los = los_df['FREQUENCY'].dropna().unique()
unique_dose_los = los_df['DOSE'].dropna().unique()

# Metric for 2D Scatter Plot
metric = ['Total DDD', 'Average DDD', 'Total DOT', 'Average DOT', 'Total dASC', 'Average dASC']

## --- END DATA MODEL INITIALISATION ---

## -- START LAYOUT HELPER FUNCTIONS --

def filter_date_and_update_text(df_copy, start_date, end_date, show_all_button_id = "show-all-button-deep-dive-page"):
    '''
    Filter a dataframe by date range, update date-related text, and handle "show all" functionality.

    Args:
    - df_copy (pandas.DataFrame): The input dataframe with a column named 'ORDER_DATE'.
    - start_date (str): The start date in the format 'YYYY-MM-DD'.
    - end_date (str): The end date in the format 'YYYY-MM-DD'.
    - show_all_button_id (str, optional): The ID for a button that triggers the "show all" functionality. Defaults to "show-all-button-deep-dive-page".

    Returns:
    - tuple: A 4-tuple containing:
        1. pandas.DataFrame: The filtered dataframe.
        2. str: Text describing the selected date range.
        3. dash.no_update or None: Indicator for the start date in the frontend. Returns None if "show all" is triggered.
        4. dash.no_update or None: Indicator for the end date in the frontend. Returns None if "show all" is triggered.

    Behavior:
    - If both start_date and end_date are provided, the function will filter the dataframe to only include rows 
      within that date range and update the associated text.
    - If only one of start_date or end_date is provided, the function will filter the dataframe based on that single date 
      and update the associated text.
    - If the "show all" functionality is triggered (checked against the show_all_button_id), the function returns 
      the unfiltered dataframe and updates the date-related text to cover the entire range.
    - If neither a date range nor the "show all" functionality is triggered, a default message prompts the user to select a date range.
    '''

    df_copy_unfiltered = df_copy.copy(deep = True)

    string_prefix = 'You have selected from'

    if start_date is not None:
        # Convert string to datetime
        start_datetime = pd.to_datetime(start_date, format='%Y-%m-%d')

        # Convert datetime to string
        start_date_string = start_datetime.strftime("%d %B, %Y")
        string_prefix = string_prefix + ' ' + start_date_string

        # Subset the dataset from start date
        df_copy = df_copy.loc[df_copy['ORDER_DATE'] >= start_datetime]
    
    if end_date is not None:
        # Convert string to datetime
        end_datetime = pd.to_datetime(end_date, format='%Y-%m-%d')

        # Convert datetime to string
        end_date_string = end_datetime.strftime("%d %B, %Y")
        string_prefix = string_prefix + ' to ' + end_date_string

        # Subset the dataset from start date to end date
        df_copy = df_copy.loc[df_copy['ORDER_DATE'] <= end_datetime]

    start_date_text = dash.no_update
    end_date_text = dash.no_update

    if show_all_button_id == ctx.triggered_id:
        # Show all the dates
        df_copy = df_copy_unfiltered
        start_date_text = None
        end_date_text = None

        # Update the date text by overriding string_prefix
        string_prefix = 'You have selected from'

        # Get minimum datetime
        start_datetime = min(pd.to_datetime(df_copy['ORDER_DATE']))
        # Convert to string
        start_date_string = start_datetime.strftime("%d %B, %Y")

        # Update date text
        string_prefix = string_prefix + ' ' + start_date_string

        # Get maximum datetime
        end_datetime = max(pd.to_datetime(df_copy['ORDER_DATE']))
         # Convert to string
        end_date_string = end_datetime.strftime("%d %B, %Y")

        # Update date text
        string_prefix = string_prefix + ' to ' + end_date_string

    if len(string_prefix) == len('You have selected from'):
        string_prefix = 'Select a date range to filter'

    return df_copy.copy(deep = True), string_prefix, start_date_text, end_date_text


def create_agg_2d_scatter_plot_df(df):
    '''
    Aggregate and prepare a dataframe for 2D scatter plot visualization.

    Args:
    - df (pandas.DataFrame): The input dataframe.

    Returns:
    - pandas.DataFrame: A merged dataframe with aggregated values for TOTAL_DDD, TOTAL_DOT, and dASC 
      grouped by 'ORDER_GENERIC'. The returned dataframe will have columns: 
      'ORDER_GENERIC', 'Total DDD', 'Average DDD', 'Total DOT', 'Average DOT', 'Total dASC', and 'Average dASC'.
    '''
    cf_DDD = df.groupby(['ORDER_GENERIC']).agg({'TOTAL_DDD': ['sum', 'mean']}).reset_index()
    cf_DDD.columns = ['ORDER_GENERIC','Total DDD','Average DDD']
    
    cf_DOT = df.groupby(['ORDER_GENERIC']).agg({'TOTAL_DOT': ['sum','mean']}).reset_index()
    cf_DOT.columns = ['ORDER_GENERIC','Total DOT', 'Average DOT']
    
    cf_dASC = df.groupby(['ORDER_GENERIC']).agg({'dASC': ['sum','mean']}).reset_index()
    cf_dASC.columns = ['ORDER_GENERIC','Total dASC', 'Average dASC']

    cfmerge_df = pd.merge(cf_DDD, cf_DOT, on = 'ORDER_GENERIC', how = 'inner')
    cfmerge_df = pd.merge(cfmerge_df, cf_dASC, on = 'ORDER_GENERIC', how = 'inner')

    return cfmerge_df

def create_agg_time_series_plot_df(tsdf):
    '''
    Aggregate and prepare a dataframe for time series plot visualization.

    Args:
    - tsdf (pandas.DataFrame): The input dataframe with time series data.

    Returns:
    - pandas.DataFrame: A merged dataframe with aggregated values for TOTAL_DDD, TOTAL_DOT, and dASC
      grouped by 'ORDER_MONTH_YEAR'. The returned dataframe will have columns: 
      'ORDER_MONTH_YEAR', 'Total DDD', 'Average DDD', 'Total DOT', 'Average DOT', 'Total dASC', and 'Average dASC'.

    Note:
    This function assumes that the 'ORDER_MONTH_YEAR' column in the input dataframe represents a time 
    period, such as a month and year, over which the data is aggregated.
    '''
    tsDDD = tsdf.groupby(['ORDER_MONTH_YEAR']).agg({'TOTAL_DDD':['sum','mean']}).reset_index()
    tsDDD.columns = ['ORDER_MONTH_YEAR','Total DDD','Average DDD']

    tsDOT = tsdf.groupby(['ORDER_MONTH_YEAR']).agg({'TOTAL_DOT': ['sum','mean']}).reset_index()
    tsDOT.columns = ['ORDER_MONTH_YEAR','Total DOT','Average DOT']

    tsdASC = tsdf.groupby(['ORDER_MONTH_YEAR']).agg({'dASC':['sum','mean']}).reset_index()
    tsdASC.columns = ['ORDER_MONTH_YEAR','Total dASC','Average dASC']

    tsmergeDF = pd.merge(tsDDD, tsDOT, on = 'ORDER_MONTH_YEAR', how = 'inner')
    tsmergeDF = pd.merge(tsmergeDF, tsdASC, on = 'ORDER_MONTH_YEAR', how = 'inner')

    return tsmergeDF


def draw_time_series(tsdf, title, axis_name):
    '''
    Plot time series data on a figure with a given title and y-axis name.
    
    Args:
    - tsdf (pandas.DataFrame): Input dataframe containing time series data. Assumes there's a column named 'ORDER_MONTH_YEAR'.
    - title (str): The title for the plot, which will be displayed as an annotation.
    - axis_name (str): The name of the column in tsdf that you want to plot on the y-axis.
    
    Returns:
    - plotly.graph_objects.Figure: A plotly figure object ready for visualization.
    
    The function uses Plotly's scatter plot and annotates the plot with the provided title.
    It also formats the x-axis to display month and year, and sets the figure dimensions and style.
    '''
    fig = px.scatter(tsdf, x = 'ORDER_MONTH_YEAR', y = axis_name)

    # Update layout
    distinct_months_years = tsdf['ORDER_MONTH_YEAR'].unique()
    fig.update_xaxes(tickvals = distinct_months_years, \
                                    ticktext = [pd.to_datetime(date).strftime('%b %Y') for date in distinct_months_years])
    
    fig.update_layout(xaxis_title = "Order Month Year")
    fig.update_traces(mode = 'lines+markers')
    fig.update_xaxes(showgrid = False)
    fig.add_annotation(x = 0, y = 0.85, xanchor = 'left', yanchor = 'bottom', xref = 'paper', yref = 'paper', showarrow = False, align = 'left', text = title)
    fig.update_layout(height = 225, margin = {'l':20,'b':30, 'r':10, 't':10})

    return fig

## -- END LAYOUT HELPER FUNCTIONS --

## - START LAYOUT

layout = html.Div(
    [
        html.Div(id='intermediate-div-deep-dive-page', style={'display': 'none'}),
        html.H3('Antimicrobial Consumption - Deep Dive'),
        dbc.Row(
                [
                    dbc.Col(id = 'output-order-date-range-deep-dive-page', style = {'font-weight': 'bold'}),
                    dbc.Col(dcc.DatePickerRange(
                        id = 'order-date-range-deep-dive-page',
                        min_date_allowed = min(filtered_df["ORDER_DATE"]),
                        max_date_allowed = max(filtered_df["ORDER_DATE"]),
                        month_format = 'MMMM Y',
                        end_date_placeholder_text = 'DD/MM/YYYY',
                        display_format = 'DD/MM/YYYY'
                    )),
                    dbc.Col(
                        [
                        dbc.Button(
                            "View full date range",  
                            id = "show-all-button-deep-dive-page",
                            color = "primary"
                            )
                        ]
                    )
                ]
        ),
        dbc.Row(
            [
                dbc.Col([
                        dbc.Col("Order Generic"),
                        dcc.Dropdown(
                            id = 'order-generic-deep-dive-page-dd',
                            options = [{'label': x, 'value': x} for x in unique_order_generic],
                            multi = True,
                            optionHeight = 50                            
                        )
                ]),
                dbc.Col([
                        dbc.Col("Medical Service"),
                        dcc.Dropdown(
                            id = 'medical-service-deep-dive-page-dd',
                            options = [{'label': x, 'value': x} for x in unique_medical_service],
                            multi = True,
                            optionHeight = 50
                        )
                ]),
                dbc.Col([
                        dbc.Col("Ward"),
                        dcc.Dropdown(
                            id = 'ward-deep-dive-page-dd',
                            options = [{'label': x, 'value': x} for x in unique_ward],
                            multi = True,
                            optionHeight = 50
                        )
                ]),
                dbc.Col([
                        dbc.Col("Doctor"),
                        dcc.Dropdown(
                            id = 'doctor-deep-dive-page-dd',
                            options = [{'label': x, 'value': x} for x in unique_doctor],
                            multi = True,
                            optionHeight = 50
                        )
                ]),
                dbc.Col([
                        dbc.Col("AMS Indication"),
                        dcc.Dropdown(
                            id = 'ams-indication-deep-dive-page-dd',
                            options = [{'label': x, 'value': x} for x in unique_ams_indication],
                            multi = True,
                            optionHeight = 60
                        )
                ]),
            ],     
        ),
        html.H3('Correlation Analysis'),
        html.Div([
            html.Label("Please select the x and y axes. Hover over the scatter plot to see the antibiotic time series for the selected axes."),
            html.Div([
                html.Div([
                    dcc.Dropdown(
                        id="crossfilter-xaxis",
                        options=[{'label':x, 'value': x} for x in metric],
                        value='Total DDD',
                        clearable=False,
                        multi=False     
                    )
                ], style={'width':"49%", 'display': 'inline-block'}),
                html.Div([
                    dcc.Dropdown(
                        id='crossfilter-yaxis',
                        options=[{'label':x, 'value': x} for x in metric],
                        value='Average DOT',
                        clearable=False,
                        multi=False
                    )
                ], style={'width':'49%', 'float': 'right', 'display': 'inline-block'})
            ], style={'padding': '10px 5px'}),
            html.Div([
                dcc.Graph(
                    id='crossfilter-scatter',
                    hoverData={'points':[{'customdata': 'Amoxicillin'}]}
                )
            ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div([
                dcc.Graph(id='x-time-series'),
                dcc.Graph(id='y-time-series')
            ], style={'width': '49%', 'display': 'inline-block'})
        ]),
        dbc.Button("Download to csv", id = "download-2d-crossfilter-scatterplot-button"),
        dcc.Download(id = "download-2d-crossfilter-scatterplot-df"),
        html.Div(
            children= [
                html.Label("Please select one type of antibiotic to display in the'Patients 'Age, Weight, and Exact Dose' and 'eGFR Insight' sections"),
                dcc.Dropdown(
                    id='generic-1type-deep-dive-page-dd',
                    options = [{'label':x, 'value':x} for x in unique_order_generic_egfr_dosef24],
                    multi = False,
                    value = unique_order_generic_egfr_dosef24[0] # get the first value for initial display
                ),
                dcc.Graph(id="aw-dose-scatter"),
                dbc.Button("Download to csv", id = "download-aw-dose-scatter-button"),
                dcc.Download(id = "download-aw-dose-scatter-df"),

                dcc.Graph(id="egfr-dose-scatter"),
                dbc.Button("Download to csv", id = "download-egfr-dose-scatter-button"),
                dcc.Download(id = "download-egfr-dose-scatter-df")
            ]
        ),
        html.H4('Length of Stay vs. Number of Patients'),
        dbc.Row(
            [
                dbc.Col([
                        dbc.Col("Order Generic"),
                        dcc.Dropdown(
                            id = 'order-generic-los-deep-dive-page-dd',
                            options = [{'label': x, 'value': x} for x in unique_order_generic_los],
                            multi = True,
                            optionHeight = 50                            
                        )
                ]),
                dbc.Col([
                        dbc.Col("AMS Indication"),
                        dcc.Dropdown(
                            id = 'ams-indication-los-deep-dive-page-dd',
                            options = [{'label': x, 'value': x} for x in unique_ams_indication_los],
                            multi = True,
                            optionHeight = 60
                        )
                ]),
                dbc.Col([
                        dbc.Col("Frequency"),
                        dcc.Dropdown(
                            id = 'frequency-los-dive-page-dd',
                            options = [{'label': x, 'value': x} for x in unique_frequency_los],
                            multi = True,
                            optionHeight = 50
                        )
                ]),
                dbc.Col([
                        dbc.Col("Dose"),
                        dcc.Dropdown(
                            id = 'dose-los-deep-dive-page-dd',
                            options = [{'label': x, 'value': x} for x in unique_dose_los],
                            multi = True,
                            optionHeight = 50
                        )
                ])
            ],     
        ),
        dcc.Graph(id = 'length-of-stay-bar'),
        dash_table.DataTable(
            id = 'length-of-stay-table',
            columns = [
                {'name': 'Length of Stay (Days)', 'id': 'LENGTH_OF_STAY_DAYS'},
                {'name': 'Number of Patients', 'id': 'NUMBER_OF_PATIENTS'}
            ],
            editable = False,
            filter_action = "native",
            sort_action = "native",
            sort_mode = "single",
            row_deletable = False,
            selected_columns = [],
            selected_rows = [],
            page_action = "native",
            page_current = 0,
            page_size = 5        
        ),
        dbc.Button("Download to csv", id = "download-length-of-stay-bar-button"),
        dcc.Download(id = "download-length-of-stay-bar-df")
    ]
)


## - END LAYOUT

@callback( 
        # Initialize order date range
        Output('order-date-range-deep-dive-page', 'min_date_allowed'),
        Output('order-date-range-deep-dive-page', 'max_date_allowed'),

        # Trigger other callback to initialize the charts
        Output('intermediate-div-deep-dive-page', 'children'),

        # From memory
        Input('memory', 'data')
)

def refresh_and_initialize_agg_data(data):
    """
    Callback function to initialize layout once the user has uploaded the new dataset from memory.
    This function executes with every refresh but provides no updates unless there is a change within 
    the update_data() page. 
    If a change occurs, it will populate the minimum dates, maximum dates, and dropdowns.
    This function also outputs intermediate_data as the input for the update_plot() function, 
    which will update the plot with the new dataset.
    
    :param data: Data from memory.
    
    :return: 
        - min_date: Minimum ORDER_DATE.
        - max_date: Maximum ORDER_DATE.
        - intermediate_data: Current timestamp.
    """
    #filtered_df = None

    min_date = dash.no_update
    max_date = dash.no_update

    intermediate_data = dash.no_update

    if data == "dataset-uploaded":
        filtered_df = read_and_transform_agg_model()

        # Fill minimum and maximum dates
        min_date = min(filtered_df["ORDER_DATE"])
        max_date = max(filtered_df["ORDER_DATE"])

        intermediate_data = str(pd.Timestamp.now())

    return min_date, max_date, \
     intermediate_data
       
# Make the filters dynamic

@callback(
        # Output Dropdown Update Options
        Output(component_id = 'order-generic-deep-dive-page-dd', component_property = 'options'),
        Output(component_id = 'medical-service-deep-dive-page-dd', component_property = 'options'),
        Output(component_id = 'ward-deep-dive-page-dd', component_property = 'options'),
        Output(component_id = 'doctor-deep-dive-page-dd', component_property = 'options'),
        Output(component_id = 'ams-indication-deep-dive-page-dd', component_property = 'options'),

        # Output for egfr dose first 24 hours
        Output(component_id = 'generic-1type-deep-dive-page-dd', component_property = 'options'),

        # Date Input Functionality
        Input(component_id = 'show-all-button-deep-dive-page', component_property = 'n_clicks'),
        Input(component_id = 'order-date-range-deep-dive-page', component_property = 'start_date'),
        Input(component_id = 'order-date-range-deep-dive-page', component_property = 'end_date'),

        # Input Dropdown Values
        Input(component_id = 'order-generic-deep-dive-page-dd', component_property = 'value'),
        Input(component_id = 'medical-service-deep-dive-page-dd', component_property = 'value'),
        Input(component_id = 'ward-deep-dive-page-dd', component_property = 'value'),
        Input(component_id = 'doctor-deep-dive-page-dd', component_property = 'value'),
        Input(component_id = 'ams-indication-deep-dive-page-dd', component_property = 'value'),

        # Coming from refresh_and_initialize_agg_data() function
        Input(component_id = 'intermediate-div-deep-dive-page', component_property = 'children')
)

def update_main_filter_values(
                                  show_all_button, 
                                  start_date, 
                                  end_date, 
                                  input_order_generic,
                                  input_medical_service,
                                  input_ward,
                                  input_doctor,
                                  input_ams_indication,
                                  intermediate_data
                                ):
    df_copy = read_and_transform_agg_model()
    
    egfr_dosef24_copy = read_and_transform_egfr_dose_first_24h_data_model()

    # Filter dates
    df_copy, _string_prefix, _start_date_text, _end_date_text = filter_date_and_update_text(df_copy = df_copy, start_date = start_date, end_date = end_date)

    egfr_dosef24_copy, _string_prefix, _start_date_text, _end_date_text = filter_date_and_update_text(df_copy = egfr_dosef24_copy, start_date = start_date, end_date = end_date)

    # Initialise unique values

    unique_order_generic = sorted(df_copy["ORDER_GENERIC"].dropna().unique())

    unique_order_generic_egfr_dosef24 = sorted(egfr_dosef24_copy['ORDER_GENERIC'].dropna().unique())

    unique_medical_service_a = df_copy["MEDICAL_SERVICE"].dropna().unique()
    unique_medical_service_b = egfr_dosef24_copy["MEDICAL_SERVICE"].dropna().unique()

    unique_medical_service = np.union1d(unique_medical_service_a, unique_medical_service_b)

    unique_ward_a = df_copy["WARD"].dropna().unique()
    unique_ward_b = egfr_dosef24_copy["WARD"].dropna().unique()

    unique_ward = np.union1d(unique_ward_a, unique_ward_b)

    unique_doctor_a = df_copy["DOCTOR"].dropna().unique()
    unique_doctor_b = egfr_dosef24_copy["DOCTOR"].dropna().unique()

    unique_doctor = np.union1d(unique_doctor_a, unique_doctor_b)

    unique_ams_indication_a = df_copy["AMS_INDICATION"].dropna().unique()
    unique_ams_indication_b = egfr_dosef24_copy["AMS_INDICATION"].dropna().unique()

    unique_ams_indication = np.union1d(unique_ams_indication_a, unique_ams_indication_b)
    
    # Filter based on dimensions    
    if input_order_generic:
        # Filter
        df_copy = df_copy[df_copy["ORDER_GENERIC"].isin(input_order_generic)]
        egfr_dosef24_copy = egfr_dosef24_copy[egfr_dosef24_copy["ORDER_GENERIC"].isin(input_order_generic)]

        # Update unique values for other dropdowns
        unique_order_generic_egfr_dosef24 = sorted(egfr_dosef24_copy['ORDER_GENERIC'].dropna().unique())

        unique_medical_service_a = df_copy["MEDICAL_SERVICE"].dropna().unique()
        unique_medical_service_b = egfr_dosef24_copy["MEDICAL_SERVICE"].dropna().unique()

        unique_medical_service = np.union1d(unique_medical_service_a, unique_medical_service_b)

        unique_ward_a = df_copy["WARD"].dropna().unique()
        unique_ward_b = egfr_dosef24_copy["WARD"].dropna().unique()

        unique_ward = np.union1d(unique_ward_a, unique_ward_b)

        unique_doctor_a = df_copy["DOCTOR"].dropna().unique()
        unique_doctor_b = egfr_dosef24_copy["DOCTOR"].dropna().unique()

        unique_doctor = np.union1d(unique_doctor_a, unique_doctor_b)

        unique_ams_indication_a = df_copy["AMS_INDICATION"].dropna().unique()
        unique_ams_indication_b = egfr_dosef24_copy["AMS_INDICATION"].dropna().unique()

        unique_ams_indication = np.union1d(unique_ams_indication_a, unique_ams_indication_b)
    
    if input_medical_service:
        # Filter
        df_copy = df_copy[df_copy["MEDICAL_SERVICE"].isin(input_medical_service)]
        egfr_dosef24_copy = egfr_dosef24_copy[egfr_dosef24_copy["MEDICAL_SERVICE"].isin(input_medical_service)]

        # Update unique values for other dropdowns
        unique_order_generic = sorted(df_copy["ORDER_GENERIC"].dropna().unique())

        unique_order_generic_egfr_dosef24 = sorted(egfr_dosef24_copy['ORDER_GENERIC'].dropna().unique())

        unique_ward_a = df_copy["WARD"].dropna().unique()
        unique_ward_b = egfr_dosef24_copy["WARD"].dropna().unique()

        unique_ward = np.union1d(unique_ward_a, unique_ward_b)

        unique_doctor_a = df_copy["DOCTOR"].dropna().unique()
        unique_doctor_b = egfr_dosef24_copy["DOCTOR"].dropna().unique()

        unique_doctor = np.union1d(unique_doctor_a, unique_doctor_b)

        unique_ams_indication_a = df_copy["AMS_INDICATION"].dropna().unique()
        unique_ams_indication_b = egfr_dosef24_copy["AMS_INDICATION"].dropna().unique()

        unique_ams_indication = np.union1d(unique_ams_indication_a, unique_ams_indication_b)

    if input_ward:
        # Filter
        df_copy = df_copy[df_copy["WARD"].isin(input_ward)]
        egfr_dosef24_copy = egfr_dosef24_copy[egfr_dosef24_copy["WARD"].isin(input_ward)]

        # Update unique values for other dropdowns
        unique_order_generic = sorted(df_copy["ORDER_GENERIC"].dropna().unique())

        unique_order_generic_egfr_dosef24 =sorted(egfr_dosef24_copy['ORDER_GENERIC'].dropna().unique())

        unique_medical_service_a = df_copy["MEDICAL_SERVICE"].dropna().unique()
        unique_medical_service_b = egfr_dosef24_copy["MEDICAL_SERVICE"].dropna().unique()

        unique_medical_service = np.union1d(unique_medical_service_a, unique_medical_service_b)

        unique_doctor_a = df_copy["DOCTOR"].dropna().unique()
        unique_doctor_b = egfr_dosef24_copy["DOCTOR"].dropna().unique()

        unique_doctor = np.union1d(unique_doctor_a, unique_doctor_b)

        unique_ams_indication_a = df_copy["AMS_INDICATION"].dropna().unique()
        unique_ams_indication_b = egfr_dosef24_copy["AMS_INDICATION"].dropna().unique()

        unique_ams_indication = np.union1d(unique_ams_indication_a, unique_ams_indication_b)

    if input_doctor:
        # Filter
        df_copy = df_copy[df_copy["DOCTOR"].isin(input_doctor)]
        egfr_dosef24_copy = egfr_dosef24_copy[egfr_dosef24_copy["DOCTOR"].isin(input_doctor)]

        # Update unique values for other dropdowns
        unique_order_generic = sorted(df_copy["ORDER_GENERIC"].dropna().unique())

        unique_order_generic_egfr_dosef24 = sorted(egfr_dosef24_copy['ORDER_GENERIC'].dropna().unique())

        unique_medical_service_a = df_copy["MEDICAL_SERVICE"].dropna().unique()
        unique_medical_service_b = egfr_dosef24_copy["MEDICAL_SERVICE"].dropna().unique()

        unique_medical_service = np.union1d(unique_medical_service_a, unique_medical_service_b)

        unique_ward_a = df_copy["WARD"].dropna().unique()
        unique_ward_b = egfr_dosef24_copy["WARD"].dropna().unique()

        unique_ward = np.union1d(unique_ward_a, unique_ward_b)

        unique_ams_indication_a = df_copy["AMS_INDICATION"].dropna().unique()
        unique_ams_indication_b = egfr_dosef24_copy["AMS_INDICATION"].dropna().unique()

        unique_ams_indication = np.union1d(unique_ams_indication_a, unique_ams_indication_b)
    
    if input_ams_indication:
        # Filter
        df_copy = df_copy[df_copy["AMS_INDICATION"].isin(input_ams_indication)]
        egfr_dosef24_copy = egfr_dosef24_copy[egfr_dosef24_copy["AMS_INDICATION"].isin(input_ams_indication)]

        # Update unique values for other dropdowns

        unique_order_generic = sorted(df_copy["ORDER_GENERIC"].dropna().unique())

        unique_order_generic_egfr_dosef24 = sorted(egfr_dosef24_copy['ORDER_GENERIC'].dropna().unique())

        unique_medical_service_a = df_copy["MEDICAL_SERVICE"].dropna().unique()
        unique_medical_service_b = egfr_dosef24_copy["MEDICAL_SERVICE"].dropna().unique()

        unique_medical_service = np.union1d(unique_medical_service_a, unique_medical_service_b)

        unique_ward_a = df_copy["WARD"].dropna().unique()
        unique_ward_b = egfr_dosef24_copy["WARD"].dropna().unique()

        unique_ward = np.union1d(unique_ward_a, unique_ward_b)

        unique_doctor_a = df_copy["DOCTOR"].dropna().unique()
        unique_doctor_b = egfr_dosef24_copy["DOCTOR"].dropna().unique()

        unique_doctor = np.union1d(unique_doctor_a, unique_doctor_b)

    unique_order_generic_options = [{'label': x, 'value': x} for x in unique_order_generic]
    unique_medical_service_options = [{'label': x, 'value':x} for x in unique_medical_service]
    unique_ward_options = [{'label': x, 'value': x} for x in unique_ward]
    unique_doctor_options = [{'label': x, 'value': x} for x in unique_doctor]
    unique_ams_indication_options = [{'label': x, 'value': x} for x in unique_ams_indication]

    unique_order_generic_egfr_dosef24_options = [{'label':x, 'value':x} for x in unique_order_generic_egfr_dosef24]

    return unique_order_generic_options, unique_medical_service_options, unique_ward_options,\
            unique_doctor_options, unique_ams_indication_options, unique_order_generic_egfr_dosef24_options

@callback(
        # Output text for order date range
        Output(component_id = 'output-order-date-range-deep-dive-page', component_property = 'children'),

        # Output start_date and end_date
        Output(component_id = "order-date-range-deep-dive-page", component_property = "start_date" ),
        Output(component_id = "order-date-range-deep-dive-page", component_property = "end_date" ),

        # Output plot
        Output(component_id = 'crossfilter-scatter', component_property = 'figure'),
        Output(component_id = 'aw-dose-scatter', component_property = 'figure'),
        Output(component_id = 'egfr-dose-scatter', component_property = 'figure'),

        # Output Plot, Table and Dropdown so it updates everytime it changes for Length of Stay
        Output(component_id = 'length-of-stay-bar', component_property = 'figure'),
        Output(component_id = 'length-of-stay-table', component_property = 'data'),

        Output(component_id = 'order-generic-los-deep-dive-page-dd', component_property = 'options'),
        Output(component_id = 'ams-indication-los-deep-dive-page-dd', component_property = 'options'),
        Output(component_id = 'frequency-los-dive-page-dd', component_property = 'options'),
        Output(component_id = 'dose-los-deep-dive-page-dd', component_property = 'options'),

        # Download csv output
        Output(component_id = 'download-2d-crossfilter-scatterplot-df', component_property = 'data'),
        Output(component_id = 'download-aw-dose-scatter-df', component_property = 'data'),
        Output(component_id = 'download-egfr-dose-scatter-df', component_property = 'data'),
        Output(component_id = 'download-length-of-stay-bar-df', component_property = 'data'),

        # Date Input Functionality
        Input(component_id = 'show-all-button-deep-dive-page', component_property = 'n_clicks'),
        Input(component_id = 'order-date-range-deep-dive-page', component_property = 'start_date'),
        Input(component_id = 'order-date-range-deep-dive-page', component_property = 'end_date'),

        # Universal Input Dropdowns
        Input(component_id = 'order-generic-deep-dive-page-dd', component_property= 'value'),
        Input(component_id = 'medical-service-deep-dive-page-dd', component_property= 'value'),
        Input(component_id = 'ward-deep-dive-page-dd', component_property= 'value'),
        Input(component_id = 'doctor-deep-dive-page-dd', component_property= 'value'),
        Input(component_id = 'ams-indication-deep-dive-page-dd', component_property= 'value'),

        # Crossfilter chart
        Input(component_id = 'crossfilter-xaxis', component_property = 'value'),
        Input(component_id = 'crossfilter-yaxis', component_property = 'value'),

        # Age, Weight, Dose and EGFR first 24 hours dose Input Dropdown
        Input(component_id = 'generic-1type-deep-dive-page-dd', component_property = 'value'),

        # Length of Stay Input Dropdown
        Input(component_id = 'order-generic-los-deep-dive-page-dd', component_property = 'value'),
        Input(component_id = 'ams-indication-los-deep-dive-page-dd', component_property = 'value'),
        Input(component_id = 'frequency-los-dive-page-dd', component_property = 'value'),
        Input(component_id = 'dose-los-deep-dive-page-dd', component_property = 'value'),

        # Download Functionality
        Input(component_id = 'download-2d-crossfilter-scatterplot-button', component_property = 'n_clicks'),
        Input(component_id = 'download-aw-dose-scatter-button', component_property = 'n_clicks'),
        Input(component_id = 'download-egfr-dose-scatter-button', component_property = 'n_clicks'),
        Input(component_id = 'download-length-of-stay-bar-button', component_property = 'n_clicks'),

        # Coming from the refresh_and_initialize_agg_data() function
        Input(component_id = 'intermediate-div-deep-dive-page', component_property = 'children')
)

def update_plot(
                show_all_button, 
                start_date, 
                end_date, 
                input_order_generic, 
                input_medical_service, 
                input_ward, 
                input_doctor, 
                input_ams_indication,
                crossfilter_xinput,
                crossfilter_yinput,
                input_generic1T,
                input_order_generic_los,
                input_ams_indication_los,
                input_frequency_los, 
                input_dose_los,
                download_2d_cross_filter_scatter_plot_n,
                download_aw_dose_scatter_n,
                download_egfr_and_first_24hr_dose_n,
                download_length_of_stay_bar_n,
                intermediate_data
                ):
    
    df_copy = read_and_transform_agg_model()

    egfr_dosef24_copy = read_and_transform_egfr_dose_first_24h_data_model()

    los_df_copy = read_and_transform_length_of_stay_data_model()

    # Unique values for los
    unique_order_generic_los = los_df_copy['ORDER_GENERIC'].dropna().unique()
    unique_ams_indication_los = los_df_copy['AMS_INDICATION'].dropna().unique()
    unique_frequency_los = los_df_copy['FREQUENCY'].dropna().unique()
    unique_dose_los = los_df_copy['DOSE'].dropna().unique()
        
    # Unique values for egfr dose first 24h
    unique_order_generic_egfr_dosef24 = egfr_dosef24_copy['ORDER_GENERIC'].dropna().unique().tolist()

    # Filter dates and update text
    df_copy, string_prefix, start_date_text, end_date_text = filter_date_and_update_text(df_copy = df_copy, start_date = start_date, end_date = end_date)

    # Assuming the date range is the same as the original dataset
    egfr_dosef24_copy, string_prefix, start_date_text, end_date_text = filter_date_and_update_text(df_copy = egfr_dosef24_copy, start_date = start_date, end_date = end_date)

    # Setup boolean values to display 3d scatterplot based on dropdown selection
    medi_bool = False
    amsindi_bool = False

    # Filter based on dimensions    
    if input_order_generic:
        # Filter
        df_copy = df_copy[df_copy["ORDER_GENERIC"].isin(input_order_generic)]
    
    if input_medical_service:
        # Filter
        df_copy = df_copy[df_copy["MEDICAL_SERVICE"].isin(input_medical_service)]
        egfr_dosef24_copy = egfr_dosef24_copy[egfr_dosef24_copy["MEDICAL_SERVICE"].isin(input_medical_service)]
        medi_bool = True
    
    if input_ward:
        # Filter
        df_copy = df_copy[df_copy["WARD"].isin(input_ward)]
        egfr_dosef24_copy = egfr_dosef24_copy[egfr_dosef24_copy["WARD"].isin(input_ward)]
    
    if input_doctor:
        # Filter
        df_copy = df_copy[df_copy["DOCTOR"].isin(input_doctor)]
        egfr_dosef24_copy = egfr_dosef24_copy[egfr_dosef24_copy["DOCTOR"].isin(input_doctor)]
    
    if input_ams_indication:
        # Filter
        df_copy = df_copy[df_copy["AMS_INDICATION"].isin(input_ams_indication)]
        egfr_dosef24_copy = egfr_dosef24_copy[egfr_dosef24_copy["AMS_INDICATION"].isin(input_ams_indication)]
        amsindi_bool = True

    # Cross filter chart plot
    cfmerge_df = create_agg_2d_scatter_plot_df(df_copy)

    # Download 2d scatterplot data
    download_2d_cross_filter_scatter_plot_val = dash.no_update
    if "download-2d-crossfilter-scatterplot-button" == ctx.triggered_id:
        download_2d_cross_filter_scatter_plot_val = dcc.send_data_frame(cfmerge_df.to_csv, "download-2d-scatterplot.csv", index = False)
    else:
        download_2d_cross_filter_scatter_plot_val = dash.no_update
    
    cross_filter_scatter_plot = px.scatter(cfmerge_df, hover_data = ['ORDER_GENERIC'], x = crossfilter_xinput, y = crossfilter_yinput)

    hoverformat = 'Order Generic: %{customdata}<br>' + crossfilter_xinput + ': %{x}<br>' + crossfilter_yinput + ': %{y}'

    cross_filter_scatter_plot.update_traces(customdata = cfmerge_df['ORDER_GENERIC'], hovertemplate = hoverformat)

    cross_filter_scatter_plot.update_layout(margin = {'l': 40, 'b': 40, 't': 10, 'r': 10}, hovermode = 'closest')

    # Filter based on dimensions for Age, Weight, Dose and EGFR dose for first 24 hours
    
    # Initialise input generic 1 type for first time view
    generic1T_value = unique_order_generic_egfr_dosef24[0]

    if input_generic1T:
        generic1T_value = input_generic1T
    
    awDose_scatter_df = df_copy[df_copy['ORDER_GENERIC'] == generic1T_value]
    egfr_dosef24_copy = egfr_dosef24_copy[egfr_dosef24_copy['ORDER_GENERIC'] == generic1T_value]
    
    # Create df
    awDose_scatter_df = awDose_scatter_df.dropna(subset=['AGE','ACTUAL_WEIGHT','TOTAL_DOSAGE'])

    # Download Age, Weight, Exact Dose Scatter Plot Data
    download_age_weight_dose_scatter_plot_val = dash.no_update
    if "download-aw-dose-scatter-button" == ctx.triggered_id:
        download_age_weight_dose_scatter_plot_val = dcc.send_data_frame(awDose_scatter_df.to_csv, "download-age-weight-exact-dose-scatterplot.csv", index = False)
    else:
        download_age_weight_dose_scatter_plot_val = dash.no_update
    
    # Download egfr and first 24h data
    download_egfr_and_first_24h_plot_val = dash.no_update
    if "download-egfr-dose-scatter-button" == ctx.triggered_id:
        download_egfr_and_first_24h_plot_val = dcc.send_data_frame(egfr_dosef24_copy.to_csv, "download-egfr-and-first-24h-plot.csv", index = False)
    else:
        download_egfr_and_first_24h_plot_val = dash.no_update

    # Filter based on dimensions for Length of Stay

    # The default group-by list is 'MRN'
    # There is a need to duplicate the 'length of stay' across different 'AMS Indications'

    los_group_by_list = ['MRN']

    if input_order_generic_los:
        # Filter
        los_df_copy = los_df_copy[los_df_copy["ORDER_GENERIC"].isin(input_order_generic_los)]

        # Update unique values for other dropdowns
        unique_ams_indication_los = los_df_copy['AMS_INDICATION'].dropna().unique()
        unique_frequency_los = los_df_copy['FREQUENCY'].dropna().unique()
        unique_dose_los = los_df_copy['DOSE'].dropna().unique()

    if input_ams_indication_los:
        # Filter
        los_df_copy = los_df_copy[los_df_copy["AMS_INDICATION"].isin(input_ams_indication_los)]

        # Update unique values for other dropdowns
        unique_order_generic_los = los_df_copy['ORDER_GENERIC'].dropna().unique()
        unique_frequency_los = los_df_copy['FREQUENCY'].dropna().unique()
        unique_dose_los = los_df_copy['DOSE'].dropna().unique()

        # Append to group by list
        # Duplicate the 'length of stay' across different 'AMS Indications'
        ams_indication_colname = 'AMS_INDICATION'

        if ams_indication_colname not in los_group_by_list:
            los_group_by_list.append(ams_indication_colname)

    if input_frequency_los:
        # Filter
        los_df_copy = los_df_copy[los_df_copy["FREQUENCY"].isin(input_frequency_los)]

        # Update unique values for other dropdowns
        unique_order_generic_los = los_df_copy['ORDER_GENERIC'].dropna().unique()
        unique_ams_indication_los = los_df_copy['AMS_INDICATION'].dropna().unique()
        unique_dose_los = los_df_copy['DOSE'].dropna().unique()

    if input_dose_los:
        # Filter
        los_df_copy = los_df_copy[los_df_copy["DOSE"].isin(input_dose_los)]

        # Update unique values for other dropdowns
        unique_order_generic_los = los_df_copy['ORDER_GENERIC'].dropna().unique()
        unique_ams_indication_los = los_df_copy['AMS_INDICATION'].dropna().unique()
        unique_frequency_los = los_df_copy['FREQUENCY'].dropna().unique()
    
    # Group by
    los_df_copy = los_df_copy.groupby(los_group_by_list).first().reset_index()

    # Update LOS dropdown Options
    unique_order_generic_los_options = [{'label': x, 'value': x} for x in unique_order_generic_los]
    unique_ams_indication_los_options = [{'label': x, 'value': x} for x in unique_ams_indication_los]
    unique_frequency_los_options = [{'label': x, 'value': x} for x in unique_frequency_los]
    unique_dose_los_options = [{'label': x, 'value': x} for x in unique_dose_los]

    #print(los_df_copy)

    # Plot LOS Bar Chart
    
    sorted_los_df_copy = los_df_copy.sort_values('LENGTH_OF_STAY', ascending=True)

    # Download length of stay data
    download_los_data_val = dash.no_update
    if "download-length-of-stay-bar-button" == ctx.triggered_id:
        download_los_data_val = dcc.send_data_frame(sorted_los_df_copy.to_csv, "download-length-of-stay-bar-plot.csv", index = False)
    else:
        download_los_data_val = dash.no_update

    los_counts_copy = sorted_los_df_copy['LENGTH_OF_STAY_DAYS'].value_counts().sort_index()

    # Create bar plot

    length_of_stay_bar = go.Figure(
        go.Bar(
            x = los_counts_copy.index,
            y = los_counts_copy.values
        )
    )

    length_of_stay_bar.update_layout(
        title = "Length of Stay vs. Number of Patients",
        xaxis_title = "Length of Stay (Days)",
        yaxis_title = "Number of Patients"
    )

    length_of_stay_df = pd.DataFrame({
        'LENGTH_OF_STAY_DAYS' : los_counts_copy.index,
        'NUMBER_OF_PATIENTS': los_counts_copy.values
    })

    length_of_stay_df['LENGTH_OF_STAY_DAYS'] =  length_of_stay_df['LENGTH_OF_STAY_DAYS'].apply(lambda x: int(x))
    length_of_stay_df['NUMBER_OF_PATIENTS'] =  length_of_stay_df['NUMBER_OF_PATIENTS'].apply(lambda x: int(x))

    # Plot Age, Weight, Dose Scatter
    if medi_bool and amsindi_bool:
        awDose_scatter = px.scatter_3d(awDose_scatter_df, x = 'AGE', y = 'ACTUAL_WEIGHT', z = 'TOTAL_DOSAGE', 
                                       color = 'MEDICAL_SERVICE', symbol = 'AMS_INDICATION',title = 'Age, Weight, and Exact Dose 3D Scatter Plot', 
                                       labels = {'AGE': 'Age','ACTUAL_WEIGHT': 'Weight','TOTAL_DOSAGE': 'Total Dosage',
                                                 'MEDICAL_SERVICE': 'Medical Service','AMS_INDICATION': 'AMS Indication'},
                                                 hover_name = "PATIENT_NAME", hover_data = ["MRN", "MEDICAL_SERVICE", "AMS_INDICATION"])
        
        awDose_scatter.update_traces(
            hovertemplate = "<b>%{hovertext}</b><br>" + # PATIENT NAME in bold
            "<b>MRN: %{customdata[0]}</b><br>" + # MRN in bold
            "<br>" + # New line
            "Age: %{x}<br>" + # Age
            "Weight: %{y}<br>" + # Weight
            "Total Dosage: %{z}<br>" + # Total Dosage
            "Medical Service: %{customdata[1]}<br>" + # Medical Service
            "AMS Indication: %{customdata[2]}"
        )

    elif medi_bool:
        awDose_scatter = px.scatter_3d(awDose_scatter_df, x='AGE', y = 'ACTUAL_WEIGHT', z = 'TOTAL_DOSAGE',
                                       color = 'MEDICAL_SERVICE', title = 'Age, Weight, and Exact Dosage 3D Scatter Plot',
                                       labels = {'AGE': 'Age', 'ACTUAL_WEIGHT': 'Weight', 'TOTAL_DOSAGE': 'Total Dosage',
                                                 'MEDICAL_SERVICE': 'Medical Service','AMS_INDICATION': 'AMS Indication'},
                                                 hover_name = "PATIENT_NAME", hover_data = ["MRN", "MEDICAL_SERVICE", "AMS_INDICATION"])
        
        awDose_scatter.update_traces(
            hovertemplate = "<b>%{hovertext}</b><br>" + # PATIENT NAME in bold
            "<b>MRN: %{customdata[0]}</b><br>" + # MRN in bold
            "<br>" + # New line
            "Age: %{x}<br>" + # Age
            "Weight: %{y}<br>" + # Weight
            "Total Dosage: %{z}<br>" + # Total Dosage
            "Medical Service: %{customdata[1]}<br>" + # Medical Service
            "AMS Indication: %{customdata[2]}"
        )

    elif amsindi_bool:
        awDose_scatter = px.scatter_3d(awDose_scatter_df, x = 'AGE', y = 'ACTUAL_WEIGHT', z = 'TOTAL_DOSAGE',
                                       color = 'AMS_INDICATION',title = 'Age, Weight, and Exact Dose 3D Scatter Plot', 
                                       labels = {'AGE': 'Age','ACTUAL_WEIGHT': 'Weight','TOTAL_DOSAGE': 'Total Dosage',
                                       'MEDICAL_SERVICE': 'Medical Service','AMS_INDICATION': 'AMS Indication'},
                                       hover_name = "PATIENT_NAME", hover_data = ["MRN", "MEDICAL_SERVICE", "AMS_INDICATION"])
        
        awDose_scatter.update_traces(
            hovertemplate = "<b>%{hovertext}</b><br>" + # PATIENT NAME in bold
            "<b>MRN: %{customdata[0]}</b><br>" + # MRN in bold
            "<br>" + # New line
            "Age: %{x}<br>" + # Age
            "Weight: %{y}<br>" + # Weight
            "Total Dosage: %{z}<br>" + # Total Dosage
            "Medical Service: %{customdata[1]}<br>" + # Medical Service
            "AMS Indication: %{customdata[2]}"
        )

    else:
        awDose_scatter = px.scatter_3d(awDose_scatter_df, x = 'AGE', y = 'ACTUAL_WEIGHT', z = 'TOTAL_DOSAGE',
                                       title = 'Age, Weight, and Exact Dose 3D Scatter Plot', 
                                       labels = {'AGE': 'Age','ACTUAL_WEIGHT': 'Weight','TOTAL_DOSAGE': 'Total Dosage',
                                       'MEDICAL_SERVICE': 'Medical Service','AMS_INDICATION': 'AMS Indication'}, 
                                       hover_name = "PATIENT_NAME", hover_data = ["MRN", "MEDICAL_SERVICE", "AMS_INDICATION"])
        
        awDose_scatter.update_traces(
            hovertemplate = "<b>%{hovertext}</b><br>" + # PATIENT NAME in bold
            "<b>MRN: %{customdata[0]}</b><br>" + # MRN in bold
            "<br>" + # New line
            "Age: %{x}<br>" + # Age
            "Weight: %{y}<br>" + # Weight
            "Total Dosage: %{z}<br>" + # Total Dosage
            "Medical Service: %{customdata[1]}<br>" + # Medical Service
            "AMS Indication: %{customdata[2]}"
        )
    
    # Plot EGFR, Dose Scatter
    egfrDose_scatter = px.scatter(
        egfr_dosef24_copy, 
        x = 'EGFR', 
        y = 'FIRST 24hr DOSE', 
        title = 'EGFR and First 24 hr Dose',
        labels = {
            'EGFR': 'EGFR',
            'FIRST 24hr DOSE': 'First 24hr Dose'
        })

    return string_prefix, start_date_text, end_date_text, cross_filter_scatter_plot, awDose_scatter, egfrDose_scatter, \
            length_of_stay_bar, length_of_stay_df.to_dict('records'), \
                unique_order_generic_los_options, unique_ams_indication_los_options, \
                unique_frequency_los_options, unique_dose_los_options, \
                download_2d_cross_filter_scatter_plot_val, download_age_weight_dose_scatter_plot_val, \
                download_egfr_and_first_24h_plot_val, download_los_data_val
                

# Update x axis time series for each hover

@callback(
    # Time series from X Scatter Plot Chart
    Output(component_id = 'x-time-series', component_property = 'figure'),

    # Date Input Functionality
    Input(component_id = 'show-all-button-deep-dive-page', component_property = 'n_clicks'),
    Input(component_id = 'order-date-range-deep-dive-page', component_property = 'start_date'),
    Input(component_id = 'order-date-range-deep-dive-page', component_property = 'end_date'),

    # Universal Input Dropdowns
    Input(component_id = 'order-generic-deep-dive-page-dd', component_property= 'value'),
    Input(component_id = 'medical-service-deep-dive-page-dd', component_property= 'value'),
    Input(component_id = 'ward-deep-dive-page-dd', component_property= 'value'),
    Input(component_id = 'doctor-deep-dive-page-dd', component_property= 'value'),
    Input(component_id = 'ams-indication-deep-dive-page-dd', component_property= 'value'),

    # Hover Input
    Input(component_id = 'crossfilter-scatter', component_property = 'hoverData'),

    # X axis Input
    Input(component_id = 'crossfilter-xaxis', component_property = 'value'))

def update_x_timeseries(
                        show_all_button, 
                        start_date, 
                        end_date, 
                        input_order_generic, 
                        input_medical_service, 
                        input_ward, 
                        input_doctor, 
                        input_ams_indication,
                        hoverData, 
                        crossfilter_xaxis
                        ):
    antibio_name = hoverData['points'][0]['customdata']

    df_copy = read_and_transform_agg_model()

    # Filter dates and update text
    df_copy, _string_prefix, _start_date_text, _end_date_text = filter_date_and_update_text(df_copy = df_copy, start_date = start_date, end_date = end_date)

    # Filter based on dimensions    
    if input_order_generic:
        # Filter
        df_copy = df_copy[df_copy["ORDER_GENERIC"].isin(input_order_generic)]
    
    if input_medical_service:
        # Filter
        df_copy = df_copy[df_copy["MEDICAL_SERVICE"].isin(input_medical_service)]
    
    if input_ward:
        # Filter
        df_copy = df_copy[df_copy["WARD"].isin(input_ward)]
    
    if input_doctor:
        # Filter
        df_copy = df_copy[df_copy["DOCTOR"].isin(input_doctor)]
    
    if input_ams_indication:
        # Filter
        df_copy = df_copy[df_copy["AMS_INDICATION"].isin(input_ams_indication)]

    tsdf = df_copy[df_copy['ORDER_GENERIC'] == antibio_name]

    tsmergeDF = create_agg_time_series_plot_df(tsdf)

    # Set title for the plot annotation
    title = '<b>{}</b><br>{}'.format(antibio_name, crossfilter_xaxis)
    
    return draw_time_series(tsmergeDF, title, crossfilter_xaxis)

# Update y axis time series for each hover

@callback(
    # Time Series from Y Scatter Plot Chart
    Output(component_id = 'y-time-series', component_property = 'figure'),

    # Date Input Functionality
    Input(component_id = 'show-all-button-deep-dive-page', component_property = 'n_clicks'),
    Input(component_id = 'order-date-range-deep-dive-page', component_property = 'start_date'),
    Input(component_id = 'order-date-range-deep-dive-page', component_property = 'end_date'),

    # Universal Input Dropdowns
    Input(component_id = 'order-generic-deep-dive-page-dd', component_property= 'value'),
    Input(component_id = 'medical-service-deep-dive-page-dd', component_property= 'value'),
    Input(component_id = 'ward-deep-dive-page-dd', component_property= 'value'),
    Input(component_id = 'doctor-deep-dive-page-dd', component_property= 'value'),
    Input(component_id = 'ams-indication-deep-dive-page-dd', component_property= 'value'),

    # Hover Input
    Input(component_id = 'crossfilter-scatter', component_property = 'hoverData'),

    # Y axis Input
    Input(component_id = 'crossfilter-yaxis', component_property = 'value'))

def update_y_timeseries(
                        show_all_button, 
                        start_date, 
                        end_date, 
                        input_order_generic, 
                        input_medical_service, 
                        input_ward, 
                        input_doctor, 
                        input_ams_indication,
                        hoverData, 
                        crossfilter_yaxis):
    
    antibio_name = hoverData['points'][0]['customdata']

    df_copy = read_and_transform_agg_model()

    # Filter dates and update text
    df_copy, _string_prefix, _start_date_text, _end_date_text = filter_date_and_update_text(df_copy = df_copy, start_date = start_date, end_date = end_date)

    # Filter based on dimensions    
    if input_order_generic:
        # Filter
        df_copy = df_copy[df_copy["ORDER_GENERIC"].isin(input_order_generic)]
    
    if input_medical_service:
        # Filter
        df_copy = df_copy[df_copy["MEDICAL_SERVICE"].isin(input_medical_service)]
    
    if input_ward:
        # Filter
        df_copy = df_copy[df_copy["WARD"].isin(input_ward)]
    
    if input_doctor:
        # Filter
        df_copy = df_copy[df_copy["DOCTOR"].isin(input_doctor)]
    
    if input_ams_indication:
        # Filter
        df_copy = df_copy[df_copy["AMS_INDICATION"].isin(input_ams_indication)]


    tsdf = df_copy[df_copy['ORDER_GENERIC'] == antibio_name]

    tsmergeDF = create_agg_time_series_plot_df(tsdf)

    # Set title for the plot annotation
    title='<b>{}</b><br>{}'.format(antibio_name, crossfilter_yaxis)
    
    return draw_time_series(tsmergeDF, title, crossfilter_yaxis)
