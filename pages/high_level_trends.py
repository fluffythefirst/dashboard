import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, Dash, State, callback, ctx, dash_table
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import plotly.express as px
import plotly.io as pio

import re

# Import utility functions
from utility_function import DataModel

dash.register_page(__name__, order = 1)

pio.templates.default = "plotly_white"

## -- SETUP GLOBAL VARIABLES -- 

ADMIT_DATE_DATE_FORMAT = '%d/%m/%Y'
DISCHARGE_DATE_DATE_FORMAT = '%d/%m/%Y %H:%M'
ORDER_DATE_DATE_FORMAT = '%d/%m/%Y %H:%M'
CSV_READ = './orders_data_clean/final_df.csv'

## -- END SETUP GLOBAL VARIABLES -- 

## --- START DATA MODEL INITIALISATION ---

data_model = DataModel(CSV_READ, ORDER_DATE_DATE_FORMAT)

## --- END DATA MODEL INITIALISATION ---

## -- START AGG DATA MODEL -- 

filtered_df = data_model.read_and_transform_agg_model()

## -- END AGG DATA MODEL --

## --- START AGG DATA MODEL INITIALISATION ---

unique_order_status = filtered_df["ORDER_STATUS"].dropna().unique()
unique_order_generic = filtered_df["ORDER_GENERIC"].dropna().unique()
unique_medical_service = filtered_df["MEDICAL_SERVICE"].dropna().unique()
unique_ward = filtered_df["WARD"].dropna().unique()
unique_doctor = filtered_df["DOCTOR"].dropna().unique()
unique_ams_indication = filtered_df["AMS_INDICATION"].dropna().unique()

top_trend_selection = [{'label' :'Order Generic', 'value': 'ORDER_GENERIC'}, 
                       {'label': 'Medical Service', 'value': 'MEDICAL_SERVICE'}, 
                       {'label': 'Ward', 'value': 'WARD' }, 
                       {'label': 'Doctors', 'value': 'DOCTOR'},
                       {'label': 'AMS Indication', 'value': 'AMS_INDICATION'}]

## --- END AGG DATA MODEL INITIALISATION ---

## -- START LAYOUT HELPER FUNCTIONS --

def generate_trends_tab(label_text, id_1, id_2, id_3, id_4):
    trends_tab = dbc.Tab(
        children = [
            dbc.CardGroup([
                dbc.Card([dbc.CardBody(dcc.Graph(id = id_1))])
            ]),
            dbc.CardGroup([
                dbc.Card([dbc.CardBody(dcc.Graph(id = id_2))])
            ]),     
            dbc.CardGroup([
                dbc.Card([dbc.CardBody(dcc.Graph(id = id_3))])
            ]),
            dbc.CardGroup([
                dbc.Card([dbc.CardBody(dcc.Graph(id = id_4))])
            ])
        ]
        ,label = label_text
    )

    return trends_tab

def create_mean_dASC_trend_plot(df_copy, group_category, group_category_text):
    # Drop NA
    dASC_df = df_copy.dropna(subset=['dASC'])
    # Group by
    grouped_df = dASC_df.groupby([group_category,'ORDER_MONTH_YEAR'])['dASC'].mean().reset_index(name='MEAN_dASC')
    # Plot
    mean_dASC_trend_plot = px.line(grouped_df, x='ORDER_MONTH_YEAR', y='MEAN_dASC', color= group_category,\
                                 markers = True)
    # Update layout
    distinct_months_years = grouped_df['ORDER_MONTH_YEAR'].unique()
    mean_dASC_trend_plot.update_xaxes(tickvals=distinct_months_years, \
                                    ticktext=[pd.to_datetime(date).strftime('%b %Y') for date in distinct_months_years])
    
    mean_dASC_trend_plot.update_layout(title="Mean dASC Trend by {}".format(group_category_text))
    mean_dASC_trend_plot.update_layout(xaxis_title="Order Month Year")
    mean_dASC_trend_plot.update_layout(yaxis_title="Mean dASC")
    mean_dASC_trend_plot.update_layout(legend_title=dict(text = "{}".format(group_category_text)), font=dict(size=9))

    return mean_dASC_trend_plot

def create_total_DOT_trend_plot(df_copy, group_category, group_category_text):
    # Drop NA
    dot_df = df_copy.dropna(subset=['TOTAL_DOT'])
    # Group by
    grouped_df = dot_df.groupby([group_category,'ORDER_MONTH_YEAR'])['TOTAL_DOT'].sum().reset_index(name='TOTAL_DOT')
    # Plot
    total_DOT_trend_plot = px.line(grouped_df, x='ORDER_MONTH_YEAR', y='TOTAL_DOT', color=group_category,\
                                markers = True)
    # Update layout
    distinct_months_years = grouped_df['ORDER_MONTH_YEAR'].unique()
    total_DOT_trend_plot.update_xaxes(tickvals=distinct_months_years, \
                                    ticktext=[pd.to_datetime(date).strftime('%b %Y') for date in distinct_months_years])
    
    total_DOT_trend_plot.update_layout(title = "Total DOT Trend by {}".format(group_category_text))
    total_DOT_trend_plot.update_layout(xaxis_title="Order Month Year")
    total_DOT_trend_plot.update_layout(yaxis_title="Total DOT")
    total_DOT_trend_plot.update_layout(legend_title=dict(text = "{}".format(group_category_text)), font=dict(size=9))

    return total_DOT_trend_plot

def create_total_DDD_trend_plot(df_copy, group_category, group_category_text):
    # Drop NA
    ddd_df = df_copy.dropna(subset=['TOTAL_DDD'])
    # Group by
    grouped_df = ddd_df.groupby([group_category,'ORDER_MONTH_YEAR'])['TOTAL_DDD'].sum().reset_index(name='TOTAL_DDD')
    # Plot
    total_DDD_trend_plot = px.line(grouped_df, x='ORDER_MONTH_YEAR', y='TOTAL_DDD', color=group_category,\
                                 markers = True)
    # Update layout
    distinct_months_years = grouped_df['ORDER_MONTH_YEAR'].unique()
    total_DDD_trend_plot.update_xaxes(tickvals=distinct_months_years, \
                                    ticktext=[pd.to_datetime(date).strftime('%b %Y') for date in distinct_months_years])
    
    total_DDD_trend_plot.update_layout(title = "Total DDD Trend by {}".format(group_category_text))
    total_DDD_trend_plot.update_layout(xaxis_title="Order Month Year")
    total_DDD_trend_plot.update_layout(yaxis_title="Total DDD")
    total_DDD_trend_plot.update_layout(legend_title=dict(text = "{}".format(group_category_text)), font=dict(size=9))

    return total_DDD_trend_plot

def create_avg_dot_trend_plot(df_copy, group_category, group_category_text):
    # Drop NA
    dot_df = df_copy.dropna(subset=['TOTAL_DOT'])
    # Group by
    grouped_df = dot_df.groupby([group_category,'ORDER_MONTH_YEAR'])['TOTAL_DOT'].mean().reset_index(name='AVG_DOT')
    # Plot
    avg_dot_trend_plot = px.line(grouped_df, x='ORDER_MONTH_YEAR', y='AVG_DOT', color=group_category,\
                                 markers = True)
    # Update layout
    distinct_months_years = grouped_df['ORDER_MONTH_YEAR'].unique()
    avg_dot_trend_plot.update_xaxes(tickvals=distinct_months_years, \
                                    ticktext=[pd.to_datetime(date).strftime('%b %Y') for date in distinct_months_years])
    
    avg_dot_trend_plot.update_layout(title="Average DOT Trend by {}".format(group_category_text))
    avg_dot_trend_plot.update_layout(xaxis_title="Order Month Year")
    avg_dot_trend_plot.update_layout(yaxis_title="Average DOT")
    avg_dot_trend_plot.update_layout(legend_title=dict(text = "{}".format(group_category_text)), font=dict(size=9))

    return avg_dot_trend_plot


def create_trend_plots_by_category(df_copy, group_category, group_category_text):

    mean_dASC_trend_plot_by_category = create_mean_dASC_trend_plot(df_copy, group_category, group_category_text)
    total_DOT_trend_plot_by_category = create_total_DOT_trend_plot(df_copy, group_category, group_category_text)
    total_DDD_trend_plot_by_category = create_total_DDD_trend_plot(df_copy, group_category, group_category_text)   
    avg_dot_trend_plot_by_category = create_avg_dot_trend_plot(df_copy, group_category, group_category_text)

    return mean_dASC_trend_plot_by_category, total_DOT_trend_plot_by_category, \
        total_DDD_trend_plot_by_category, avg_dot_trend_plot_by_category

def filter_date_and_update_text(df_copy, start_date, end_date, show_all_button_id = "show-all-button"):
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


def create_top_trend_df(df_copy, group_category):
    # Drop NA
    dASC_df = df_copy.dropna(subset=['dASC'])
    # Group by
    grouped_dASC_df = dASC_df.groupby([group_category,'ORDER_MONTH_YEAR'])['dASC'].mean().reset_index(name='MEAN_dASC')
    # Drop NA
    dot_df = df_copy.dropna(subset=['TOTAL_DOT'])
    # Group by
    grouped_dot_df = dot_df.groupby([group_category,'ORDER_MONTH_YEAR'])['TOTAL_DOT'].sum().reset_index(name='TOTAL_DOT')
    # Merge the two
    merged_df = pd.merge(grouped_dASC_df, grouped_dot_df, on = [group_category, 'ORDER_MONTH_YEAR'], how = 'outer')
    # Drop NA
    ddd_df = df_copy.dropna(subset=['TOTAL_DDD'])
    # Group by
    grouped_ddd_df = ddd_df.groupby([group_category,'ORDER_MONTH_YEAR'])['TOTAL_DDD'].sum().reset_index(name='TOTAL_DDD')
    # Merge the three
    merged_df = pd.merge(merged_df, grouped_ddd_df, on = [group_category, 'ORDER_MONTH_YEAR'], how = 'outer')
    # Drop NA
    dot_df = df_copy.dropna(subset=['TOTAL_DOT'])
    # Group by
    grouped_dot_df = dot_df.groupby([group_category,'ORDER_MONTH_YEAR'])['TOTAL_DOT'].mean().reset_index(name='AVG_DOT')
    # Merge the four
    merged_df = pd.merge(merged_df, grouped_dot_df, on = [group_category, 'ORDER_MONTH_YEAR'], how = 'outer')
    
    # Round to two decimal places
    merged_df = merged_df.round({'MEAN_dASC': 2, 'TOTAL_DOT': 2, 'TOTAL_DDD': 2, 'AVG_DOT': 2})
    
    return merged_df.copy(deep = True)


## -- END LAYOUT HELPER FUNCTIONS -- 

## --- START LAYOUT ---

layout = html.Div(
            [
            html.Div(id='intermediate-div', style={'display': 'none'}),
            html.H3('Antimicrobial Consumption - High Level Trends'),

            dbc.Row(
                [
                    dbc.Col(id = 'output-order-date-range', style = {'font-weight': 'bold'}),
                    dbc.Col(dcc.DatePickerRange(
                        id = 'order-date-range',
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
                            id = "show-all-button",
                            color = "primary"
                            )
                        ]
                    )
                ]
            ),

            html.Div([
                html.H4("Top Trend Month Year", style = {'font-family': "Helvetica Neue"}),
                dbc.Col(
                    dbc.RadioItems(
                        id = 'top-trend-selection', 
                        options = top_trend_selection,
                        value = top_trend_selection[0]['value'], 
                        inline = True
                    )
                ),
                html.Div(id = "output-trend-table")]
            ),

            html.H4("Antimicrobial Consumption Trends"),

            dbc.Row(
                [
                    dbc.Col([
                            dbc.Col("Order Status"),
                            dcc.Dropdown(
                                id = 'order-status-high-level-page-dd',
                                options = [{'label': x, 'value': x} for x in unique_order_status],
                                multi = True                            )
                    ]),
                    dbc.Col([
                            dbc.Col("Order Generic"),
                            dcc.Dropdown(
                                id = 'order-generic-high-level-page-dd',
                                options = [{'label': x, 'value': x} for x in unique_order_generic],
                                multi = True,
                                optionHeight = 50                            
                            )
                    ]),
                    dbc.Col([
                            dbc.Col("Medical Service"),
                            dcc.Dropdown(
                                id = 'medical-service-high-level-page-dd',
                                options = [{'label': x, 'value': x} for x in unique_medical_service],
                                multi = True,
                                optionHeight = 50
                            )
                    ]),
                    dbc.Col([
                            dbc.Col("Ward"),
                            dcc.Dropdown(
                                id = 'ward-high-level-page-dd',
                                options = [{'label': x, 'value': x} for x in unique_ward],
                                multi = True,
                                optionHeight = 50
                            )
                    ]),
                    dbc.Col([
                            dbc.Col("Doctor"),
                            dcc.Dropdown(
                                id = 'doctor-high-level-page-dd',
                                options = [{'label': x, 'value': x} for x in unique_doctor],
                                multi = True,
                                optionHeight = 50
                            )
                    ]),
                    dbc.Col([
                            dbc.Col("AMS Indication"),
                            dcc.Dropdown(
                                id = 'ams-indication-high-level-page-dd',
                                options = [{'label': x, 'value': x} for x in unique_ams_indication],
                                multi = True,
                                optionHeight = 60
                            )
                    ]),
                ],     
            ),

            dbc.Tabs(children = [
                    generate_trends_tab(
                        label_text = "Trend by Order Generic", 
                        id_1 = "dASC-order-generic-plot", 
                        id_2 = "total-dot-order-generic-plot",
                        id_3 = "total-ddd-order-generic-plot",
                        id_4 = "avg-dot-order-generic-plot"),
                    
                    generate_trends_tab(
                        label_text = "Trend by Medical Service", 
                        id_1 = "dASC-medical-service-plot", 
                        id_2 = "total-dot-medical-service-plot",
                        id_3 = "total-ddd-medical-service-plot",
                        id_4 = "avg-dot-medical-service-plot"),
                    
                     generate_trends_tab(
                        label_text = "Trend by Ward", 
                        id_1 = "dASC-ward-plot", 
                        id_2 = "total-dot-ward-plot",
                        id_3 = "total-ddd-ward-plot",
                        id_4 = "avg-dot-ward-plot"),

                     generate_trends_tab(
                        label_text = "Trend by Doctor", 
                        id_1 = "dASC-doctor-plot", 
                        id_2 = "total-dot-doctor-plot",
                        id_3 = "total-ddd-doctor-plot",
                        id_4 = "avg-dot-doctor-plot"),

                    generate_trends_tab(
                        label_text = "Trend by AMS Indication", 
                        id_1 = "dASC-ams-indication-plot", 
                        id_2 = "total-dot-ams-indication-plot",
                        id_3 = "total-ddd-ams-indication-plot",
                        id_4 = "avg-dot-ams-indication-plot")
                ]
            ),
            dbc.Button("Download csv", id = "btn-csv"),
            dcc.Download(id = "download-dataframe-csv")
         ]
)

## --- END LAYOUT ---

@callback( 
        # Initialize order date range
        Output('order-date-range', 'min_date_allowed'),
        Output('order-date-range', 'max_date_allowed'),

        # Trigger other callback to initialize the charts
        Output('intermediate-div', 'children'),

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

    min_date = dash.no_update
    max_date = dash.no_update

    intermediate_data = dash.no_update

    if data == "dataset-uploaded":

        filtered_df = data_model.read_and_transform_agg_model()

        min_date = min(filtered_df["ORDER_DATE"])
        max_date = max(filtered_df["ORDER_DATE"])

        intermediate_data = str(pd.Timestamp.now())

    return min_date, max_date, \
        intermediate_data

# Make the filters dynamic

@callback(
        # Output Dropdown Update Options
        Output(component_id = 'order-status-high-level-page-dd', component_property = 'options'),
        Output(component_id = 'order-generic-high-level-page-dd', component_property = 'options'),
        Output(component_id = 'medical-service-high-level-page-dd', component_property = 'options'),
        Output(component_id = 'ward-high-level-page-dd', component_property = 'options'),
        Output(component_id = 'doctor-high-level-page-dd', component_property = 'options'),
        Output(component_id = 'ams-indication-high-level-page-dd', component_property = 'options'),

        # Date functionality
        Input(component_id = 'show-all-button', component_property = 'n_clicks'),
        Input(component_id = 'order-date-range', component_property = 'start_date'),
        Input(component_id = 'order-date-range', component_property = 'end_date'),

        # Input Dropdown Values
        Input(component_id = 'order-status-high-level-page-dd', component_property = 'value'),
        Input(component_id = 'order-generic-high-level-page-dd', component_property = 'value'),
        Input(component_id = 'medical-service-high-level-page-dd', component_property = 'value'),
        Input(component_id = 'ward-high-level-page-dd', component_property = 'value'),
        Input(component_id = 'doctor-high-level-page-dd', component_property = 'value'),
        Input(component_id = 'ams-indication-high-level-page-dd', component_property = 'value'),

        # Coming from refresh_and_initialize_agg_data() function
        Input(component_id = 'intermediate-div', component_property = 'children')

)

def update_main_filter_values(
                              show_all_button, 
                              start_date, 
                              end_date, 
                              input_order_status,
                              input_order_generic, 
                              input_medical_service, 
                              input_ward, 
                              input_doctor, 
                              input_ams_indication,
                              intermediate_data,
                            ):
    
    df_copy = data_model.read_and_transform_agg_model()

    # Filter date and update text for date selection
    df_copy, _string_prefix, _start_date_text, _end_date_text = filter_date_and_update_text(df_copy, start_date, end_date)

    # Initialise unique values
    unique_order_status = sorted(df_copy["ORDER_STATUS"].dropna().unique())
    unique_order_generic = sorted(df_copy["ORDER_GENERIC"].dropna().unique())
    unique_medical_service = sorted(df_copy["MEDICAL_SERVICE"].dropna().unique())
    unique_ward = sorted(df_copy["WARD"].dropna().unique())
    unique_doctor = sorted(df_copy["DOCTOR"].dropna().unique())
    unique_ams_indication = sorted(df_copy["AMS_INDICATION"].dropna().unique())

    # Filter based on dimensions
    if input_order_status:
        # Filter
        df_copy = df_copy[df_copy["ORDER_STATUS"].isin(input_order_status)]

        # Update unique values for other dropdowns
        unique_order_generic = sorted(df_copy["ORDER_GENERIC"].dropna().unique())
        unique_medical_service = sorted(df_copy["MEDICAL_SERVICE"].dropna().unique())
        unique_ward = sorted(df_copy["WARD"].dropna().unique())
        unique_doctor = sorted(df_copy["DOCTOR"].dropna().unique())
        unique_ams_indication = sorted(df_copy["AMS_INDICATION"].dropna().unique())
    
    if input_order_generic:
        # Filter
        df_copy = df_copy[df_copy["ORDER_GENERIC"].isin(input_order_generic)]

        # Update unique values for other dropdowns
        unique_order_status = sorted(df_copy["ORDER_STATUS"].dropna().unique())
        unique_medical_service = sorted(df_copy["MEDICAL_SERVICE"].dropna().unique())
        unique_ward = sorted(df_copy["WARD"].dropna().unique())
        unique_doctor = sorted(df_copy["DOCTOR"].dropna().unique())
        unique_ams_indication = sorted(df_copy["AMS_INDICATION"].dropna().unique())
    
    if input_medical_service:
        # Filter
        df_copy = df_copy[df_copy["MEDICAL_SERVICE"].isin(input_medical_service)]

        # Update unique values for other dropdowns
        unique_order_status = sorted(df_copy["ORDER_STATUS"].dropna().unique())
        unique_order_generic = sorted(df_copy["ORDER_GENERIC"].dropna().unique())
        unique_ward = sorted(df_copy["WARD"].dropna().unique())
        unique_doctor = sorted(df_copy["DOCTOR"].dropna().unique())
        unique_ams_indication = sorted(df_copy["AMS_INDICATION"].dropna().unique())
    
    if input_ward:
        # Filter
        df_copy = df_copy[df_copy["WARD"].isin(input_ward)]

        # Update unique values for other dropdowns
        unique_order_status = sorted(df_copy["ORDER_STATUS"].dropna().unique())
        unique_order_generic = sorted(df_copy["ORDER_GENERIC"].dropna().unique())
        unique_medical_service = sorted(df_copy["MEDICAL_SERVICE"].dropna().unique())
        unique_doctor = sorted(df_copy["DOCTOR"].dropna().unique())
        unique_ams_indication = sorted(df_copy["AMS_INDICATION"].dropna().unique())
    
    if input_doctor:
        # Filter
        df_copy = df_copy[df_copy["DOCTOR"].isin(input_doctor)]

        # Update unique values for other dropdowns
        unique_order_status = sorted(df_copy["ORDER_STATUS"].dropna().unique())
        unique_order_generic = sorted(df_copy["ORDER_GENERIC"].dropna().unique())
        unique_medical_service = sorted(df_copy["MEDICAL_SERVICE"].dropna().unique())
        unique_ward = sorted(df_copy["WARD"].dropna().unique())
        unique_ams_indication = sorted(df_copy["AMS_INDICATION"].dropna().unique())
    
    if input_ams_indication:
        # Filter
        df_copy = df_copy[df_copy["AMS_INDICATION"].isin(input_ams_indication)]

        unique_order_status = sorted(df_copy["ORDER_STATUS"].dropna().unique())
        unique_order_generic = sorted(df_copy["ORDER_GENERIC"].dropna().unique())
        unique_medical_service = sorted(df_copy["MEDICAL_SERVICE"].dropna().unique())
        unique_ward = sorted(df_copy["WARD"].dropna().unique())
        unique_doctor = sorted(df_copy["DOCTOR"].dropna().unique())

    unique_order_status_options = [{'label': x, 'value': x} for x in unique_order_status]
    unique_order_generic_options = [{'label': x, 'value': x} for x in unique_order_generic]
    unique_medical_service_options = [{'label': x, 'value':x} for x in unique_medical_service]
    unique_ward_options = [{'label': x, 'value': x} for x in unique_ward]
    unique_doctor_options = [{'label': x, 'value': x} for x in unique_doctor]
    unique_ams_indication_options = [{'label': x, 'value': x} for x in unique_ams_indication]

    # print(df_copy)

    return unique_order_status_options, unique_order_generic_options, unique_medical_service_options, \
           unique_ward_options, unique_doctor_options, unique_ams_indication_options

@callback(
        # Output text for order date range
        Output(component_id = 'output-order-date-range', component_property = 'children'),

        # Output start_date and end_date
        Output(component_id = "order-date-range", component_property = "start_date" ),
        Output(component_id = "order-date-range", component_property = "end_date" ),

        # Output plot
        # Trend by Order Generic
        Output(component_id = "dASC-order-generic-plot", component_property = "figure"),
        Output(component_id = "total-dot-order-generic-plot", component_property = "figure"),
        Output(component_id = "total-ddd-order-generic-plot", component_property = "figure"),
        Output(component_id = "avg-dot-order-generic-plot", component_property = "figure"),

        # Trend by Medical Service
        Output(component_id = "dASC-medical-service-plot", component_property = "figure"),
        Output(component_id = "total-dot-medical-service-plot", component_property = "figure"),
        Output(component_id = "total-ddd-medical-service-plot", component_property = "figure"),
        Output(component_id = "avg-dot-medical-service-plot", component_property = "figure"),

        # Trend by Ward
        Output(component_id = "dASC-ward-plot", component_property = "figure"),
        Output(component_id = "total-dot-ward-plot", component_property = "figure"),
        Output(component_id = "total-ddd-ward-plot", component_property = "figure"),
        Output(component_id = "avg-dot-ward-plot", component_property = "figure"),

        # Trend by Doctor
        Output(component_id = "dASC-doctor-plot", component_property = "figure"),
        Output(component_id = "total-dot-doctor-plot", component_property = "figure"),
        Output(component_id = "total-ddd-doctor-plot", component_property = "figure"),
        Output(component_id = "avg-dot-doctor-plot", component_property = "figure"),

        # Trend by AMS Indication
        Output(component_id = "dASC-ams-indication-plot", component_property = "figure"),
        Output(component_id = "total-dot-ams-indication-plot", component_property = "figure"),
        Output(component_id = "total-ddd-ams-indication-plot", component_property = "figure"),
        Output(component_id = "avg-dot-ams-indication-plot", component_property = "figure"),

        # Download csv output
        Output(component_id = "download-dataframe-csv", component_property = "data"),

        # Date functionality
        Input(component_id = 'show-all-button', component_property = 'n_clicks'),
        Input(component_id = 'order-date-range', component_property = 'start_date'),
        Input(component_id = 'order-date-range', component_property = 'end_date'),

        # Dropdowns
        Input(component_id = 'order-status-high-level-page-dd', component_property= 'value'),
        Input(component_id = 'order-generic-high-level-page-dd', component_property= 'value'),
        Input(component_id = 'medical-service-high-level-page-dd', component_property= 'value'),
        Input(component_id = 'ward-high-level-page-dd', component_property= 'value'),
        Input(component_id = 'doctor-high-level-page-dd', component_property= 'value'),
        Input(component_id = 'ams-indication-high-level-page-dd', component_property= 'value'),

        # Download functionality
        Input(component_id = 'btn-csv', component_property = 'n_clicks'),

        # Coming from the refresh_and_initialize_agg_data() function
        Input(component_id = 'intermediate-div', component_property = 'children')
)

def update_plot(
                show_all_button, 
                start_date, 
                end_date, 
                input_order_status,
                input_order_generic, 
                input_medical_service, 
                input_ward, 
                input_doctor, 
                input_ams_indication,
                n_clicks_csv, 
                intermediate_data
                ):
    
    df_copy = data_model.read_and_transform_agg_model()

    # Filter date and update text for date selection
    df_copy, string_prefix, start_date_text, end_date_text = filter_date_and_update_text(df_copy, start_date, end_date)

    # Filter based on dimensions
    if input_order_status:
        # Filter
        df_copy = df_copy[df_copy["ORDER_STATUS"].isin(input_order_status)]
    
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
    
    # Download csv functionality
    download_val = None

    if "btn-csv" == ctx.triggered_id:
        download_val = dcc.send_data_frame(df_copy.to_csv, "download-high-level-trends.csv", index = False)
    else:
        download_val = dash.no_update

    # Create Trend Plots

    # Trend Plot By Order Generic
    mean_dASC_by_order_generic_trend_plot,  \
    total_DOT_by_order_generic_trend_plot, \
    total_DDD_by_order_generic_trend_plot, \
    avg_dot_by_order_generic_trend_plot = create_trend_plots_by_category(df_copy, 'ORDER_GENERIC', 'Order Generic')

    # Trend Plot By Medical Service
    mean_dASC_by_medical_service_trend_plot,  \
    total_DOT_by_medical_service_trend_plot, \
    total_DDD_by_medical_service_trend_plot, \
    avg_dot_by_medical_service_trend_plot = create_trend_plots_by_category(df_copy, 'MEDICAL_SERVICE', 'Medical Service')

    # Trend Plot By Ward
    mean_dASC_by_ward_trend_plot,  \
    total_DOT_by_ward_trend_plot, \
    total_DDD_by_ward_trend_plot, \
    avg_dot_by_ward_trend_plot = create_trend_plots_by_category(df_copy, 'WARD', 'Ward')

    # Trend Plot By Ward
    mean_dASC_by_doctor_trend_plot,  \
    total_DOT_by_doctor_trend_plot, \
    total_DDD_by_doctor_trend_plot, \
    avg_dot_by_doctor_trend_plot = create_trend_plots_by_category(df_copy, 'DOCTOR', 'Doctor')

    # Trend Plot By AMS Indication
    mean_dASC_by_ams_indication_trend_plot,  \
    total_DOT_by_ams_indication_trend_plot, \
    total_DDD_by_ams_indication_trend_plot, \
    avg_dot_by_ams_indication_trend_plot = create_trend_plots_by_category(df_copy, 'AMS_INDICATION', 'AMS Indication')


    return  string_prefix, start_date_text, end_date_text,\
            mean_dASC_by_order_generic_trend_plot, total_DOT_by_order_generic_trend_plot, \
            total_DDD_by_order_generic_trend_plot, avg_dot_by_order_generic_trend_plot, \
            mean_dASC_by_medical_service_trend_plot, total_DOT_by_medical_service_trend_plot, \
            total_DDD_by_medical_service_trend_plot, avg_dot_by_medical_service_trend_plot, \
            mean_dASC_by_ward_trend_plot, total_DOT_by_ward_trend_plot, \
            total_DDD_by_ward_trend_plot, avg_dot_by_ward_trend_plot, \
            mean_dASC_by_doctor_trend_plot, total_DOT_by_doctor_trend_plot, \
            total_DDD_by_doctor_trend_plot, avg_dot_by_doctor_trend_plot, \
            mean_dASC_by_ams_indication_trend_plot, total_DOT_by_ams_indication_trend_plot, \
            total_DDD_by_ams_indication_trend_plot, avg_dot_by_ams_indication_trend_plot, \
            download_val


@callback (
    # Top Trend Table Output        
    Output(component_id = 'output-trend-table', component_property = 'children'),

    # Date Functionality
    Input(component_id = 'show-all-button', component_property = 'n_clicks'),
    Input(component_id = 'order-date-range', component_property = 'start_date'),
    Input(component_id = 'order-date-range', component_property = 'end_date'),

    # Radio Button Input
    Input(component_id = 'top-trend-selection', component_property = 'value')
)

def update_datatable_trends(show_all_button, start_date, end_date, value):

    df_copy = data_model.read_and_transform_agg_model()

    df_copy, _string_prefix, _start_date_text, _end_date_text = filter_date_and_update_text(df_copy, start_date, end_date)

    top_trend_df = create_top_trend_df(df_copy, value)

    top_trend_df_columns_dict = {
        'ORDER_GENERIC': 'Order Generic',
        'MEDICAL_SERVICE': 'Medical Service',
        'WARD': 'Ward',
        'DOCTOR': 'Doctor',
        'AMS_INDICATION': 'AMS Indication',
        'ORDER_MONTH_YEAR': 'Order Month Year',
        'MEAN_dASC': 'Mean dASC',
        'TOTAL_DOT': 'Total DOT',
        'TOTAL_DDD': 'Total DDD',
        'AVG_DOT': 'Average DOT'
    }

    columns = [
            {"name":top_trend_df_columns_dict[i] , "id": i, "deletable": False, "selectable": True} for i in top_trend_df.columns
    ]

    return html.Div([
        dash_table.DataTable(
            id = 'datatable-trends',
            columns= columns,
            data = top_trend_df.to_dict('records'),
            editable = False,
            filter_action = "native",
            sort_action = "native",
            sort_mode = "single",
            row_deletable = False,
            selected_columns = [],
            selected_rows = [],
            page_action = "native",
            page_current = 0,
            page_size = 5,
            style_cell = {'font-family': 'Helvetica Neue', 'fontSize': 14},
            style_header = {
                'backgroundColor': 'rgb(173,216,230)',
                'fontWeight': 'bold'
            }
        )
    ])