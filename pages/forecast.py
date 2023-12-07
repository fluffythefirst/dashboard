import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import dash
from dash import dcc, html, Input, Output, Dash, State, callback, ctx, dash_table
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import plotly.express as px
import plotly.io as pio

import re

dash.register_page(__name__, order = 3)

pio.templates.default = "plotly_white"

## -- SETUP GLOBAL VARIABLES -- 

ADMIT_DATE_DATE_FORMAT = '%d/%m/%Y'
DISCHARGE_DATE_DATE_FORMAT = '%d/%m/%Y %H:%M'
ORDER_DATE_DATE_FORMAT = '%d/%m/%Y %H:%M'
CSV_READ = './orders_data_clean/final_df.csv'

## -- END SETUP GLOBAL VARIABLES -- 

## --- START AGG DATA MODEL ---

def read_and_transform_agg_model():
    df = pd.read_csv(CSV_READ)
    # print("df READ - forecast page")

    # rename for shorter labels
    df.rename(columns={'LOCATION_OF_PATIENT_AT_THE_TIME_OF_ORDER': 'WARD'}, inplace=True)
    df.rename(columns={'ATTENDING_MEDICAL_OFFICER': 'DOCTOR'}, inplace=True)
    df.rename(columns={'total_DDD': 'TOTAL_DDD'}, inplace=True)
    df.rename(columns={'total_dosage': 'TOTAL_DOSAGE'}, inplace=True)
    df.rename(columns={'ORDER_PLACED_DATE': 'ORDER_DATE'}, inplace=True)
    df.rename(columns={'DAYS_OF_THERAPY': 'TOTAL_DOT'}, inplace=True)

    df['DOCTOR'] = df['DOCTOR'].str.replace(r'\s*\([^)]*\)$', '', regex=True) # remove (MO)(SMO)

    filtered_df = df.loc[:, ['MRN', 'ORDER_DATE', 'ORDER_STATUS','ORDER_GENERIC','MEDICAL_SERVICE','WARD','AMS_INDICATION','DOCTOR','TOTAL_DDD','TOTAL_DOSAGE','TOTAL_DOT', 'AGE', 'ACTUAL_WEIGHT', 'dASC', 'PATIENT_NAME']] # select relevant attributes only

    filtered_df['ORDER_DATE'] = pd.to_datetime(filtered_df['ORDER_DATE'], format = ORDER_DATE_DATE_FORMAT )
    
    filtered_df['ORDER_MONTH_YEAR'] = pd.to_datetime(filtered_df['ORDER_DATE']).dt.strftime('%Y-%m')

    return filtered_df.copy(deep = True)

## --- END AGG DATA MODEL ---

filtered_df = read_and_transform_agg_model()

## -- START UTILITIES HELPER FUNCTIONS --

def filter_date_and_update_text(df_copy, start_date, end_date, show_all_button_id = "show-all-button-forecast-page"):
    
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

## -- END UTILITIES HELPER FUNCTIONS -- 

## -- START FORECAST HELPER FUNCTIONS --

def create_mean_dASC_trend_forecast_version_df(df_copy, group_category):
    # Drop NA
    dASC_df = df_copy.dropna(subset=['dASC'])
    # Group by
    dASC_grouped_df = dASC_df.groupby([group_category,'ORDER_MONTH_YEAR'])['dASC'].mean().reset_index(name='MEAN_dASC')

    # Remove instances with fewer than four ORDER_MONTH_YEAR, as they cannot be forecasted
    dASC_grouped_df_filtered = dASC_grouped_df.groupby([group_category]).filter(lambda x: len(x) > 3)

    return dASC_grouped_df_filtered.copy(deep = True)


def create_mean_dASC_trend_forecast_version_plot_from_grouped_df(dASC_grouped_df_filtered, group_category, group_category_text, trend_text, max_actual_month_year = None):
    # Plot
    mean_dASC_trend_plot = px.line(dASC_grouped_df_filtered, x='ORDER_MONTH_YEAR', y='MEAN_dASC', color= group_category,\
                                 markers = True)
    # Update layout
    distinct_months_years = dASC_grouped_df_filtered['ORDER_MONTH_YEAR'].unique()
    mean_dASC_trend_plot.update_xaxes(tickvals=distinct_months_years, \
                                    ticktext=[pd.to_datetime(date).strftime('%b %Y') for date in distinct_months_years])
    
    mean_dASC_trend_plot.update_layout(title="Mean dASC {} Trend by {}".format(trend_text, group_category_text))
    mean_dASC_trend_plot.update_layout(xaxis_title="Order Month Year")
    mean_dASC_trend_plot.update_layout(yaxis_title="Mean dASC")
    mean_dASC_trend_plot.update_layout(legend_title=dict(text = "{}".format(group_category_text)), font=dict(size=9))

    # Only shows the dotted line only if max_actual_month_year is provided it contains forecasted data 
    # The line is added only if the dataframe is not empty

    if not dASC_grouped_df_filtered.empty and max_actual_month_year and max_actual_month_year != max(dASC_grouped_df_filtered['ORDER_MONTH_YEAR']):
        mean_dASC_trend_plot.add_vline(
            x = max_actual_month_year,
            opacity = 1,
            line_width = 1.5, 
            line_dash = "dash",
            line_color = "blue"
        )

    return mean_dASC_trend_plot

def split_trend_lines(mean_dASC_trend_df):
    delta_changes = []
    order_generic_list = mean_dASC_trend_df["ORDER_GENERIC"].unique().tolist() 
    
    mean_dASC_trend_df_columns = mean_dASC_trend_df.columns.tolist()

    increasing_trend = pd.DataFrame(columns = mean_dASC_trend_df_columns)
    stable_trend = pd.DataFrame(columns = mean_dASC_trend_df_columns)
    decreasing_trend = pd.DataFrame(columns = mean_dASC_trend_df_columns)

    for order_generic in order_generic_list:
        mean_dASC_trend_df_order_genericA = mean_dASC_trend_df[mean_dASC_trend_df["ORDER_GENERIC"] == order_generic]
        min_index = np.where(mean_dASC_trend_df_order_genericA["ORDER_MONTH_YEAR"] == \
                        mean_dASC_trend_df_order_genericA["ORDER_MONTH_YEAR"].min())
        min_row = mean_dASC_trend_df_order_genericA.iloc[min_index]
        max_index = np.where(mean_dASC_trend_df_order_genericA["ORDER_MONTH_YEAR"] == \
                        mean_dASC_trend_df_order_genericA["ORDER_MONTH_YEAR"].max())
        max_row =  mean_dASC_trend_df_order_genericA.iloc[max_index]
        delta_change = float(max_row["MEAN_dASC"].iloc[0]) - float(min_row["MEAN_dASC"].iloc[0])
        delta_changes.append(delta_change)

        """
        Antibiotic dASC trend is stable if delta_change is between -1 to 1
        Antibiotic dASC trend is increasing if delta_change is > 1
        Antibiotic dASC trend is decreasing if delta_change is < 1       
        """

        # Need to create a copy if the dataframe is empty. See: 
        # https://stackoverflow.com/questions/77254777/alternative-to-concat-of-empty-dataframe-now-that-it-is-being-deprecated

        if delta_change >= -1 and delta_change <= 1:
            # Concat to the stable dataframe
            for index, row in mean_dASC_trend_df_order_genericA.iterrows():
                row_df = pd.DataFrame([row])

                if stable_trend.empty:
                    stable_trend = row_df.copy()
                else:
                    stable_trend = pd.concat([stable_trend, row_df], ignore_index=True)
        
        if delta_change > 1:
            # Concat to the increasing dataframe
            for index, row in mean_dASC_trend_df_order_genericA.iterrows():
                row_df = pd.DataFrame([row])

                if increasing_trend.empty:
                    increasing_trend = row_df.copy()
                else:
                    increasing_trend = pd.concat([increasing_trend, row_df], ignore_index=True)
        
        if delta_change < 1:
            # Concat to the decreasing dataframe
            for index, row in mean_dASC_trend_df_order_genericA.iterrows():
                row_df = pd.DataFrame([row])

                if decreasing_trend.empty:
                    decreasing_trend = row_df.copy()
                else:
                    decreasing_trend = pd.concat([decreasing_trend, row_df], ignore_index=True)

    return increasing_trend, stable_trend, decreasing_trend

## -- END FORECAST HELPER FUNCTIONS -- 

## --- START FORECAST LAYOUT VARIABLES ---

# By default, order_generic_list contains the full list with at least four time stamps        
mean_dASC_trend_df = create_mean_dASC_trend_forecast_version_df(filtered_df, "ORDER_GENERIC")
unique_order_generic = mean_dASC_trend_df["ORDER_GENERIC"].unique()

## --- END FORECAST LAYOUT VARIABLES ---


## -- START LAYOUT ---

layout = html.Div([
    html.Div(id='intermediate-div-forecast-page', style={'display': 'none'}),

    dbc.Row(
            [
                dbc.Col(id = 'output-order-date-range-forecast-page', style = {'font-weight': 'bold'}),
                dbc.Col(dcc.DatePickerRange(
                    id = 'order-date-range-forecast-page',
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
                        id = "show-all-button-forecast-page",
                        color = "primary"
                        )
                    ]
                )
            ]
        ),

    html.H3("Forecast", style = {'font-family': "Helvetica Neue"}),
    html.Div("For the forecast page, we retain only those grouped categories that have at least four timetamp data points, as we cannot generate a forecast for categories with fewer than four data points."),
    html.Div("For a complete view of the trend chart, inclusive of the entire dataset, please refer to the High-Level Trends page."),    
    dcc.Dropdown(
        id = 'order-generic-forecast-page-dd',
        options = [{'label': x, 'value': x} for x in unique_order_generic],
        multi = True,
        optionHeight = 50                            
    ),

    dcc.Graph(id = "dASC-order-generic-plot-increasing-forecast-page"),
    dcc.Graph(id = "dASC-order-generic-plot-stable-forecast-page"),
    dcc.Graph(id = "dASC-order-generic-plot-decreasing-forecast-page"),

    html.Span(id = "forecast-output-forecast-page", style = {"display": "flex", "justifyContent": "center", "alignItems": "center", "height": "100%"}),
    html.Br(),
    dbc.Row([
        dbc.Col(dbc.Button("Reset to Original", id="reset-to-original-button-forecast-page"), width="auto"),
        dbc.Col(dbc.Button("Forecast for 6 Months", id="forecast-6-months-button-forecast-page"), width="auto"),
        dbc.Col(dbc.Button("Forecast for 12 Months", id="forecast-12-months-button-forecast-page"), width="auto"),
        dbc.Col([
            dbc.DropdownMenu(
                [
                    dbc.DropdownMenuItem(
                        "Original", id = "download-to-csv-button-forecast-page"
                    ),
                    dbc.DropdownMenuItem(
                        "Original + 6 Month Forecast", id = "download-to-csv-button-6-month-forecast-page"
                    ),
                    dbc.DropdownMenuItem(
                        "Original + 12 Month Forecast", id = "download-to-csv-button-12-month-forecast-page"
                    )
                ],  label = "Download to csv", id = "dropdown-menu-download-to-csv-button-forecast-page"
            )
        ], width = "auto"
           
        ),
        dcc.Download(id = "download-dataframe-csv-forecast-page")
    ], justify = "center")
])

## -- END LAYOUT ---

@callback( 
        # Initialize order date range
        Output('order-date-range-forecast-page', 'min_date_allowed'),
        Output('order-date-range-forecast-page', 'max_date_allowed'),

        # Initialize forecast dropdown data
        Output('order-generic-forecast-page-dd', 'options'),

        # Trigger other callback to initialize the charts
        Output('intermediate-div-forecast-page', 'children'),

        # From memory
        Input('memory', 'data')
)

def refresh_and_initialize_forecast_data(data):
    """
    Callback function to initialize layout once the user has uploaded the new dataset from memory.
    This function executes with every refresh but provides no updates unless there is a change within 
    the update_data() page. 
    If a change occurs, it will populate the forecast layout but not the graphs.
    This function also outputs intermediate_data as the input for the update_plot() function, 
    which will update the plot with the new dataset.
    
    :param data: Data from memory.
    
    :return: 
        - min_date: Minimum ORDER_DATE.
        - max_date: Maximum ORDER_DATE.
        - unique_order_generic_options: A list of dictionaries, each containing unique order generics with at least 4 months of data.
        - intermediate_data: Current timestamp.
    """

    filtered_df = None

    min_date = dash.no_update
    max_date = dash.no_update

    unique_order_generic_options = dash.no_update

    intermediate_data = dash.no_update

    if data == "dataset-uploaded":

        filtered_df = read_and_transform_agg_model()
        #print("refresh_and_initialize_forecast_data() from forecast page")

        min_date = min(filtered_df["ORDER_DATE"])
        max_date = max(filtered_df["ORDER_DATE"])

        # By default, order_generic_list contains the full list with at least four time stamps        
        mean_dASC_trend_df = create_mean_dASC_trend_forecast_version_df(filtered_df, "ORDER_GENERIC")
        unique_order_generic = mean_dASC_trend_df["ORDER_GENERIC"].unique()
    
        unique_order_generic_options = [{'label': x, 'value': x} for x in unique_order_generic]

        intermediate_data = str(pd.Timestamp.now())
    
    return min_date, max_date, \
        unique_order_generic_options,  \
        intermediate_data

@callback(
        # Output text for order date range
        Output(component_id = 'output-order-date-range-forecast-page', component_property = 'children'),

        # Output start_date and end_date
        Output(component_id = "order-date-range-forecast-page", component_property = "start_date" ),
        Output(component_id = "order-date-range-forecast-page", component_property = "end_date" ),

        # Output text and plot

        # Text to show if it is an original data or forecasted data
        Output(component_id = "forecast-output-forecast-page", component_property = "children"),
        
        # mean dASC by Order Generic Plot for incrasing, stable, and decreasing values
        Output(component_id = "dASC-order-generic-plot-increasing-forecast-page", component_property = "figure"),
        Output(component_id = "dASC-order-generic-plot-stable-forecast-page", component_property = "figure"),
        Output(component_id = "dASC-order-generic-plot-decreasing-forecast-page", component_property = "figure"),

        # Download csv output
        Output(component_id = 'download-dataframe-csv-forecast-page', component_property = 'data'),

        # Date functionality
        Input(component_id = 'show-all-button-forecast-page', component_property = 'n_clicks'),
        Input(component_id = 'order-date-range-forecast-page', component_property = 'start_date'),
        Input(component_id = 'order-date-range-forecast-page', component_property = 'end_date'),

        # Forecast buttons
        Input(component_id = "reset-to-original-button-forecast-page", component_property = "n_clicks"),
        Input(component_id = "forecast-6-months-button-forecast-page", component_property = "n_clicks"),
        Input(component_id = "forecast-12-months-button-forecast-page", component_property = "n_clicks"),

        # Dropdowns
        Input(component_id = 'order-generic-forecast-page-dd', component_property = 'value'),

        # Download to csv button
        Input(component_id = 'download-to-csv-button-forecast-page', component_property = 'n_clicks'),
        Input(component_id = 'download-to-csv-button-6-month-forecast-page', component_property = 'n_clicks'),
        Input(component_id = 'download-to-csv-button-12-month-forecast-page', component_property = 'n_clicks'),

        # Coming from the refresh_and_initialize_agg_data() function
        Input(component_id = 'intermediate-div-forecast-page', component_property = 'children')
)

def update_plot(
                show_all_button, 
                start_date, 
                end_date,
                n,
                n2,
                n3,
                input_order_generic, 
                download_to_csv_n,
                download_to_csv_2_n,
                download_to_csv_3_n,
                intermediate_data
                ):
    
    df_copy = read_and_transform_agg_model()
    #print("update_plot() from forecast page")

    # Filter date and update text for date selection
    df_copy, string_prefix, start_date_text, end_date_text = filter_date_and_update_text(df_copy, start_date, end_date)

    # By default, order_generic_list contains the full list with at least four time stamps        
    mean_dASC_trend_df = create_mean_dASC_trend_forecast_version_df(df_copy, "ORDER_GENERIC")
    order_generic_list = list(mean_dASC_trend_df["ORDER_GENERIC"].unique())

    if input_order_generic:
        # Filter df_copy
        df_copy = df_copy[df_copy["ORDER_GENERIC"].isin(input_order_generic)]

        # Select order_generic_list
        order_generic_list = input_order_generic
    
    mean_dASC_trend_df = create_mean_dASC_trend_forecast_version_df(df_copy, "ORDER_GENERIC")

    # Find the maximum month and year value from the date filter
    max_actual_month_year = None

    if not mean_dASC_trend_df.empty:
        max_actual_month_year = max(mean_dASC_trend_df['ORDER_MONTH_YEAR'].unique())

    mean_dASC_trend_df_6_months_forecast = mean_dASC_trend_df.copy(deep = True)
    mean_dASC_trend_df_12_months_forecast = mean_dASC_trend_df.copy(deep = True)

    # Forecast for 6 and 12 months
    for order_generic in order_generic_list:
        order_generic_A = mean_dASC_trend_df[mean_dASC_trend_df["ORDER_GENERIC"] == order_generic]
        order_generic_A_mean_dASC = order_generic_A["MEAN_dASC"].to_list()

        # Create time axis 
        # Time axis will be just the index of the Mean dASC's
        months =  np.array(range(1, len(order_generic_A_mean_dASC) + 1)).reshape(-1, 1)

        # Fit the model
        model = LinearRegression()
        model.fit(months, order_generic_A_mean_dASC)

        # Get the Order Month Year of order_generic_A
        max_date = max(pd.to_datetime(order_generic_A["ORDER_MONTH_YEAR"]))

        # Predict 6 months into the future
        future_6_months = np.array(range(len(order_generic_A_mean_dASC) + 1, len(order_generic_A_mean_dASC) + 7)).reshape(-1, 1)
        predictions_6_months = model.predict(future_6_months)
        date_series_6_months = [max_date + pd.DateOffset(months=i) for i in range(1, 7)]
        date_str_series_6_months = [date.strftime('%Y-%m') for date in date_series_6_months]
        new_6_months_data = {
                'ORDER_GENERIC': [order_generic] * 6,
                'ORDER_MONTH_YEAR': date_str_series_6_months,
                'MEAN_dASC': predictions_6_months
        }
        new_6_months_df = pd.DataFrame(new_6_months_data)

        # Predict 12 months into the future
        future_12_months = np.array(range(len(order_generic_A_mean_dASC) + 1, len(order_generic_A_mean_dASC) + 13)).reshape(-1, 1)
        predictions_12_months = model.predict(future_12_months)
        date_series_12_months = [max_date + pd.DateOffset(months=i) for i in range(1, 13)]
        date_str_series_12_months = [date.strftime('%Y-%m') for date in date_series_12_months]
        new_12_months_data = {
                'ORDER_GENERIC': [order_generic] * 12,
                'ORDER_MONTH_YEAR': date_str_series_12_months,
                'MEAN_dASC': predictions_12_months
        }
        new_12_months_df = pd.DataFrame(new_12_months_data)

        # Add order_generic_A to the new grouped dataframe
        mean_dASC_trend_df_6_months_forecast = pd.concat([mean_dASC_trend_df_6_months_forecast, new_6_months_df], ignore_index = True)
        mean_dASC_trend_df_12_months_forecast = pd.concat([mean_dASC_trend_df_12_months_forecast, new_12_months_df], ignore_index = True)

    

    # Split the trend lines
    increasing_trend, stable_trend, decreasing_trend = split_trend_lines(mean_dASC_trend_df)
   
    increasing_trend_6_months_forecast, stable_trend_6_months_forecast, decreasing_trend_6_months_forecast = \
            split_trend_lines(mean_dASC_trend_df_6_months_forecast)
    
    increasing_trend_12_months_forecast, stable_trend_12_months_forecast, decreasing_trend_12_months_forecast = \
            split_trend_lines(mean_dASC_trend_df_12_months_forecast)

    # # Forecast button is not clicked
    # # Display the forecasted plot
    forecast_output_text = "Original Data"
    increasing_trend_selection = increasing_trend
    stable_trend_selection = stable_trend
    decreasing_trend_selection = decreasing_trend

    # output_mean_dASC_trend_plot = mean_dASC_trend_plot

    # Forecast button is clicked
    if "forecast-6-months-button-forecast-page" == ctx.triggered_id:
        forecast_output_text = "Forecasted for 6 Month's (The Last 6 Month's Data Represents the Forecast)"
        increasing_trend_selection = increasing_trend_6_months_forecast
        stable_trend_selection = stable_trend_6_months_forecast
        decreasing_trend_selection = decreasing_trend_6_months_forecast
        #print("FORECAST 6 MONTH CLICKED")
    
    if "forecast-12-months-button-forecast-page" == ctx.triggered_id:
        forecast_output_text = "Forecasted for 12 Month's (The Last 12 Month's Data Represents the Forecast)"
        increasing_trend_selection = increasing_trend_12_months_forecast
        stable_trend_selection = stable_trend_12_months_forecast
        decreasing_trend_selection = decreasing_trend_12_months_forecast
        #print("FORECAST 12 MONTH CLICKED")
    
    mean_dASC_increasing_trend_plot = \
        create_mean_dASC_trend_forecast_version_plot_from_grouped_df(increasing_trend_selection, "ORDER_GENERIC", "Order Generic", "Increasing", max_actual_month_year)

    mean_dASC_stable_trend_plot = \
        create_mean_dASC_trend_forecast_version_plot_from_grouped_df(stable_trend_selection, "ORDER_GENERIC", "Order Generic", "Stable", max_actual_month_year)

    mean_dASC_decreasing_trend_plot = \
        create_mean_dASC_trend_forecast_version_plot_from_grouped_df(decreasing_trend_selection, "ORDER_GENERIC", "Order Generic", "Decreasing", max_actual_month_year)

    combined_trend_df = pd.concat([increasing_trend_selection, stable_trend_selection, decreasing_trend_selection], ignore_index = True)

    download_val = dash.no_update

    # If Original download button is pressed
    if "download-to-csv-button-forecast-page" == ctx.triggered_id:
        download_val = dcc.send_data_frame(combined_trend_df.to_csv, "download-forecast.csv", index = False)
    
    # If Original + 6 Month Forecast download button is pressed
    elif "download-to-csv-button-6-month-forecast-page" == ctx.triggered_id:
        increasing_trend_selection = increasing_trend_6_months_forecast
        stable_trend_selection = stable_trend_6_months_forecast
        decreasing_trend_selection = decreasing_trend_6_months_forecast

        combined_trend_df = pd.concat([increasing_trend_selection, stable_trend_selection, decreasing_trend_selection], ignore_index = True)

        download_val = dcc.send_data_frame(combined_trend_df.to_csv, "download-forecast.csv", index = False)
 
    # If Original + 12 Month Forecast download button is pressed
    elif "download-to-csv-button-12-month-forecast-page" == ctx.triggered_id:
        increasing_trend_selection = increasing_trend_12_months_forecast
        stable_trend_selection = stable_trend_12_months_forecast
        decreasing_trend_selection = decreasing_trend_12_months_forecast

        combined_trend_df = pd.concat([increasing_trend_selection, stable_trend_selection, decreasing_trend_selection], ignore_index = True)

        download_val = dcc.send_data_frame(combined_trend_df.to_csv, "download-forecast.csv", index = False)
    
    elif "reset-to-original-button-forecast-page" == ctx.triggered_id:
        download_val = dash.no_update

    else:
        download_val = dash.no_update

    return string_prefix, start_date_text, end_date_text,\
            forecast_output_text,\
            mean_dASC_increasing_trend_plot, mean_dASC_stable_trend_plot, mean_dASC_decreasing_trend_plot, \
                download_val


