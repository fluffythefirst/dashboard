import pandas as pd
from datetime import datetime

def transform_dot(df, filepath = 'ASC_SCORES.csv'):
    asc_scores = pd.read_csv(filepath) # read asc score table
    df2 = df.loc[:, ['MRN', 'ADMIT_DATE', 'ORDER_GENERIC', 'START_DTTM', 'STOP_DTTM', 'MEDICAL_SERVICE', 'AMS_INDICATION']]  # select relevant attributes only
    df2.dropna(subset=['MRN', 'ADMIT_DATE', 'ORDER_GENERIC', 'START_DTTM', 'STOP_DTTM'], inplace=True)  # Drop record if any of these fields contain nulls
    df3 = df2.loc[:, ['MRN', 'ADMIT_DATE', 'ORDER_GENERIC', 'START_DTTM', 'STOP_DTTM']]

    df3.sort_values(by=['MRN', 'ADMIT_DATE', 'ORDER_GENERIC', 'START_DTTM'], inplace=True)
    df3['ADMIT_DATE'] = pd.to_datetime(df3['ADMIT_DATE'], dayfirst= True, format = '%d/%m/%Y %H:%M').dt.date  # truncate time
    df3['START_DTTM'] = pd.to_datetime(df3['START_DTTM'], dayfirst= True, format = '%d/%m/%Y %H:%M').dt.date  # truncate time
    df3['STOP_DTTM'] = pd.to_datetime(df3['STOP_DTTM'], dayfirst= True, format = '%d/%m/%Y %H:%M').dt.date  # truncate time
    df4 = df3.drop_duplicates(subset=['MRN', 'ADMIT_DATE', 'ORDER_GENERIC', 'START_DTTM', 'STOP_DTTM'])  # de-duplicate on date

    final_df = pd.DataFrame(columns=['MRN', 'ADMIT_DATE', 'ORDER_GENERIC', 'START_DTTM', 'STOP_DTTM'])  # create df5 to store results

    # set empty variables
    current_mrn = None
    current_admit_date = None
    current_antibiotic = None
    current_start_date = None
    current_stop_date = None
    current_medical_service = None
    current_ams_indication = None

    # Create a list to store DataFrames to concatenate
    dfs_to_concat = []

    # loop through each row
    for index, row in df4.iterrows():
        if current_mrn is None:
            # Initialize current variables for the first row
            current_mrn = row['MRN']
            current_admit_date = row['ADMIT_DATE']
            current_antibiotic = row['ORDER_GENERIC']
            current_start_date = row['START_DTTM']
            current_stop_date = row['STOP_DTTM']
        else:
            # if next mrn, admit_date, order_generic don't match, append
            if (current_mrn != row['MRN'] or current_admit_date != row['ADMIT_DATE'] or current_antibiotic != row['ORDER_GENERIC']):
                dfs_to_concat.append(pd.DataFrame({'MRN': current_mrn,
                                                'ADMIT_DATE': current_admit_date,
                                                'ORDER_GENERIC': current_antibiotic,
                                                'START_DTTM': current_start_date,
                                                'STOP_DTTM': current_stop_date}, index=[0]))
                current_mrn = row['MRN']
                current_admit_date = row['ADMIT_DATE']
                current_antibiotic = row['ORDER_GENERIC']
                current_start_date = row['START_DTTM']
                current_stop_date = row['STOP_DTTM']
            # if next mrn, admit_date, order_generic match
            elif (current_mrn == row['MRN'] and
                current_admit_date == row['ADMIT_DATE'] and
                current_antibiotic == row['ORDER_GENERIC']):
                # if start date overlaps with last treatment and the stop date extends, change stop date to new max.
                if row['START_DTTM'] <= current_stop_date and row['STOP_DTTM'] > current_stop_date:
                    current_stop_date = row['STOP_DTTM']  # Extend the current stop date if there's an overlap
                # if start date overlaps with last treatment and the stop date doesn't extend
                elif row['START_DTTM'] <= current_stop_date and row['STOP_DTTM'] <= current_stop_date:
                    pass
                # if no overlap, append
                elif row['START_DTTM'] > current_stop_date:
                    dfs_to_concat.append(pd.DataFrame({'MRN': current_mrn,
                                                    'ADMIT_DATE': current_admit_date,
                                                    'ORDER_GENERIC': current_antibiotic,
                                                    'START_DTTM': current_start_date,
                                                    'STOP_DTTM': current_stop_date}, index=[0]))
                    current_mrn = row['MRN']
                    current_admit_date = row['ADMIT_DATE']
                    current_antibiotic = row['ORDER_GENERIC']
                    current_start_date = row['START_DTTM']
                    current_stop_date = row['STOP_DTTM']

    # Append last row
    if current_mrn is not None:
        dfs_to_concat.append(pd.DataFrame({'MRN': current_mrn,
                                        'ADMIT_DATE': current_admit_date,
                                        'ORDER_GENERIC': current_antibiotic,
                                        'START_DTTM': current_start_date,
                                        'STOP_DTTM': current_stop_date}, index=[0]))

    # Concatenate all DataFrames
    final_df = pd.concat(dfs_to_concat, ignore_index=True)

    # Calculate days of therapy metric
    # final_df['DAYS_OF_THERAPY'] = (pd.to_datetime(final_df['STOP_DTTM']) - pd.to_datetime(final_df['START_DTTM'])) + pd.Timedelta(days=1)
    # final_df['DAYS_OF_THERAPY'] = pd.to_datetime(final_df['DAYS_OF_THERAPY']).dt.days  # stop date - start date + 1 day
    final_df['DAYS_OF_THERAPY'] = (pd.to_datetime(final_df['STOP_DTTM'], dayfirst= True, format = '%d/%m/%Y %H:%M') - pd.to_datetime(final_df['START_DTTM'], dayfirst= True, format = '%d/%m/%Y %H:%M')).dt.days + 1
    final_df = final_df.groupby(['MRN', 'ADMIT_DATE', 'ORDER_GENERIC'])['DAYS_OF_THERAPY'].sum().reset_index()  # sum up therapy days for same order_generic types per admission

    # Merge the 'ASC_SCORE' column from 'asc_scores' into 'grouped_df' based on 'ORDER_GENERIC'
    final_df = final_df.merge(asc_scores[['ORDER_GENERIC', 'ASC_SCORE']], on='ORDER_GENERIC', how='left')
    final_df['dASC'] = final_df['DAYS_OF_THERAPY'] * final_df['ASC_SCORE']

    # Find the earliest start timestamp for each antibiotic type per admission
    df2['START_DTTM'] = pd.to_datetime(df2['START_DTTM'], dayfirst= True, format = '%d/%m/%Y %H:%M') # convert to datetime
    earliest_start = df2.groupby(['MRN', 'ADMIT_DATE', 'ORDER_GENERIC'])['START_DTTM'].idxmin()  # df2 is with timestamp for ordering
    df2['ADMIT_DATE'] = pd.to_datetime(df2['ADMIT_DATE'], dayfirst= True, format = '%d/%m/%Y %H:%M').dt.date  # truncate timestamp
    df2 = df2.loc[earliest_start, ['MRN', 'ADMIT_DATE', 'ORDER_GENERIC', 'MEDICAL_SERVICE', 'AMS_INDICATION']]  # Select corresponding medical service and ams indication

    # Left join
    result_df = final_df.merge(df2, on=['MRN', 'ADMIT_DATE', 'ORDER_GENERIC'], how='left')
    return result_df
    