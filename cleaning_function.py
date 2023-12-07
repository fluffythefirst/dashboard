import numpy as np
import pandas as pd
import datetime
# extrac numbers from string
def extract_num(df):
    #df["MRN"] = df["MRN"].str.extract(r'(\d+)')
    df["AGE"] = df["AGE"].str.extract(r'(\d+)').astype('Int64')
    df["ACTUAL_WEIGHT"] = df["ACTUAL_WEIGHT"].str.extract(r'(\d+)').astype('float')
    df["HEIGHT"] = df["HEIGHT"].str.extract(r'(\d+)').astype('float')
    return df

# convert '>90' to 91. Note: the values are stored as float
def convert_to_91(col):
    col = col.str.extract(r'([<>]?\d+)').replace('>90','91').astype('float')    
    return col

# leave as '>90'. Note: the values are stored as string here
def greater_than_90(col):
    col = col.str.extract(r'([<>]?\d+)').apply(lambda x: x.str.zfill(2))
    return col

# processing missing value in order generic
def fill_generic(df):
    # find the index where order_generic is missing
    missing_index = set(df[df['ORDER_GENERIC'].isna()].index)
    
    # find the index where order_name contains 'sterile' before the + sign
    sterile_ind = df[df['ORDER_NAME'].str.split('+', n = 1).str[0].str.contains('sterile')].index
    
    # exclude index which also appear in sterile_ind
    ind = list(missing_index.difference(set(sterile_ind)))
    
    # extract the first word in order name as order generic
    df.loc[ind,'ORDER_GENERIC'] = df.loc[ind, 'ORDER_NAME'].str.split().str[0]
    
    # extract the first word right after the + sign as order generic
    df.loc[sterile_ind,'ORDER_GENERIC'] = df.loc[sterile_ind, 'ORDER_NAME'].str.split('+',n = 1).str[1].str.split().str[0]
    
    # capitalise the first letter in order generic
    df['ORDER_GENERIC'] = df['ORDER_GENERIC'].str.lower().str.capitalize()
    return df

# filling in missing values in start/stop_dttm
def fill_dttm(df):
    # if frequency == once, and start_dttm is missing, use order place date to fill start and stop dttm
    once_missing_index = df[(df['FREQUENCY'] == 'ONCE') & (df['START_DTTM'].isna())].index
    df.loc[once_missing_index, 'START_DTTM'] = df.loc[once_missing_index, 'ORDER_PLACED_DATE']
    df.loc[once_missing_index, 'STOP_DTTM'] = df.loc[once_missing_index, 'ORDER_PLACED_DATE']
    return df

# coverting datetime
def date_processing(df):
    to_date = ['ORDER_PLACED_DATE','ADMIT_DATE','DISCHARGE_DATE','START_DTTM','STOP_DTTM']

#     # convert to date only
#     # for i in to_date:
#     #     df[i] = df[i].str.split().str[0]
#     #     df[i] = pd.to_datetime(df[i], format = '%d/%m/%Y')

    # convert to date + time
    for i in to_date:
        df[i] = pd.to_datetime(df[i], dayfirst= True, format = '%d/%m/%Y %H:%M').dt.strftime('%d/%m/%Y %H:%M')
        
    return df

# transforming frequency to numeric values
def frequency_transform(df):
    # only need string before bucket, then strip and lower case
    df['FREQUENCY'] = df['FREQUENCY'].str.split('(',n = 1).str[0].str.strip()
    df['FREQUENCY'] = df['FREQUENCY'].str.lower()
    
    # replace any string with once a week xxxxx with once a week
    once_a_week = df['FREQUENCY'].str.contains('once a week', na = False)
    df.loc[once_a_week,'FREQUENCY'] = 'once a week'
    
    # transformation
    df['FREQUENCY'] = df['FREQUENCY'].apply(clean_frequency)
    return df 


# cleaning dictionary for frequency
def clean_frequency(freq):
    mapping = {
        '1 hourly': 24,
        '12 hourly': 2,
        '2 hourly': 12,
        '24 hourly': 1,
        '3 hourly': 8,
        '36 hourly': 1/1.5,
        '4 hourly': 6,
        '48 hourly': 1/2,
        '6 hourly': 4,
        '8 hourly': 3,
        'afternoon': 1,
        'alternate days': 0.5,
        'bd': 2,
        'daily': 1,
        'evening': 1,
        'every 3 days': 1/3,
        'every 3 weeks': 1/21,
        'every 4 weeks': 1/28,
        'five times a day': 5,
        'four times a week ': 4/7,
        'midday': 1,
        'mid-morning': 1,
        'monthly': 1/28,
        'morning': 1,
        'night': 1,
        'once': 1,
        'once a week': 1/7,
        'pre-op': 1,
        'post-op': 1,
        'qid': 4,
        'seven times a day': 7,
        'tds': 3,
        'three times a week ': 3/7,
        'twice a week ': 2/7,
        'weekly': 1/7
        # Add any other mappings you need
    }
    return mapping.get(freq, None)

def cleaning(df):
    
    # remove empty rows
    df.dropna(how = 'all', inplace = True)
    df.reset_index(drop = True, inplace = True)

    #select columns
    df = df[['MRN','VISIT_ID','PATIENT_NAME','AGE',
             'ACTUAL_WEIGHT',
             'HEIGHT',
             'EGFR',
             'ORDER_PLACED_DATE',
             'ORDER_NAME',
             'ORDER_GENERIC',
             'MEDICAL_SERVICE',
             'DOSE',
             'VOLUME_DOSE',
             'RX_ROUTE',
             'FREQUENCY',
             'ATTENDING_MEDICAL_OFFICER',
             'ORDERING_PROVIDER',
             'ADMIT_DATE',
             'DISCHARGE_DATE',
             'ORDER_STATUS',
             'START_DTTM',
             'STOP_DTTM',
             'LOCATION_OF_PATIENT_AT_THE_TIME_OF_ORDER',
             'AMS_INDICATION']]
    # remove dummy patient
    df = df[df['ORDERING_PROVIDER'] != 'Test, Card']
    df.drop(columns = 'ORDERING_PROVIDER', inplace = True)

    # extract number
    df = extract_num(df)

    # depend on your need, two options to process EGFR, remember to comment out the other 
    df["EGFR"] = greater_than_90(df["EGFR"])
    #df["EGFR"] = convert_to_91(df["EGFR"])

    # extract first word to fill in order_generic
    df = fill_generic(df)

    # drop missing value in location of patient
    df.dropna(subset = ['LOCATION_OF_PATIENT_AT_THE_TIME_OF_ORDER'], inplace = True)

    # remove rows that are shifted
    df = df[~df['ORDER_GENERIC'].str.contains(r'\d')]

    # convert datetime column from string to datetime type
    df = date_processing(df)
    
    # fill in missing dttm with order palced time if frequency is once
    df = fill_dttm(df)

    # remove medical officer and senior medical officer
    df['ATTENDING_MEDICAL_OFFICER'] = df['ATTENDING_MEDICAL_OFFICER'].str.rsplit('(',n = 1).str[0]
    
    # transforming frequency to numeric values
    df = frequency_transform(df)
    
    return df

def update(df, df1):

    # extract rows in exisitng file which have same MRN as in the new file
    extract = df[df['MRN'].isin(df1['MRN'])]
    the_rest = df[~df['MRN'].isin(df1['MRN'])]
    
    # join the extract file with new file and remove duplicate
    extract_df1 = pd.concat([extract,df1])
    extract_df1 = extract_df1.drop_duplicates()
    
    # fill in discharge date, weight, height
    extract_df1['DISCHARGE_DATE'] = extract_df1.groupby(['MRN','VISIT_ID'])['DISCHARGE_DATE'].transform(lambda x: x.bfill())
    extract_df1[['ACTUAL_WEIGHT', 'HEIGHT']] = extract_df1.groupby(['MRN'])[['ACTUAL_WEIGHT', 'HEIGHT']].transform(lambda x: x.ffill())
    
    # join all together
    output = pd.concat([extract_df1,the_rest])
    output = output.drop_duplicates()

    return output